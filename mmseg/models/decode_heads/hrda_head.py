# Obtained from: https://github.com/lhoyer/HRDA
# Modifications:
# - Add return_logits flag
# - Update debug_output
# ---------------------------------------------------------------
# Copyright (c) 2022 ETH Zurich, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------
# MaskTwins
# - Add complementary masking
# - Modify the crop box
# - Add dual-form losses


from copy import deepcopy

import torch
from torch.nn import functional as F
import os
import random

import numpy as np
import tifffile as tiff
from ...core import add_prefix
from ...ops import resize as _resize
from .. import builder
from ..builder import HEADS
from ..segmentors.hrda_encoder_decoder import crop
from .decode_head import BaseDecodeHead


def scale_box(box, scale):
    y1, y2, x1, x2 = box
    # assert y1 % scale == 0
    # assert y2 % scale == 0
    # assert x1 % scale == 0
    # assert x2 % scale == 0
    y1 = int(y1 / scale)
    y2 = int(y2 / scale)
    x1 = int(x1 / scale)
    x2 = int(x2 / scale)
    return y1, y2, x1, x2


@HEADS.register_module()
class HRDAHead(BaseDecodeHead):

    def __init__(self,
                 single_scale_head,
                 lr_loss_weight=0,
                 hr_loss_weight=0,
                 scales=[1],
                 attention_embed_dim=256,
                 attention_classwise=True,
                 enable_hr_crop=False,
                 hr_slide_inference=True,
                 fixed_attention=None,
                 debug_output_attention=False,
                 **kwargs):
        head_cfg = deepcopy(kwargs)
        attn_cfg = deepcopy(kwargs)
        if single_scale_head == 'DAFormerHead':
            attn_cfg['channels'] = attention_embed_dim
            attn_cfg['decoder_params']['embed_dims'] = attention_embed_dim
            if attn_cfg['decoder_params']['fusion_cfg']['type'] == 'aspp':
                attn_cfg['decoder_params']['fusion_cfg'] = dict(
                    type='conv',
                    kernel_size=1,
                    act_cfg=dict(type='ReLU'),
                    norm_cfg=attn_cfg['decoder_params']['fusion_cfg']
                    ['norm_cfg'])
            kwargs['init_cfg'] = None
            kwargs['input_transform'] = 'multiple_select'
            self.os = 4
        elif single_scale_head == 'DLV2Head':
            kwargs['init_cfg'] = None
            kwargs.pop('dilations')
            kwargs['channels'] = 1
            self.os = 8
        else:
            raise NotImplementedError(single_scale_head)
        super(HRDAHead, self).__init__(**kwargs)
        del self.conv_seg
        del self.dropout

        head_cfg['type'] = single_scale_head
        self.head = builder.build_head(head_cfg)

        attn_cfg['type'] = single_scale_head
        if not attention_classwise:
            attn_cfg['num_classes'] = 1
        if fixed_attention is None:
            self.scale_attention = builder.build_head(attn_cfg)
        else:
            self.scale_attention = None
            self.fixed_attention = fixed_attention
        self.lr_loss_weight = lr_loss_weight
        self.hr_loss_weight = hr_loss_weight
        self.scales = scales
        self.enable_hr_crop = enable_hr_crop
        self.hr_crop_box = dict()
        self.hr_slide_inference = hr_slide_inference
        self.debug_output_attention = debug_output_attention

    def set_hr_crop_box(self, boxes,count=None):
        if count==1:
            self.hr_crop_box['1'] = boxes
        elif count==2:
            self.hr_crop_box['2'] = boxes
        else:
            self.hr_crop_box = boxes

    def hr_crop_slice(self, scale,count=None):
        if count==1:
            crop_y1, crop_y2, crop_x1, crop_x2 = scale_box(self.hr_crop_box['1'], scale)
        elif count==2:
            crop_y1, crop_y2, crop_x1, crop_x2 = scale_box(self.hr_crop_box['2'], scale)
        else:
            crop_y1, crop_y2, crop_x1, crop_x2 = scale_box(self.hr_crop_box, scale)
        return slice(crop_y1, crop_y2), slice(crop_x1, crop_x2)

    def resize(self, input, scale_factor):
        return _resize(
            input=input,
            scale_factor=scale_factor,
            mode='bilinear',
            align_corners=self.align_corners)

    def decode_hr(self, inp, bs):
        if isinstance(inp, dict) and 'boxes' in inp.keys():
            features = inp['features']
            boxes = inp['boxes']
            dev = features[0][0].device
            h_img, w_img = 0, 0
            for i in range(len(boxes)):
                boxes[i] = scale_box(boxes[i], self.os)
                y1, y2, x1, x2 = boxes[i]
                if h_img < y2:
                    h_img = y2
                if w_img < x2:
                    w_img = x2
            preds = torch.zeros((bs, self.num_classes, h_img, w_img),
                                device=dev)
            count_mat = torch.zeros((bs, 1, h_img, w_img), device=dev)

            crop_seg_logits = self.head(features)
            for i in range(len(boxes)):
                y1, y2, x1, x2 = boxes[i]
                crop_seg_logit = crop_seg_logits[i * bs:(i + 1) * bs]
                preds += F.pad(crop_seg_logit,
                               (int(x1), int(preds.shape[3] - x2), int(y1),
                                int(preds.shape[2] - y2)))

                count_mat[:, :, y1:y2, x1:x2] += 1

            assert (count_mat == 0).sum() == 0
            preds = preds / count_mat
            return preds
        else:
            return self.head(inp)

    def get_scale_attention(self, inp):
        if self.scale_attention is not None:
            att = torch.sigmoid(self.scale_attention(inp))
        else:
            att = self.fixed_attention
        return att

    def forward(self, inputs,count=None,use_feat=False):
        assert len(inputs) == 2
        hr_inp = inputs[1]
        hr_scale = self.scales[1]
        lr_inp = inputs[0]
        lr_sc_att_inp = inputs[0]  # separate var necessary for stack hr_fusion
        lr_scale = self.scales[0]
        batch_size = lr_inp[0].shape[0]
        assert lr_scale <= hr_scale

        has_crop = len(self.hr_crop_box)!=0

        lr_seg = self.head(lr_inp)
 
        hr_seg = self.decode_hr(hr_inp, batch_size) 


        att = self.get_scale_attention(lr_sc_att_inp)
        if has_crop:
            mask = lr_seg.new_zeros([lr_seg.shape[0], 1, *lr_seg.shape[2:]])
            sc_os = self.os / lr_scale
            slc = self.hr_crop_slice(sc_os,count=count)
            mask[:, :, slc[0], slc[1]] = 1
            att = att * mask

        lr_seg = (1 - att) * lr_seg

        up_lr_seg = self.resize(lr_seg, hr_scale / lr_scale)
        if torch.is_tensor(att):
            att = self.resize(att, hr_scale / lr_scale)

        if has_crop:
            hr_seg_inserted = torch.zeros_like(up_lr_seg)
            slc = self.hr_crop_slice(self.os,count=count)
            hr_seg_inserted[:, :, slc[0], slc[1]] = hr_seg
        else:
            hr_seg_inserted = hr_seg


        fused_seg = att * hr_seg_inserted + up_lr_seg

        if self.debug_output_attention:
            att = torch.sum(
                att * torch.softmax(fused_seg, dim=1), dim=1, keepdim=True)
            return att, None, None

        if self.debug:
            self.debug_output.update({
                'High Res':
                torch.max(hr_seg, dim=1)[1].detach().cpu().numpy(),
                'High Res Inserted':
                torch.max(hr_seg_inserted, dim=1)[1].detach().cpu().numpy(),
                'Low Res':
                torch.max(lr_seg, dim=1)[1].detach().cpu().numpy(),
                'Fused':
                torch.max(fused_seg, dim=1)[1].detach().cpu().numpy(),
            })
            if torch.is_tensor(att):
                self.debug_output['Attention'] = torch.sum(
                    att * torch.softmax(fused_seg, dim=1), dim=1,
                    keepdim=True).detach().cpu().numpy()

        return fused_seg, lr_seg, hr_seg

    def reset_crop(self):
        del self.hr_crop_box
        self.hr_crop_box = dict()

    def forward_train(self,
                      inputs,
                      img_metas,
                      gt_semantic_seg,
                      train_cfg,
                      seg_weight=None,
                      return_logits=False,
                      if2mask=False,
                      cm_weight=1,
                      feat=False):
        """Forward function for training."""
        if self.enable_hr_crop:
            assert len(self.hr_crop_box)!=0
        if not if2mask:
            seg_logits = self.forward(inputs)
            losses = self.losses(seg_logits, gt_semantic_seg, seg_weight)
            if return_logits:
                losses['logits'] = seg_logits
            self.reset_crop()
            return losses
        elif feat:
            losses = dict()

            seg_logits = self.forward(inputs,use_feat=feat)
            
        else:
            
            losses = dict()
            inputs1=inputs['1']
            inputs2=inputs['2']

            seg_logits1 = self.forward(inputs1,count=1)
            losses1 = self.losses(seg_logits1, gt_semantic_seg, seg_weight,count=1)

            seg_logits2 = self.forward(inputs2,count=2)

            losses2 = self.losses(seg_logits2, gt_semantic_seg, seg_weight,count=2)

            losses['loss_seg']=0.5*(losses1['loss_seg']+losses2['loss_seg'])
            
            losses['hr.loss_seg']=0.5*(losses1['hr.loss_seg']+losses2['hr.loss_seg'])

            losses['m1.acc_seg'] = losses1['acc_seg']
            losses['m2.acc_seg'] = losses2['acc_seg']

            losses['m1.hr.acc_seg']=losses1['hr.acc_seg']
            losses['m2.hr.acc_seg']=losses2['hr.acc_seg']


            seg_cm_stack={'1':seg_logits1,'2':seg_logits2}
            
            loss_cm = self.losses(seg_cm_stack,gt_semantic_seg)
            
            loss_cm['loss_seg'] *= cm_weight
            loss_cm=add_prefix(loss_cm, 'cm')
            losses.update(loss_cm)
     

            if return_logits:
                losses['logits'] = 0.5*(seg_logits1+seg_logits2)
            self.reset_crop()
            return losses

    def forward_test(self, inputs, img_metas, test_cfg):
        """Forward function for testing, only ``fused_seg`` is used."""
        return self.forward(inputs)[0]

    def get_pseudo_label(self, logits):

        ema_softmax = torch.softmax(logits.detach(), dim=1)
        _, pseudo_label = torch.max(ema_softmax, dim=1)

        return pseudo_label

    def losses(self, seg_logit, seg_label, seg_weight=None, count=3):
        """Compute losses."""
        if isinstance(seg_logit,dict):
            loss = dict()
            fused_seg1, lr_seg1, hr_seg1 = seg_logit['1']  
            fused_seg2, lr_seg2, hr_seg2 = seg_logit['2']

            mse_loss1 = torch.nn.functional.mse_loss(fused_seg1, fused_seg2)

            loss['loss_seg'] = mse_loss1

            if self.debug:
                self.debug_output['GT'] = \
                    seg_label.squeeze(1).detach().cpu().numpy()
                # Remove debug output from cross entropy loss
                self.debug_output.pop('Seg. Pred.', None)
                self.debug_output.pop('Seg. GT', None)
            return loss
        else:
            fused_seg, lr_seg, hr_seg = seg_logit

            loss = super(HRDAHead, self).losses(fused_seg, seg_label, seg_weight)
            if self.hr_loss_weight == 0 and self.lr_loss_weight == 0:
                return loss

            if self.lr_loss_weight > 0:
                loss.update(
                    add_prefix(
                        super(HRDAHead, self).losses(lr_seg, seg_label,
                                                    seg_weight), 'lr'))
            if self.hr_loss_weight > 0 and self.enable_hr_crop:
                if count==1:
                    boxes=self.hr_crop_box['1']
                elif count==2:
                    boxes=self.hr_crop_box['2']
                else:
                    boxes=self.hr_crop_box
                cropped_seg_label = crop(seg_label, boxes)
                if seg_weight is not None:
                    cropped_seg_weight = crop(seg_weight, boxes)
                else:
                    cropped_seg_weight = seg_weight
                if self.debug:
                    self.debug_output['Cropped GT'] = \
                        cropped_seg_label.squeeze(1).detach().cpu().numpy()
                loss.update(
                    add_prefix(
                        super(HRDAHead, self).losses(hr_seg, cropped_seg_label,
                                                    cropped_seg_weight), 'hr'))
            elif self.hr_loss_weight > 0:
                loss.update(
                    add_prefix(
                        super(HRDAHead, self).losses(hr_seg, seg_label,
                                                    seg_weight), 'hr'))
            loss['loss_seg'] *= (1 - self.lr_loss_weight - self.hr_loss_weight)
            if self.lr_loss_weight > 0:
                loss['lr.loss_seg'] *= self.lr_loss_weight
            if self.hr_loss_weight > 0:
                loss['hr.loss_seg'] *= self.hr_loss_weight

            if self.debug:
                self.debug_output['GT'] = \
                    seg_label.squeeze(1).detach().cpu().numpy()
                # Remove debug output from cross entropy loss
                self.debug_output.pop('Seg. Pred.', None)
                self.debug_output.pop('Seg. GT', None)

            return loss
