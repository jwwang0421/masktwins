_base_ = [
    '../_base_/default_runtime.py',
    '../_base_/models/daformer_sepaspp_mitb5.py',
    '../_base_/datasets/uda_synthiaHR_to_cityscapesHR_1024x1024.py',
    '../_base_/uda/dacs_a999_fdthings.py',
    '../_base_/schedules/adamw.py',
    '../_base_/schedules/poly10warm.py'
]

seed = 1
# Model Hyperparameters
model = dict(
    type='HRDAEncoderDecoder',
    decode_head=dict(
        type='HRDAHead',
        single_scale_head='DAFormerHead',
        attention_classwise=True,
        hr_loss_weight=0.1),
    scales=[1, 0.5],
    hr_crop_size=(512, 512),
    feature_scale=0.5,
    crop_coord_divisible=8,
    hr_slide_inference=True,
    test_cfg=dict(
        mode='slide',
        batched_slide=True,
        stride=[512, 512],
        crop_size=[1024, 1024]))
data = dict(
    train=dict(
        rare_class_sampling=dict(
            min_pixels=3000, class_temp=0.01, min_crop_ratio=2.0),
        target=dict(crop_pseudo_margins=[30, 240, 30, 30]),
    ),
    workers_per_gpu=1,
    # Batch size: not advised to set to 1
    samples_per_gpu=2,
)
uda = dict(
    mask_mode='separatetrgaug',
    mask_alpha='same',
    mask_pseudo_threshold='same',
    mask_lambda=1,
    # - TwoMask: If True, use two masks for each image
    # - CMask: If True, use complementary masking; else use random masking
    # - cm_weight: Weight for the complementary mask
    # - feature_level: If True, use feature-level masking
    # - mask ratio: Default is 0.5
    TwoMask=True, 
    CMask=True,
    cm_weight=0.01,
    feature_level=False,
    mask_generator=dict(
        type='block', mask_ratio=[0.5], mask_block_size=64, _delete_=True))
# Optimizer Hyperparameters
optimizer_config = None
optimizer = dict(
    lr=6e-05,
    paramwise_cfg=dict(
        custom_keys=dict(
            head=dict(lr_mult=10.0),
            pos_block=dict(decay_mult=0.0),
            norm=dict(decay_mult=0.0))))
n_gpus = 2
gpu_model = 'NVIDIATITANRTX'
runner = dict(type='IterBasedRunner', max_iters=40000)
# Logging Configuration
# Modify EvalHook
# - save_n: save the best n checkpoints
checkpoint_config = dict(by_epoch=False, interval=4000, max_keep_ckpts=1)
evaluation = dict(interval=4000, metric='mIoU',save_best='mIoU',save_n=5)
# Meta Information for Result Analysis
name = 'syn2cs_cm'
exp = 'base'
name_dataset = 'synthiaHR_to_cityscapesHR_1024x1024'
name_architecture = 'hrda1-512-0.1_daformer_sepaspp_sl_mitb5'
name_encoder = 'mitb5'
name_decoder = 'hrda1-512-0.1_daformer_sepaspp_sl'
name_uda = 'dacs_a999_fdthings_rcs0.01-2.0_cpl2_m64-0.7-spta'
name_opt = 'adamw_6e-05_pmTrue_poly10warm_1x2_40k'

