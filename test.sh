# Obtained from: https://github.com/lhoyer/MIC


TEST_ROOT=$1
CONFIG_FILE="${TEST_ROOT}/*.py"  # or .json for old configs
CHECKPOINT_FILE="${TEST_ROOT}/iter_40000.pth" 
SHOW_DIR="$./show"
echo 'Config File:' $CONFIG_FILE
echo 'Checkpoint File:' $CHECKPOINT_FILE
echo 'Predictions Output Directory:' $SHOW_DIR
python -m tools.test ${CONFIG_FILE} ${CHECKPOINT_FILE} --eval mIoU --show-dir ${SHOW_DIR} --opacity 1
