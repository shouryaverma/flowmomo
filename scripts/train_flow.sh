#!/bin/bash
########## Rectified Flow Training Script ##########
# Simplified single-stage training for end-to-end rectified flow
# Usage: GPU=0,1,2,3 bash scripts/train_flow.sh <save_path> <flow_config>

########## setup project directory ##########
CODE_DIR=`realpath $(dirname "$0")/..`
echo "Locate the project folder at ${CODE_DIR}"
cd ${CODE_DIR}

######### check number of args ##########
HELP="Usage example: GPU=0,1,2,3 bash $0 <save_path> <flow_config>"
if [ -z $1 ]; then
    echo "Experiment saving path missing. ${HELP}"
    exit 1;
else
    SAVE_PATH=$1
fi
if [ -z $2 ]; then
    echo "Rectified Flow config missing. ${HELP}"
    exit 1;
else
    FLOWCONFIG=$2
fi

SUFFIX=`basename ${SAVE_PATH}`
FLOW_SAVE_DIR=$SAVE_PATH/Flow_${SUFFIX}
OUTLOG=$SAVE_PATH/output.log

########## Handle existing directories ##########
update_max_epoch() {
    local config_file=$1
    local ckpt_file=$2
    config_max_epoch=`cat $config_file | grep -E "max_epoch: [0-9]+" | grep -oE "[0-9]+"`
    current_epoch=`basename $ckpt_file | grep -oE "epoch[0-9]+" | grep -oE "[0-9]+"`
    echo $((config_max_epoch - current_epoch - 1))
}

get_latest_checkpoint() {
    local save_dir=$1
    if [ -f ${save_dir}/version_0/checkpoint/topk_map.txt ]; then
        latest_ckpt=`cat ${save_dir}/version_0/checkpoint/topk_map.txt | head -n 1 | awk -F " " '{print $2}'`
        echo $latest_ckpt
    else
        echo ""
    fi
}

organize_folder() {
    local save_dir=$1
    if [ -e $save_dir/old ]; then   # clean old checkpoints
        rm -r $save_dir/old
    fi
    if [ -e ${save_dir}/version_0 ]; then
        mv ${save_dir}/version_0 ${save_dir}/old
        echo ${save_dir}/old
    fi
}

# Create save directory
if [[ ! -e $SAVE_PATH ]]; then
    mkdir -p $SAVE_PATH
fi

# Handle existing flow model checkpoints
FLOW_CONTINUE_ARGS=""
if [[ -e $FLOW_SAVE_DIR ]]; then
    echo "Directory ${FLOW_SAVE_DIR} exists! Checking for existing checkpoints..."
    LATEST_FLOW_CKPT=$(get_latest_checkpoint $FLOW_SAVE_DIR)
    if [ -n "$LATEST_FLOW_CKPT" ]; then
        echo "Found Rectified Flow checkpoint: $LATEST_FLOW_CKPT"
        FLOW_UPDATE_MAX_EPOCH=$(update_max_epoch $FLOWCONFIG $LATEST_FLOW_CKPT)
        echo "Updated max epoch to $FLOW_UPDATE_MAX_EPOCH"
        echo "Moved old checkpoints to $(organize_folder $FLOW_SAVE_DIR)"
        LATEST_FLOW_CKPT=${LATEST_FLOW_CKPT/version_0/old}
        FLOW_CONTINUE_ARGS="--load_ckpt $LATEST_FLOW_CKPT --trainer.config.max_epoch $FLOW_UPDATE_MAX_EPOCH"
    else
        echo "No checkpoint found. Training will start from scratch."
    fi
fi

########## train rectified flow ##########
echo "Training Rectified Flow with config $FLOWCONFIG:" > $OUTLOG
cat $FLOWCONFIG >> $OUTLOG
echo "Additional args: $FLOW_CONTINUE_ARGS"

# Extract any additional arguments beyond the first two (save_path and config)
shift 2  # Remove the first two arguments (save_path and flow_config)
EXTRA_ARGS="$@"

if [ -z $FLOW_UPDATE_MAX_EPOCH ] || [ $FLOW_UPDATE_MAX_EPOCH -gt 0 ]; then
    echo "Starting Rectified Flow training..."
    bash scripts/train.sh $FLOWCONFIG --trainer.config.save_dir=$FLOW_SAVE_DIR $FLOW_CONTINUE_ARGS $EXTRA_ARGS
    
    if [ $? -eq 0 ]; then
        echo "Successfully completed Rectified Flow training!"
        echo "Model saved at: $FLOW_SAVE_DIR"
        
        # Get final checkpoint for future use
        FINAL_CKPT=$(get_latest_checkpoint $FLOW_SAVE_DIR)
        echo "Final checkpoint: $FINAL_CKPT" >> $OUTLOG
        echo "Training completed successfully!" >> $OUTLOG
    else
        echo "Failed to train Rectified Flow"
        echo "Training failed!" >> $OUTLOG
        exit 1
    fi
else
    echo "Rectified Flow already finished training"
    echo "Training already completed!" >> $OUTLOG
fi

echo "Training log saved at: $OUTLOG"