#!/bin/bash
########## setup project directory ##########
CODE_DIR=`realpath $(dirname "$0")/..`
echo "Locate the project folder at ${CODE_DIR}"
cd ${CODE_DIR}

######### check number of args ##########
HELP="Usage example: GPU=0,1,2,3 bash $0 <path> <AE config> <Flow config> [mode: e.g. 11]"
if [ -z $1 ]; then
    echo "Experiment saving path missing. ${HELP}"
    exit 1;
else
    SAVE_PATH=$1
fi
if [ -z $2 ]; then
    echo "Autoencoder config missing. ${HELP}"
    exit 1;
else
    AECONFIG=$2
fi
if [ -z $3 ]; then
    echo "Flow config missing. ${HELP}"
    exit 1;
else
    FLOWCONFIG=$3
fi
if [ -z $4 ]; then
    MODE=11
else
    MODE=$4
fi

echo "Mode: $MODE, [train AE] / [train Flow]"
TRAIN_AE_FLAG=${MODE:0:1}
TRAIN_FLOW_FLAG=${MODE:1:1}

SUFFIX=`basename ${SAVE_PATH}`

AE_SAVE_DIR=$SAVE_PATH/AE_${SUFFIX}
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
    latest_ckpt=`cat ${save_dir}/version_0/checkpoint/topk_map.txt | head -n 1 | awk -F " " '{print $2}'`
    echo $latest_ckpt
}


organize_folder() {
    local save_dir=$1
    if [ -e $save_dir/old ]; then   # clean old checkpoints
        rm -r $save_dir/old
    fi
    mv ${save_dir}/version_0 ${save_dir}/old
    echo ${save_dir}/old
}


if [[ ! -e $SAVE_PATH ]]; then
    mkdir -p $SAVE_PATH
fi

if [[ -e $AE_SAVE_DIR ]] && [ "$TRAIN_AE_FLAG" = "1" ]; then
    echo "Directory ${AE_SAVE_DIR} exisits! But training flag is 1!"
    LATEST_AE_CKPT=$(get_latest_checkpoint $AE_SAVE_DIR)
    if [ -n "$LATEST_AE_CKPT" ]; then
        echo "Found Autoencoder checkpoint: $LATEST_AE_CKPT"
        AE_UPDATE_MAX_EPOCH=$(update_max_epoch $AECONFIG $LATEST_AE_CKPT)
        echo "Updated max epoch to $AE_UPDATE_MAX_EPOCH"
        echo "Moved old checkpoints to $(organize_folder $AE_SAVE_DIR)"
        LATEST_AE_CKPT=${LATEST_AE_CKPT/version_0/old}
        AE_CONTINUE_ARGS="--load_ckpt $LATEST_AE_CKPT --trainer.config.max_epoch $AE_UPDATE_MAX_EPOCH"
    else
        echo "No checkpoint found. Training will start from scratch."
        AE_CONTINUE_ARGS=""
    fi
fi

if [[ -e $FLOW_SAVE_DIR ]] && [ "$TRAIN_FLOW_FLAG" = "1" ]; then
    echo "Directory ${FLOW_SAVE_DIR} exisits! But training flag is 1!"
    LATEST_FLOW_CKPT=$(get_latest_checkpoint $FLOW_SAVE_DIR)
    if [ -n "$LATEST_FLOW_CKPT" ]; then
        echo "Found Flow checkpoint: $LATEST_FLOW_CKPT"
        FLOW_UPDATE_MAX_EPOCH=$(update_max_epoch $FLOWCONFIG $LATEST_FLOW_CKPT)
        echo "Updated max epoch to $FLOW_UPDATE_MAX_EPOCH"
        echo "Moved old checkpoints to $(organize_folder $FLOW_SAVE_DIR)"
        LATEST_FLOW_CKPT=${LATEST_FLOW_CKPT/version_0/old}
        FLOW_CONTINUE_ARGS="--load_ckpt $LATEST_FLOW_CKPT --trainer.config.max_epoch $FLOW_UPDATE_MAX_EPOCH"
    else
        echo "No checkpoint found. Training will start from scratch."
        FLOW_CONTINUE_ARGS=""
    fi
fi

########## train autoencoder ##########
echo "Training Autoencoder with config $AECONFIG:" > $OUTLOG
cat $AECONFIG >> $OUTLOG
echo "Overwriting args $ARGS1"
if [ "$TRAIN_AE_FLAG" = "1" ]; then
    if  [ -z $AE_UPDATE_MAX_EPOCH ] || [ $AE_UPDATE_MAX_EPOCH -gt 0 ]; then
        bash scripts/train.sh $AECONFIG --trainer.config.save_dir=$AE_SAVE_DIR $ARGS1 $AE_CONTINUE_ARGS
        if [ $? -eq 0 ]; then
            echo "Succeeded in training AutoEncoder"
        else
            echo "Failed to train AutoEncoder"
            exit 1;
        fi
    else
        echo "AutoEncoder already finished training"
    fi
fi

########## train flow ##########
echo "Training Flow with config $FLOWCONFIG:" >> $OUTLOG
cat $FLOWCONFIG >> $OUTLOG
echo "Overwriting args $ARGS2"
AE_CKPT=`cat ${AE_SAVE_DIR}/version_0/checkpoint/topk_map.txt | head -n 1 | awk -F " " '{print $2}'`
echo "Using Autoencoder checkpoint: ${AE_CKPT}" >> $OUTLOG
if [ "$TRAIN_FLOW_FLAG" = "1" ]; then
    if  [ -z $FLOW_UPDATE_MAX_EPOCH ] || [ $FLOW_UPDATE_MAX_EPOCH -gt 0 ]; then
        bash scripts/train.sh $FLOWCONFIG --trainer.config.save_dir=$FLOW_SAVE_DIR --model.autoencoder_ckpt=$AE_CKPT $ARGS2 $FLOW_CONTINUE_ARGS
    else
        echo "Flow already finished training"
    fi
fi