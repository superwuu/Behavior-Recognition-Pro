#!/bin/bash
RECORD=1106_joint_guo_rot
WORKDIR=work_dir/$RECORD
MODELNAME=runs/$RECORD

CONFIG=./config/guosai/train_joint_rot.yaml  
START_EPOCH=50
EPOCH_NUM=60
BATCH_SIZE=53  #56
WARM_UP=5
SEED=777

python3 main.py --config $CONFIG --work-dir $WORKDIR -model_saved_name $MODELNAME --device 1 --batch-size $BATCH_SIZE --test-batch-size $BATCH_SIZE --warm_up_epoch $WARM_UP --only_train_epoch $EPOCH_NUM --seed $SEED

