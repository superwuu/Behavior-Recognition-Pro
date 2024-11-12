#!/bin/bash
RECORD=phase2_train_joint_none
WORKDIR=work_dir/$RECORD
MODELNAME=runs/$RECORD

CONFIG=/media/sdd/robot/TE-GCN/config/phase2/train_joint_none.yaml

# joint none 43.8
# joint 2 
# joint rot 

START_EPOCH=20
EPOCH_NUM=50
BATCH_SIZE=32
WARM_UP=5
SEED=777

python3 main.py --config $CONFIG --work-dir $WORKDIR -model_saved_name $MODELNAME --device 2 --batch-size $BATCH_SIZE --test-batch-size $BATCH_SIZE --warm_up_epoch $WARM_UP --only_train_epoch $EPOCH_NUM --seed $SEED


# sh ./scripts/phase2/TRAIN_V1_joint_none.sh