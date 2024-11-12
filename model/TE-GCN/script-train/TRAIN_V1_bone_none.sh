#!/bin/bash
RECORD=phase2_train_bone_none
WORKDIR=work_dir/$RECORD
MODELNAME=runs/$RECORD

CONFIG=/media/sdd/robot/TE-GCN/config/phase2/train_bone_none.yaml

# none bone 42.2

# rot bone 43.65

# 2 42.30

START_EPOCH=20
EPOCH_NUM=50
BATCH_SIZE=32
WARM_UP=5
SEED=777

python3 main.py --config $CONFIG --work-dir $WORKDIR -model_saved_name $MODELNAME --device 0 --batch-size $BATCH_SIZE --test-batch-size $BATCH_SIZE --warm_up_epoch $WARM_UP --only_train_epoch $EPOCH_NUM --seed $SEED

# sh ./scripts/phase2/TRAIN_V1_bone_none.sh