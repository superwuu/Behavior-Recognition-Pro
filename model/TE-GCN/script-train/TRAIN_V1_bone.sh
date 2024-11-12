#!/bin/bash
RECORD=phase2_train_bone
WORKDIR=work_dir/$RECORD
MODELNAME=runs/$RECORD

# CONFIG=./config/uav-cross-subjectv1/train_joint_motion.yaml
CONFIG=/media/sdd/robot/TE-GCN/config/phase2/train_bone.yaml
# CONFIG=/media/sdd/robot/TE-GCN/config/uav-cross-subjectv1/train.yaml

START_EPOCH=20
EPOCH_NUM=50
BATCH_SIZE=64
WARM_UP=5
SEED=777

python3 main.py --config $CONFIG --work-dir $WORKDIR -model_saved_name $MODELNAME --device 1 2 --batch-size $BATCH_SIZE --test-batch-size $BATCH_SIZE --warm_up_epoch $WARM_UP --only_train_epoch $EPOCH_NUM --seed $SEED
