#!/bin/bash
RECORD=res_phase2_joint_none

WORKDIR=work_dir/$RECORD
MODELNAME=runs/$RECORD

CONFIG=/media/sdd/robot/TE-GCN/config/phase2/test_joint_none.yaml

WEIGHTS=/media/sdd/robot/TE-GCN/runs/phase2_train_joint_none-43-22968.pt


BATCH_SIZE=256
python3  main.py  --config $CONFIG --work-dir $WORKDIR -model_saved_name $MODELNAME --device 2  --batch-size $BATCH_SIZE --test-batch-size $BATCH_SIZE --weights $WEIGHTS
# python3  main.py  --config $CONFIG --work-dir $WORKDIR -model_saved_name $MODELNAME --device 0 1 2 3 --batch-size $BATCH_SIZE --test-batch-size $BATCH_SIZE --weights $WEIGHTS

# python3 -m debugpy --listen 5678 --wait-for-client main.py  --config $CONFIG --work-dir $WORKDIR -model_saved_name $MODELNAME --device 0 1 2 3 --batch-size $BATCH_SIZE --test-batch-size $BATCH_SIZE --weights $WEIGHTS
