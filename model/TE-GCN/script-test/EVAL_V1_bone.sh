#!/bin/bash

RECORD=res_phase2_bone

WORKDIR=work_dir/$RECORD
MODELNAME=runs/$RECORD

CONFIG=/media/sdd/robot/TE-GCN/config/phase2/test_bone.yaml


WEIGHTS=/media/sdd/robot/TE-GCN/runs/phase2_train_bone-45-12006.pt



BATCH_SIZE=256
python3  main.py  --config $CONFIG --work-dir $WORKDIR -model_saved_name $MODELNAME --device 1  --batch-size $BATCH_SIZE --test-batch-size $BATCH_SIZE --weights $WEIGHTS
# python3  main.py  --config $CONFIG --work-dir $WORKDIR -model_saved_name $MODELNAME --device 0 1 2 3 --batch-size $BATCH_SIZE --test-batch-size $BATCH_SIZE --weights $WEIGHTS

# python3 -m debugpy --listen 5678 --wait-for-client main.py  --config $CONFIG --work-dir $WORKDIR -model_saved_name $MODELNAME --device 0 1 2 3 --batch-size $BATCH_SIZE --test-batch-size $BATCH_SIZE --weights $WEIGHTS
