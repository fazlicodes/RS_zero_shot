#!/bin/bash
# custom config
DATA=data
TRAINER=LaFTer
CFG=vit_b32
dset="$1"
txt_cls=lafter
v_encoder=georsclip-50epochs
CUDA_VISIBLE_DEVICES=0 python LaFTer.py \
--root ${DATA} \
--trainer ${TRAINER} \
--dataset-config-file configs/datasets/"${dset}".yaml \
--config-file configs/trainers/text_cls/${CFG}.yaml \
--output-dir output/${TRAINER}/${CFG}/"${dset}/${v_encoder}" \
--lr 0.0005 \
--epochs 50 \
--txt_cls ${txt_cls}
