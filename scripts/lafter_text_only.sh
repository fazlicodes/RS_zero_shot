#!/bin/bash
# custom config
DATA=data
TRAINER=LaFTer
CFG=vit_b32
dset="$1"
txt_cls=lafter
nos_epochs=50
v_encoder="georsclip-${nos_epochs}epochs_text_only"
CUDA_VISIBLE_DEVICES=0 python LaFTer.py \
--root ${DATA} \
--trainer ${TRAINER} \
--dataset-config-file configs/datasets/"${dset}".yaml \
--config-file configs/trainers/text_cls/${CFG}.yaml \
--output-dir output/${TRAINER}/${CFG}/"${dset}/${v_encoder}" \
--lr 0.0005 \
--epochs ${nos_epochs} \
--txt_cls ${txt_cls} \
--text_only

