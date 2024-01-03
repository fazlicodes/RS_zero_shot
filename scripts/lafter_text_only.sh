#!/bin/bash
# custom config
DATA=data
TRAINER=LaFTer
CFG=vit_b32
dset="$1"
txt_cls=lafter
nos_epochs=12
v_encoder="clip-${nos_epochs}_epochs_text_only_train_test_val_2"
CUDA_VISIBLE_DEVICES=0 python LaFTer.py \
--root ${DATA} \
--trainer ${TRAINER} \
--dataset-config-file configs/datasets/"${dset}".yaml \
--config-file configs/trainers/text_cls/${CFG}.yaml \
--output-dir output/${TRAINER}/${CFG}/"${dset}/${v_encoder}" \
--lr 0.0005 \
--epochs ${nos_epochs} \
--batch_size 50 \
--txt_cls ${txt_cls} \
--text_only


