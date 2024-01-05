#!/bin/bash
# custom config
DATA=data
TRAINER=LaFTer
CFG=vit_b32 #'GeoRSCLIP' or 'vit_b32' or 'RemoteCLIP' or 'EVA02_CLIP_B_psz16_s8B'
dset="$1"
txt_cls=lafter
nos_epochs=50
CUDA_VISIBLE_DEVICES=0 python LaFTer.py \
--root ${DATA} \
--trainer ${TRAINER} \
--dataset-config-file configs/datasets/"${dset}".yaml \
--config-file configs/trainers/text_cls/${CFG}.yaml \
--output-dir output_3/${TRAINER}/${CFG}/"${dset}_${nos_epochs}/single_PL_LN_Frozen" \
--lr 0.0005 \
--epochs ${nos_epochs} \
--batch_size 50 \
--txt_cls ${txt_cls} \
--text_only \
--ln_frozen