#!/bin/bash
# custom config
DATA=data
TRAINER=LaFTer
CFG=GeoRSCLIP #'GeoRSCLIP' or 'vit_b32' or 'RemoteCLIP' or 'EVA02_CLIP_B_psz16_s8B'
dset="$1"
txt_cls=lafter
nos_epochs=50
bws="avg" # 'conf_alpha' or 'fixed_alpha_{value}' with alpha rate or 'avg'
CUDA_VISIBLE_DEVICES=0 python LaFTer.py \
--root ${DATA} \
--trainer ${TRAINER} \
--dataset-config-file configs/datasets/"${dset}".yaml \
--config-file configs/trainers/text_cls/${CFG}.yaml \
--output-dir output_3/test-delete \
--lr 0.0005 \
--epochs ${nos_epochs} \
--txt_cls ${txt_cls} \
--bws ${bws} \
--ln_frozen \
--batch_size 50