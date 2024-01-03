#!/bin/bash
# custom config
DATA=data
TRAINER=LaFTer
CFG=GeoRSCLIP
dset="$1"
txt_cls=lafter
nos_epochs=50
# v_encoder=georsclip-2epochs
# v_encoder="georsclip-${nos_epochs}-refined_pl"
# v_encoder="clip-${nos_epochs}-avg_pl"
# v_encoder="clip-${nos_epochs}epochs_alpha_7.5"
# v_encoder="clip-${nos_epochs}_epochs_avg_pl"
v_encoder="clip-${nos_epochs}_epochs_conf_frozen_arch_fp32_test"
CUDA_VISIBLE_DEVICES=0 python LaFTer.py \
--root ${DATA} \
--trainer ${TRAINER} \
--dataset-config-file configs/datasets/"${dset}".yaml \
--config-file configs/trainers/text_cls/${CFG}.yaml \
--output-dir output/${TRAINER}/${CFG}/"${dset}/${v_encoder}" \
--lr 0.0005 \
--epochs ${nos_epochs} \
--txt_cls ${txt_cls} \
--batch_size 50
