#!/bin/bash
# custom config
DATA=data
TRAINER=LaFTer
CFG=vit_b32 #'GeoRSCLIP' or 'vit_b32' or 'RemoteCLIP' or 'EVA02_CLIP_B_psz16_s8B'
dset="$1"
txt_cls=lafter
nos_epochs=50
svl_model_path=svl_adapter_models
bws="avg" # 'conf_alpha' or 'fixed_alpha_{value}' with alpha rate or 'avg'
CUDA_VISIBLE_DEVICES=2 python LaFTer.py \
--root ${DATA} \
--trainer ${TRAINER} \
--dataset-config-file configs/datasets/"${dset}".yaml \
--config-file configs/trainers/text_cls/${CFG}.yaml \
--output-dir output_svl_pl_LN_learnable_svlenc_learnable_wo_ZS/${TRAINER}/${CFG}/"${dset}"_"${nos_epochs}"/"${bws}" \
--lr 0.0005 \
--epochs ${nos_epochs} \
--txt_cls ${txt_cls} \
--bws ${bws} \
--batch_size 50 \
--svl_pl \
--svl_model_path ${svl_model_path}
# --ln_frozen \