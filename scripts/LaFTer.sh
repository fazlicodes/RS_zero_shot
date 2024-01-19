#!/bin/bash
# custom config
DATA=data
TRAINER=LaFTer
CFG=GeoRSCLIP #'GeoRSCLIP' or 'vit_b32' or 'RemoteCLIP' or 'EVA02_CLIP_B_psz16_s8B'
dset="$1"
txt_cls=lafter
nos_epochs=50
svl_model_path=svl_adapter_models
bws="avg" # 'conf_alpha' or 'fixed_alpha_{value}' with alpha rate or 'avg'
pl_technique="pl_text" #choices=['None','pl_text', 'pl_svl', 'pl_text_svl','svl_only']
CUDA_VISIBLE_DEVICES=0 python LaFTer.py \
--root ${DATA} \
--trainer ${TRAINER} \
--dataset ${dset} \
--dataset-config-file configs/datasets/"${dset}".yaml \
--config-file configs/trainers/text_cls/${CFG}.yaml \
--output-dir final_output/PL1_avg_lnfrozen/"${CFG}"_"${dset}"_"${nos_epochs}"_"${bws}" \
--lr 0.0005 \
--epochs ${nos_epochs} \
--txt_cls ${txt_cls} \
--batch_size 50 \
--bws ${bws} \
--pl_technique ${pl_technique} \
--ln_frozen
# --ve_unshared
# --text_only
# --svl_model_path ${svl_model_path}
# --desc_noise 0.1