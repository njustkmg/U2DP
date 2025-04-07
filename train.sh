id="aoanet_ucm_0.2"
if [ ! -f log/log_$id/infos_$id.pkl ]; then
start_from=""
else
start_from="--start_from log/log_$id"
fi
python train.py --id $id \
    --caption_model aoa \
    --refine 1 \
    --refine_aoa 1 \
    --use_ff 0 \
    --decoder_type AoA \
    --alpha 0.27 \
    --beta 0.24 \
    --lambda1 0.2 \
    --gamma 15 \
    --use_multi_head 2 \
    --num_heads 8 \
    --multi_head_scale 1 \
    --mean_feats 1 \
    --ctx_drop 1 \
    --dropout_aoa 0.3 \
    --label_smoothing 0.2 \
    --input_json ./dataset/dataset_bu/UCM_captions/0.2/ucmtalk.json \
    --input_label_h5 ./dataset/dataset_bu/UCM_captions/0.2/ucmtalk_label.h5 \
    --input_fc_dir  ./dataset/dataset_bu/UCM_captions/ucm_bu_fc \
    --input_att_dir  ./dataset/dataset_bu/UCM_captions/ucm_bu_att \
    --input_box_dir  ./dataset/dataset_bu/UCM_captions/ucm_bu_box \
    --clip_feat_path ./dataset/clip_feature/ucm_224 \
    --seq_per_img 5 \
    --batch_size 10 \
    --beam_size 1 \
    --learning_rate 2e-4 \
    --num_layers 2 \
    --input_encoding_size 1024 \
    --rnn_size 1024 \
    --learning_rate_decay_start 0 \
    --scheduled_sampling_start 0 \
    --checkpoint_path log/log_$id  \
    $start_from \
    --save_checkpoint_every 50 \
    --language_eval 1 \
    --val_images_use -1 \
    --max_epochs 100 \
    --scheduled_sampling_increase_every 5 \
    --scheduled_sampling_max_prob 0.5 \
    --learning_rate_decay_every 3 \
> "./log/result_$id.txt"


python train.py --id $id \
    --caption_model aoa \
    --refine 1 \
    --refine_aoa 1 \
    --use_ff 0 \
    --decoder_type AoA \
    --use_multi_head 2 \
    --num_heads 8 \
    --multi_head_scale 1 \
    --mean_feats 1 \
    --ctx_drop 1 \
    --dropout_aoa 0.3 \
    --input_json ./dataset/dataset_bu/UCM_captions/0.2/ucmtalk.json \
    --input_label_h5 ./dataset/dataset_bu/UCM_captions/0.2/ucmtalk_label.h5 \
    --input_fc_dir  ./dataset/dataset_bu/UCM_captions/ucm_bu_fc \
    --input_att_dir  ./dataset/dataset_bu/UCM_captions/ucm_bu_att \
    --input_box_dir  ./dataset/dataset_bu/UCM_captions/ucm_bu_box \
    --cached_tokens ./dataset/dataset_bu/UCM_captions/0.2/ucm-train-idxs.p \
    --clip_feat_path ./dataset/clip_feature/ucm_224 \
    --seq_per_img 5 \
    --batch_size 10 \
    --beam_size 1 \
    --num_layers 2 \
    --input_encoding_size 1024 \
    --rnn_size 1024 \
    --language_eval 1 \
    --val_images_use -1 \
    --save_checkpoint_every 50 \
    --start_from log/log_$id \
    --checkpoint_path log/log_$id"_rl" \
    --learning_rate 2e-5 \
    --max_epochs 70 \
    --self_critical_after 0 \
    --learning_rate_decay_start -1 \
    --scheduled_sampling_start -1 \
    --reduce_on_plateau 