#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

# 强制让终端先进入项目根目录
cd /data1/zhouhongliang/Time-Series-Library

model_name=ProSTA

# python -u run.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --root_path ./dataset/ETT-small/ \
#   --data_path ETTh1.csv \
#   --model_id ETTh1_168_168 \
#   --model $model_name \
#   --data ETTh1 \
#   --features M \
#   --seq_len 168 \
#   --label_len 48 \
#   --pred_len 168 \
#   --enc_in 7 \
#   --dec_in 7 \
#   --c_out 7 \
#   --des 'Exp' \
#   --d_model 128 \
#   --d_ff 128 \
#   --batch_size 8 \
#   --learning_rate 0.0001 \
#   --itr 1 \
#   --patience 5 \
#   --dropout 0.3 \
#   --st_dim 256 \
#   --local_window 24 \
#   --top_k 5 \
#   --alpha 0.6

# python -u run.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --root_path ./dataset/ETT-small/ \
#   --data_path ETTh2.csv \
#   --model_id ETTh2_168_168 \
#   --model $model_name \
#   --data ETTh2 \
#   --features M \
#   --seq_len 168 \
#   --label_len 48 \
#   --pred_len 168 \
#   --enc_in 7 \
#   --dec_in 7 \
#   --c_out 7 \
#   --des 'Exp' \
#   --d_model 128 \
#   --d_ff 128 \
#   --batch_size 8 \
#   --learning_rate 0.0001 \
#   --itr 1 \
#   --patience 5 \
#   --dropout 0.3 \
#   --st_dim 256 \
#   --local_window 24 \
#   --top_k 5 \
#   --alpha 0.6

# python -u run.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --root_path ./dataset/ETT-small/ \
#   --data_path ETTm1.csv \
#   --model_id ETTm1_672_672 \
#   --model $model_name \
#   --data ETTm1 \
#   --features M \
#   --seq_len 672 \
#   --label_len 48 \
#   --pred_len 672 \
#   --enc_in 7 \
#   --dec_in 7 \
#   --c_out 7 \
#   --des 'Exp' \
#   --d_model 128 \
#   --d_ff 128 \
#   --batch_size 8 \
#   --learning_rate 0.0001 \
#   --itr 1 \
#   --patience 5 \
#   --dropout 0.3 \
#   --st_dim 256 \
#   --local_window 24 \
#   --top_k 5 \
#   --alpha 0.6

# python -u run.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --root_path ./dataset/ETT-small/ \
#   --data_path ETTm2.csv \
#   --model_id ETTm2_672_672 \
#   --model $model_name \
#   --data ETTm2 \
#   --features M \
#   --seq_len 672 \
#   --label_len 48 \
#   --pred_len 672 \
#   --enc_in 7 \
#   --dec_in 7 \
#   --c_out 7 \
#   --des 'Exp' \
#   --d_model 128 \
#   --d_ff 128 \
#   --batch_size 8 \
#   --learning_rate 0.0001 \
#   --itr 1 \
#   --patience 5 \
#   --dropout 0.3 \
#   --st_dim 256 \
#   --local_window 24 \
#   --top_k 5 \
#   --alpha 0.6

# python -u run.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --root_path ./dataset/electricity/ \
#   --data_path electricity.csv \
#   --model_id ECL_168_168 \
#   --model $model_name \
#   --data custom \
#   --features M \
#   --seq_len 168 \
#   --label_len 48 \
#   --pred_len 168 \
#   --scales 168 24 12 6 \
#   --enc_in 321 \
#   --dec_in 321 \
#   --c_out 321 \
#   --des 'Exp' \
#   --d_model 256 \
#   --d_ff 256 \
#   --batch_size 8 \
#   --learning_rate 0.0001 \
#   --itr 1 \
#   --patience 5 \
#   --dropout 0.3 \
#   --st_dim 256 \
#   --local_window 24 \
#   --top_k 5 \
#   --alpha 0.6

python -u run.py \
--task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/traffic/ \
  --data_path traffic.csv \
  --model_id traffic_168_168 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 168 \
  --label_len 48 \
  --pred_len 168 \
  --enc_in 862 \
  --dec_in 862 \
  --c_out 862 \
  --des 'Exp' \
  --d_model 256 \
  --d_ff 256 \
  --batch_size 8 \
  --learning_rate 0.0001 \
  --itr 1 \
  --patience 5 \
  --dropout 0.3 \
  --st_dim 256 \
  --local_window 24 \
  --top_k 5 \
  --alpha 0.6

  # python -u run.py \
  # --task_name long_term_forecast \
  # --is_training 1 \
  # --root_path ./dataset/weather/ \
  # --data_path weather.csv \
  # --model_id weather_1008_1008 \
  # --model $model_name \
  # --data custom \
  # --features M \
  # --seq_len 1008 \
  # --label_len 48 \
  # --pred_len 1008 \
  # --enc_in 21 \
  # --dec_in 21 \
  # --c_out 21 \
  # --des 'Exp' \
  # --d_model 256 \
  # --d_ff 256 \
  # --batch_size 8 \
  # --learning_rate 0.0001 \
  # --itr 1 \
  # --patience 5 \
  # --dropout 0.3 \
  # --st_dim 256 \
  # --local_window 24 \
  # --top_k 5 \
  # --alpha 0.6
