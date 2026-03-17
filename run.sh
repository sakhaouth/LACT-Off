llm_model_name=TimeLLM
train_epochs=20
llm_learning_rate=0.01
lstm_learning_rate=0.01
# llama_layers=32
llama_layers=32

master_port=2025
num_process=4 #original 8
# batch_size=24
batch_size=1
# d_model=32
d_model=16
# d_ff=128
d_ff=128

comment='time-series-predictor'

# accelerate launch --multi_gpu --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port run_main.py \
# accelerate launch --cpu --num_processes $num_process --main_process_port $master_port --num_cpu_threads_per_process 8 run_main.py \
python predictor_training.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small \
  --data_path ETTh1.csv \
  --model_id ETTh1_512_96 \
  --llm_model $llm_model_name \
  --data ETTh1 \
  --features M \
  --seq_len 16 \
  --label_len 8 \
  --pred_len 2 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --itr 1 \
  --d_model $d_model \
  --d_ff $d_ff \
  --freq  mi\
  --seasonal_patterns every 15 minutes \
  --batch_size $batch_size \
  --llm_learning_rate $llm_learning_rate \
  --lstm_learning_rate $lstm_learning_rate \
  --llm_layers $llama_layers \
  --train_epochs $train_epochs \
  --model_comment $comment