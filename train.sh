conda activate sim3dafford
cd ./qwen-vl-finetune

export WANDB_MODE=offline
export WANDB_DIR=./wandb_offline

torchrun --nproc_per_node=8 qwenvl/train/train_qwen.py \
  --model_name_or_path ./model \
  --dataset_use affordance \
  --data_flatten True \
  --tune_mm_vision True \
  --tune_mm_mlp True \
  --tune_mm_llm True \
  --bf16 \
  --fsdp "full_shard auto_wrap" \
  --fsdp_config ./scripts/fsdp_config_qwen2_5_vl.json \
  --gradient_checkpointing True \
  --output_dir ./output_physx_fullft \
  --num_train_epochs 3 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --max_pixels 262144 \
  --min_pixels 65536 \
  --model_max_length 32768 \
  --save_steps 300 \
  --save_total_limit 1 \
  --learning_rate 2e-5 \
  --warmup_ratio 0.03 \
  --report_to wandb \
  --logging_steps 1