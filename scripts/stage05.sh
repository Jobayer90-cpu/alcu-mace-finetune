#!/bin/bash
cd ~/alcu
mace_run_train \
  --name="AlCu_ft2" \
  --foundation_model="medium" \
  --multiheads_finetuning=False \
  --train_file="data_train.xyz" \
  --valid_fraction=0.15 \
  --energy_key="energy" --forces_key="forces" \
  --E0s="{13:-1127.907024, 29:-1127.907024}" \
  --lr=1e-4 \
  --energy_weight=1.0 --forces_weight=100.0 \
  --max_num_epochs=120 \
  --batch_size=2 \
  --patience=30 \
  --default_dtype="float64" \
  --device=cpu \
  --model_dir="." --checkpoints_dir="./ckpt3" \
  --results_dir="./results" --log_dir="./logs"
