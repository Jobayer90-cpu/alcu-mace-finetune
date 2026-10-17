#!/bin/bash
cd ~/alcu
E0=$(cat E0.txt)
mace_run_train --name="AlCu_ft3" --foundation_model="medium" \
  --multiheads_finetuning=False --train_file="data_train.xyz" \
  --valid_fraction=0.15 --energy_key="energy" --forces_key="forces" \
  --E0s="{13:${E0}, 29:${E0}}" --lr=1e-4 \
  --energy_weight=10.0 --forces_weight=10.0 \
  --max_num_epochs=120 --batch_size=2 --patience=30 \
  --default_dtype="float64" --device=cpu \
  --model_dir="." --checkpoints_dir="./ckpt4" --results_dir="./results" --log_dir="./logs"
