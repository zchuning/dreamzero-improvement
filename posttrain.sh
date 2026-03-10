export WANDB_API_KEY=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
export DROID_DATA_ROOT=/mnt/aws-lfs-02/shared/chuningz/datasets/polaris_food_bussing_fixres
export WAN_CKPT_DIR=/mnt/aws-lfs-02/shared/chuningz/checkpoints/Wan2.1-I2V-14B-480P
export OUTPUT_DIR=/mnt/aws-lfs-02/shared/chuningz/checkpoints/dreamzero_ckpts/polaris_food_bussing_lora_noactionloss
export NUM_GPUS=4
bash scripts/train/droid_training.sh