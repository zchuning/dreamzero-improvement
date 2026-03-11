#!/bin/bash
# DreamZero Full Fine-Tuning with Loss-Weighted Behavior Cloning
#
# Weights the per-sample diffusion training loss by softmax(reward) across the batch.
# Higher-reward frames contribute more to the gradient.
#
# Prerequisites:
#   - Reward-labeled dataset (with next.robometer_progress column)
#   - Pretrained DreamZero checkpoint

export HYDRA_FULL_ERROR=1
export WANDB_API_KEY=${WANDB_API_KEY:-"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"}

# ============ USER CONFIGURATION ============
DROID_DATA_ROOT=${DROID_DATA_ROOT:-"/mnt/aws-lfs-02/shared/chuningz/private_robometer/polaris_labeled"}
WAN_CKPT_DIR=${WAN_CKPT_DIR:-"/mnt/aws-lfs-02/shared/chuningz/checkpoints/Wan2.1-I2V-14B-480P"}
TOKENIZER_DIR=${TOKENIZER_DIR:-"./checkpoints/umt5-xxl"}
OUTPUT_DIR=${OUTPUT_DIR:-"/mnt/aws-lfs-02/shared/chuningz/checkpoints/dreamzero_ckpts/polaris_loss_weighted"}
PRETRAINED_MODEL=${PRETRAINED_MODEL:-"/mnt/aws-lfs-02/shared/chuningz/checkpoints/dreamzero_ckpts/droid_release"}
NUM_GPUS=${NUM_GPUS:-4}

REWARD_COLUMN=${REWARD_COLUMN:-"robometer_progress"}
REWARD_TEMPERATURE=${REWARD_TEMPERATURE:-1.0}
# =============================================

# Validate dataset exists
if [ ! -d "$DROID_DATA_ROOT" ]; then
    echo "ERROR: Dataset not found at $DROID_DATA_ROOT"
    exit 1
fi

torchrun --nproc_per_node $NUM_GPUS --standalone groot/vla/experiment/experiment.py \
    report_to=wandb \
    data=dreamzero/droid_relative \
    wandb_project=dreamzero \
    +wandb_run_name=polaris_loss_weighted \
    train_architecture=full \
    num_frames=33 \
    action_horizon=24 \
    num_views=3 \
    model=dreamzero/vla \
    model/dreamzero/action_head=wan_flow_matching_action_tf \
    model/dreamzero/transform=dreamzero_cotrain \
    num_frame_per_block=2 \
    num_action_per_block=24 \
    num_state_per_block=1 \
    seed=42 \
    training_args.learning_rate=1e-5 \
    training_args.deepspeed="groot/vla/configs/deepspeed/zero2_offload.json" \
    save_steps=500 \
    training_args.warmup_ratio=0.05 \
    output_dir=$OUTPUT_DIR \
    per_device_train_batch_size=1 \
    max_steps=2000 \
    weight_decay=1e-5 \
    save_total_limit=10 \
    upload_checkpoints=false \
    bf16=true \
    tf32=true \
    eval_bf16=true \
    dataloader_pin_memory=false \
    dataloader_num_workers=1 \
    image_resolution_width=320 \
    image_resolution_height=176 \
    save_lora_only=false \
    max_chunk_size=4 \
    frame_seqlen=880 \
    save_strategy=steps \
    droid_data_root=$DROID_DATA_ROOT \
    dit_version=$WAN_CKPT_DIR \
    text_encoder_pretrained_path=$WAN_CKPT_DIR/models_t5_umt5-xxl-enc-bf16.pth \
    image_encoder_pretrained_path=$WAN_CKPT_DIR/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth \
    vae_pretrained_path=$WAN_CKPT_DIR/Wan2.1_VAE.pth \
    tokenizer_path=$TOKENIZER_DIR \
    use_global_metadata=true \
    disable_action_loss=false \
    pretrained_model_path=$PRETRAINED_MODEL \
    reward_weighting_mode=loss_weighted \
    reward_softmax_temperature=$REWARD_TEMPERATURE \
    "reward_column=$REWARD_COLUMN"
