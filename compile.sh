#!/bin/bash

# Compile Qwen-Image-Edit-2511 for Neuron (trn2.48xlarge)
#
# Configuration (hardcoded):
#   - Output size:          1024x512
#   - VAE tile size:        512x512 (fixed, tiled processing)
#   - Vision encoder size:  448x448
#   - TP degree:            16 (transformer), 4 (language model + vision encoder)
#   - World size:           32 (TP=16, DP=2 for CFG parallel)
#   - Max sequence length:  1024
#   - Patch multiplier:     3 (2-image merging for virtual try-on)
#   - Batch size:           1
#
# Usage:
#   ./compile.sh 2>&1 | tee compile_$(date +%Y%m%d_%H%M%S).log
#
# Run inference after compilation:
#   NEURON_RT_NUM_CORES=32 python run_qwen_image_edit.py \
#       --images blue-shirt.png model.png \
#       --prompt "your edit instruction" \
#       --height 1024 --width 512 \
#       --patch_multiplier 3 \
#       --compiled_models_dir /opt/dlami/nvme/compiled_models

set -e

export PYTHONPATH=`pwd`:$PYTHONPATH

# Hardcoded settings for v3_tp16
HEIGHT=1024
WIDTH=512
IMAGE_SIZE=448
TP_DEGREE=16
MAX_SEQ_LEN=1024
PATCH_MULTIPLIER=3
BATCH_SIZE=1
VAE_TILE_SIZE=512

COMPILED_MODELS_DIR="/opt/dlami/nvme/compiled_models"
COMPILER_WORKDIR="/opt/dlami/nvme/compiler_workdir"

# Helper: print a timestamped banner
ts() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

SCRIPT_START=$(date +%s)

echo "============================================"
echo "Qwen-Image-Edit-2511 Compilation for Neuron"
echo "V3 TP16 (TP=16, DP=2, world_size=32)"
echo "============================================"
echo "Output Size:              ${HEIGHT}x${WIDTH}"
echo "VAE Tile Size:            ${VAE_TILE_SIZE}x${VAE_TILE_SIZE} (fixed)"
echo "Vision Encoder Size:      ${IMAGE_SIZE}x${IMAGE_SIZE}"
echo "Transformer TP Degree:    ${TP_DEGREE}"
echo "World Size:               32 (TP=16 x DP=2)"
echo "Max Sequence Length:      ${MAX_SEQ_LEN}"
echo "Patch Multiplier:         ${PATCH_MULTIPLIER} (2-image merging)"
echo "Batch Size:               ${BATCH_SIZE}"
echo "Compiled Models Dir:      ${COMPILED_MODELS_DIR}"
ts "Compilation started"
echo "============================================"
echo ""

# ── Step 1: Compile VAE ───────────────────────────────────────────────────────
STEP1_START=$(date +%s)
ts "[Step 1/3] Compiling VAE (encoder + decoder, tile: ${VAE_TILE_SIZE}x${VAE_TILE_SIZE}, bfloat16)..."
echo "  Using modified VAE with 'nearest' interpolation (Neuron doesn't support 'nearest-exact')"
python neuron_qwen_image_edit/compile_vae.py \
    --height ${VAE_TILE_SIZE} \
    --width ${VAE_TILE_SIZE} \
    --temporal_frames 1 \
    --batch_size ${BATCH_SIZE} \
    --compiled_models_dir ${COMPILED_MODELS_DIR} \
    --compiler_workdir ${COMPILER_WORKDIR}
STEP1_END=$(date +%s)
ts "[Step 1/3] VAE compiled successfully! ($(( STEP1_END - STEP1_START ))s)"
echo ""

# ── Step 2a: Compile Transformer ──────────────────────────────────────────────
STEP2A_START=$(date +%s)
ts "[Step 2a/3] Compiling Transformer V3 CFG (TP=16, DP=2, world_size=32, bfloat16, NKI Flash Attention)..."
echo "  24 attention heads padded to 32 → 2 heads/rank"
python neuron_qwen_image_edit/compile_transformer_v3_cfg.py \
    --height ${HEIGHT} \
    --width ${WIDTH} \
    --tp_degree ${TP_DEGREE} \
    --world_size 32 \
    --patch_multiplier ${PATCH_MULTIPLIER} \
    --max_sequence_length ${MAX_SEQ_LEN} \
    --compiled_models_dir ${COMPILED_MODELS_DIR} \
    --compiler_workdir ${COMPILER_WORKDIR}
STEP2A_END=$(date +%s)
ts "[Step 2a/3] Transformer compiled successfully! ($(( STEP2A_END - STEP2A_START ))s)"
echo ""

# ── Step 2b: Compile Language Model ───────────────────────────────────────────
STEP2B_START=$(date +%s)
ts "[Step 2b/3] Compiling Language Model (TP=4, world_size=32, bfloat16)..."
echo "  TP=4: perfect GQA fit (28Q/4=7 heads/rank, 4KV/4=1 head/rank)"
python neuron_qwen_image_edit/compile_language_model_v3.py \
    --max_sequence_length ${MAX_SEQ_LEN} \
    --batch_size ${BATCH_SIZE} \
    --world_size 32 \
    --compiled_models_dir ${COMPILED_MODELS_DIR} \
    --compiler_workdir ${COMPILER_WORKDIR}
STEP2B_END=$(date +%s)
ts "[Step 2b/3] Language Model compiled successfully! ($(( STEP2B_END - STEP2B_START ))s)"
echo ""

# ── Step 3: Compile Vision Encoder ────────────────────────────────────────────
STEP3_START=$(date +%s)
ts "[Step 3/3] Compiling Vision Encoder (TP=4, world_size=32, float32)..."
echo "  TP=4 is max: MLP intermediate size (3420) not divisible by 8"
echo "  float32 precision required for accuracy"
python neuron_qwen_image_edit/compile_vision_encoder_v3.py \
    --image_size ${IMAGE_SIZE} \
    --world_size 32 \
    --compiled_models_dir ${COMPILED_MODELS_DIR} \
    --compiler_workdir ${COMPILER_WORKDIR}
STEP3_END=$(date +%s)
ts "[Step 3/3] Vision Encoder compiled successfully! ($(( STEP3_END - STEP3_START ))s)"
echo ""

# ── Summary ───────────────────────────────────────────────────────────────────
SCRIPT_END=$(date +%s)
TOTAL=$(( SCRIPT_END - SCRIPT_START ))

echo "============================================"
echo "Compilation Complete!"
echo "============================================"
echo ""
echo "Compiled models saved to: ${COMPILED_MODELS_DIR}/"
echo "  - vae_encoder/          (tile: ${VAE_TILE_SIZE}x${VAE_TILE_SIZE}, batch: ${BATCH_SIZE})"
echo "  - vae_decoder/          (tile: ${VAE_TILE_SIZE}x${VAE_TILE_SIZE}, batch: ${BATCH_SIZE})"
echo "  - transformer_v3_cfg/   (TP=16, DP=2, world_size=32, output: ${HEIGHT}x${WIDTH})"
echo "  - language_model_v3/    (TP=4, world_size=32)"
echo "  - vision_encoder_v3/    (TP=4, world_size=32, float32)"
echo ""
echo "--------------------------------------------"
echo "Time Summary"
echo "--------------------------------------------"
printf "  Step 1 - VAE:                  %4ds\n" $(( STEP1_END  - STEP1_START  ))
printf "  Step 2a - Transformer:         %4ds\n" $(( STEP2A_END - STEP2A_START ))
printf "  Step 2b - Language Model:      %4ds\n" $(( STEP2B_END - STEP2B_START ))
printf "  Step 3  - Vision Encoder:      %4ds\n" $(( STEP3_END  - STEP3_START  ))
echo "  ------------------------------------------"
printf "  Total:                         %4ds  (~%dm)\n" ${TOTAL} $(( TOTAL / 60 ))
echo "--------------------------------------------"
echo ""
ts "Compilation finished"
echo ""
echo "To run inference:"
echo "  NEURON_RT_NUM_CORES=32 python run_qwen_image_edit.py \\"
echo "      --images blue-shirt.png model.png \\"
echo "      --prompt \"your edit instruction\" \\"
echo "      --height ${HEIGHT} \\"
echo "      --width ${WIDTH} \\"
echo "      --patch_multiplier ${PATCH_MULTIPLIER} \\"
echo "      --compiled_models_dir ${COMPILED_MODELS_DIR}"
echo ""
