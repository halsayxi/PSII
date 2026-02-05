#!/usr/bin/env bash
set -e

#######################################
# Global config
#######################################
GPU=6
OUTPUT_BASE="outputs"

MODELS=(
    qwen2.5-7b
    qwen2.5-14b
    mistral-24b
    llama3.1-8b
)

#######################################
# Runner
#######################################
run () {
    local ARGS="$*"

    if [[ "$ARGS" == *"--method"* && "$ARGS" == *"value_vector"* ]]; then
        ARGS="$ARGS --value_vectors_dir value_vectors/vectors_${MODEL}"
    fi

    if [[ "$ARGS" == *"--method"* && "$ARGS" == *"demographic_vectors"* ]]; then
        ARGS="$ARGS --demographic_vectors_dir demographic_vectors/vectors_${MODEL}"
    fi

    echo "▶ [$MODEL] $ARGS"
    eval "$CMD_PREFIX $ARGS"
    echo
}

#######################################
# Main loop
#######################################
for MODEL in "${MODELS[@]}"; do

    OUTPUT_DIR="${OUTPUT_BASE}"
    mkdir -p "${OUTPUT_DIR}"

    NOISE_ARGS=""
    VECTOR_Q_CODES=""
    VECTOR_Q_CODES_70=""
    case "$MODEL" in
        qwen2.5-7b)
            NOISE_ARGS="--noise_std 0.3"
            VECTOR_Q_CODES="271:14,285:14,263:18,289:18,260:22,273:22,275:22,279:22,281:22,284:22,286:22,287:22,288:22"
            VECTOR_Q_CODES_70="9999:20"
            ;;
        qwen2.5-14b)
            NOISE_ARGS="--noise_std 0.35"
            VECTOR_Q_CODES="271:24,285:24,263:32,289:32,260:38,273:38,275:38,279:38,281:38,284:38,286:38,287:38,288:38"
            VECTOR_Q_CODES_70="9999:34"
            ;;
        mistral-24b)
            NOISE_ARGS="--noise_std 0.09"
            VECTOR_Q_CODES="271:20,285:20,263:26,289:26,260:32,273:32,275:32,279:32,281:32,284:32,286:32,287:32,288:32"
            VECTOR_Q_CODES_70="9999:28"
            ;;
        llama3.1-8b)
            NOISE_ARGS="--noise_std 0.07"
            VECTOR_Q_CODES="271:16,285:16,263:21,289:21,260:26,273:26,275:26,279:26,281:26,284:26,286:26,287:26,288:26"
            VECTOR_Q_CODES_70="9999:22"
            ;;
        *)
            echo "❌ Unknown model: $MODEL"
            exit 1
            ;;
    esac
    DEMOGRAPHIC_ARGS="--coef 2.0 --steering_type all \
        --vector_q_codes ${VECTOR_Q_CODES} \
        ${NOISE_ARGS}"
    DEMOGRAPHIC_NO_NOISE_ARGS="--coef 2.0 --steering_type all \
        --vector_q_codes ${VECTOR_Q_CODES}"
    DEMOGRAPHIC_NO_COEF_ARGS="--steering_type all \
        --vector_q_codes ${VECTOR_Q_CODES} \
        ${NOISE_ARGS}"
    DEMOGRAPHIC_70_ARGS="--coef 2.0 --steering_type all \
        --vector_q_codes ${VECTOR_Q_CODES_70} \
        ${NOISE_ARGS}"
    
    CMD_PREFIX="CUDA_VISIBLE_DEVICES=${GPU} python run.py \
        --use_local_model \
        --model_name ${MODEL} \
        --output_dir ${OUTPUT_DIR}"

    echo "======================================"
    echo "🚀 Running experiments for model: ${MODEL}"
    echo "======================================"

    #######################################
    # 1. Baselines
    #######################################
    run "--method direct --temperature 0.7"
    run "--method direct --temperature 2"
    run "--method prompt_engineering"
    run "--method multilingual"
    run "--method requesting_diversity"
    

    #######################################
    # 2. Our methods
    # #######################################
    run "--method value_vector,demographic_vectors,prompt_engineering ${DEMOGRAPHIC_ARGS}"

    #######################################
    # 3. Ablation Studys
    #######################################
    run "--method demographic_vectors,prompt_engineering ${DEMOGRAPHIC_ARGS}"
    run "--method value_vector,demographic_vectors ${DEMOGRAPHIC_ARGS}"
    run "--method value_vector,prompt_engineering"
    run "--method value_vector,demographic_vectors,prompt_engineering ${DEMOGRAPHIC_NO_NOISE_ARGS}"
    run "--method value_vector,demographic_vectors,prompt_engineering ${DEMOGRAPHIC_70_ARGS}"

done

echo "✅ All models & experiments finished."
