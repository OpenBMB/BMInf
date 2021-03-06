MODEL_TYPE="blocklm-130B"
MODEL_ARGS="--num-layers 70 \
            --hidden-size 12288 \
            --inner-hidden-size 32768 \
            --vocab-size 150528 \
            --num-attention-heads 96 \
            --max-sequence-length 1025 \
            --tokenizer-type icetk-glm-130B \
            --layernorm-order post \
            --skip-init \
            --task-mask \
            --load ${CHECKPOINT_PATH}/iter_0020000"
            # 
