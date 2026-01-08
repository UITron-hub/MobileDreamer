#!/bin/bash
# 使用 vLLM 服务进行世界模型推理


export CLAUDE_BASE_URL=""
export OPENAI_BASE_URL=""
export OPENAI_API_KEY=""
export SANDBOX_CLUSTER_ENDPOINT=""
export SANDBOX_APPLICATION_TOKEN=""

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "${SCRIPT_DIR}"

SAVE_DIR="${SCRIPT_DIR}/results"

seed=42

# 实验名称
exp_name="textual-sketch-world-model"

REASONER_MODEL=""            # 规划者模型 
ACTOR_MODEL=""               # 执行者模型
NUM_THREADS=8                # 并行线程数
TASK_FAMILY="android_world"  # 任务集


# vLLM 服务地址
WORLD_MODEL_URL="http://"
WORLD_MODEL_NAME="textual-sketch-world-model"

# 候选动作数量
NUM_CANDIDATE_ACTIONS=3


echo "检查 vLLM 世界模型服务"

VLLM_CHECK=$(python -c "
from openai import OpenAI
try:
    client = OpenAI(base_url='${WORLD_MODEL_URL}', api_key='EMPTY', timeout=30)
    models = client.models.list()
    print('OK')
except Exception as e:
    print('FAIL')
" 2>/dev/null)

if [ "$VLLM_CHECK" = "OK" ]; then
    echo "✓ vLLM 服务可用: ${WORLD_MODEL_URL}"
else
    echo "⚠ 警告: vLLM 服务可能不可用: ${WORLD_MODEL_URL}"
    echo "  请确保已启动 vllm_setup_lora.sh"
    echo ""
    read -p "是否继续执行？(y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

mkdir -p "${SAVE_DIR}/${exp_name}"

python rollout_imagination.py \
    --save_path "${SAVE_DIR}/${exp_name}/trajectory" \
    --task_family "${TASK_FAMILY}" \
    --num_threads ${NUM_THREADS} \
    --reasoner_model_name "${REASONER_MODEL}" \
    --actor_model_name "${ACTOR_MODEL}" \
    --seed ${seed} \
    --n_task_combinations 1 \
    --world_model_url "${WORLD_MODEL_URL}" \
    --world_model_name "${WORLD_MODEL_NAME}" \
    --num_candidate_actions ${NUM_CANDIDATE_ACTIONS} \
    --wait_after_action_seconds 2.0 \
    2>&1 | tee "${SAVE_DIR}/${exp_name}/$(date +%Y%m%d-%H%M%S).log"


echo "运行完成！结果保存在: ${SAVE_DIR}/${exp_name}"
