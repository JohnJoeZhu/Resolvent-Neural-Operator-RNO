TASK="diffusion"          # 任务名称
LOCAL_RANK=1     # local_rank 参数
CONFIG="config_glno.yaml" # 替换为实际配置文件路径（如 config.yaml）

# 定义 seed 列表（按需修改）
SEEDS=(
    42
    13
    56
    72
    82 
    234
    137
    41
    295
    184
)
    

CHANNELS=(
    16
    32
    64
)

MODES=(
    2
    4
    6
)

SIGMA=(
    2
    4
    6
)

echo "开始训练，任务名称：${TASK}，配置文件：${CONFIG}，local_rank：${LOCAL_RANK}"

# 遍历所有 seed 并执行训练
for SEED in "${SEEDS[@]}"; do
    # for CHANNEL in "${CHANNELS[@]}"; do
    #     for MODE in "${MODES[@]}"; do
    #         for SIGMA in "${SIGMA[@]}"; do

                echo "========================================="
                echo "开始训练，seed = ${SEED}" #, channel = ${CHANNEL}, mode = ${MODE}, sigma = ${SIGMA}
                echo "========================================="

                python train.py \
                    --task="${TASK}" \
                    --config="${CONFIG}" \
                    --local_rank="${LOCAL_RANK}" \
                    --seed="${SEED}" \
                    # --channel="${CHANNEL}" \
                    # --mode="${MODE}" \
                    # --sigma="${SIGMA}" \
                    

                # 检查训练是否成功
                if [ $? -ne 0 ]; then
                    echo "错误：seed = ${SEED} channel = ${CHANNEL}, mode = ${MODE}, sigma = ${SIGMA} 的训练失败，退出脚本。"
                    exit 1
                fi

                echo "seed = ${SEED} 的训练完成。" #chabnnel = ${CHANNEL}, mode = ${MODE}, sigma = ${SIGMA} 
                echo
    #         done
    #     done
    # done
done

echo "所有 seed 训练完毕。"