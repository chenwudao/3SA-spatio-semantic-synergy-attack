#!/bin/bash
# 阶段1实验快速启动脚本
# 用法: bash scripts/run_stage1.sh [mock|real] [sample_size]

set -e

MODE=${1:-mock}        # mock 或 real
SAMPLE_SIZE=${2:-50}   # 每类采样数量

echo "============================================"
echo "阶段1实验启动"
echo "============================================"
echo "模式: $MODE"
echo "每类采样: $SAMPLE_SIZE 张"
echo "============================================"

# 1. 采样数据
SAMPLE_DIR="data/sample_${SAMPLE_SIZE}"
if [ ! -d "$SAMPLE_DIR" ]; then
    echo ""
    echo "[1/3] 采样数据..."
    python scripts/sample_dataset.py \
        --data-dir "./data/MM-SafetyBench(imgs)" \
        --per-class "$SAMPLE_SIZE" \
        --output "$SAMPLE_DIR" \
        --gen-prompts \
        --symlinks
else
    echo ""
    echo "[1/3] 样本目录已存在: $SAMPLE_DIR (跳过采样)"
fi

# 2. 运行实验
OUTPUT_DIR="results/stage1_${MODE}_sample${SAMPLE_SIZE}"
echo ""
echo "[2/3] 运行阶段1实验..."
echo "输出目录: $OUTPUT_DIR"

if [ "$MODE" = "mock" ]; then
    python scripts/stage1_calibration.py \
        --data-dir "$SAMPLE_DIR" \
        --save-dir "$OUTPUT_DIR" \
        --use-mock
else
    python scripts/stage1_calibration.py \
        --data-dir "$SAMPLE_DIR" \
        --save-dir "$OUTPUT_DIR"
fi

# 3. 显示结果
echo ""
echo "[3/3] 实验完成！"
echo "============================================"
echo "结果位置: $OUTPUT_DIR"
echo "============================================"
ls -la "$OUTPUT_DIR"/*.png "$OUTPUT_DIR"/*.csv "$OUTPUT_DIR"/*.json 2>/dev/null || true
echo ""
echo "查看结果:"
echo "  - 阈值扫描曲线: $OUTPUT_DIR/threshold_sweep_plot.png"
echo "  - 相关性矩阵:   $OUTPUT_DIR/soft_correlation_matrix.png"
echo "  - 详细数据:     $OUTPUT_DIR/threshold_sweep.csv"
echo "  - 摘要:         $OUTPUT_DIR/calibration_summary.json"
