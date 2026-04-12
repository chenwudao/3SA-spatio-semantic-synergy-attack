"""
批量运行阶段1实验脚本。

支持按类别、按prompt层次分组运行，自动聚合结果。

用法:
    # 运行所有类别，使用所有prompt
    python scripts/batch_stage1.py --sample-dir data/sample_50 --output-dir results/stage1_batch

    # 仅运行特定类别
    python scripts/batch_stage1.py --sample-dir data/sample_50 --categories 01-Illegal_Activitiy 02-HateSpeech

    # 仅运行特定prompt层次
    python scripts/batch_stage1.py --sample-dir data/sample_50 --prompt-level L1 L2

    # 使用mock模型快速验证
    python scripts/batch_stage1.py --sample-dir data/sample_50 --use-mock --max-images-per-cat 5
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from datetime import datetime

# 添加scripts目录到路径
sys.path.insert(0, str(Path(__file__).parent))
from prompt_templates import get_prompts_for_category, PROMPT_TEMPLATES


def run_stage1_iou(
    data_dir: Path,
    save_dir: Path,
    prompt: str = "",
    use_mock: bool = False,
    max_images: int = 0,
    topk_ratio: float = 0.10,
) -> dict:
    """运行stage1_iou.py并返回结果摘要."""
    cmd = [
        sys.executable,
        str(Path(__file__).parent / "stage1_iou.py"),
        "--data-dir", str(data_dir),
        "--save-dir", str(save_dir),
        "--topk-ratio", str(topk_ratio),
    ]

    if prompt:
        cmd.extend(["--prompt", prompt])
    if use_mock:
        cmd.append("--use-mock")
    if max_images > 0:
        cmd.extend(["--max-images", str(max_images)])

    print(f"  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"  ERROR: {result.stderr}")
        return {"status": "error", "stderr": result.stderr}

    # 解析输出获取IoU
    iou_value = None
    for line in result.stdout.split("\n"):
        if "iou=" in line:
            try:
                iou_value = float(line.split("iou=")[1].strip().split()[0])
            except (ValueError, IndexError):
                pass

    return {
        "status": "success",
        "iou": iou_value,
        "stdout": result.stdout[-500:] if result.stdout else "",  # 最后500字符
    }


def run_stage1_calibration(
    data_dir: Path,
    save_dir: Path,
    prompt: str = "",
    max_images: int = 0,
    use_mock: bool = False,
    topk_ratios: list[float] | None = None,
) -> dict:
    """运行stage1_calibration.py（支持mock和真实模型）."""
    cmd = [
        sys.executable,
        str(Path(__file__).parent / "stage1_calibration.py"),
        "--data-dir", str(data_dir),
        "--save-dir", str(save_dir),
    ]

    if prompt:
        cmd.extend(["--prompt", prompt])
    if max_images > 0:
        cmd.extend(["--max-images", str(max_images)])
    if use_mock:
        cmd.append("--use-mock")
    if topk_ratios:
        cmd.extend(["--topk-ratios"] + [str(r) for r in topk_ratios])

    print(f"  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"  ERROR: {result.stderr}")
        return {"status": "error", "stderr": result.stderr}

    return {
        "status": "success",
        "stdout": result.stdout[-500:] if result.stdout else "",
    }


def discover_categories(sample_dir: Path) -> list[str]:
    """发现样本目录中的类别."""
    categories = []
    for item in sorted(sample_dir.iterdir()):
        if item.is_dir() and item.name.startswith(tuple(f"{i:02d}" for i in range(1, 20))):
            categories.append(item.name)
    return categories


def main() -> None:
    parser = argparse.ArgumentParser(description="批量运行阶段1实验")
    parser.add_argument("--sample-dir", required=True, help="抽样后的数据目录")
    parser.add_argument("--output-dir", required=True, help="结果输出目录")
    parser.add_argument("--categories", nargs="*", default=None, help="指定运行的类别（默认全部）")
    parser.add_argument("--prompt-level", nargs="*", default=["all"], choices=["L1", "L2", "L3", "all"],
                        help="指定prompt层次（默认全部）")
    parser.add_argument("--use-mock", action="store_true", help="使用mock模型")
    parser.add_argument("--max-images-per-cat", type=int, default=0, help="每个类别最大图像数（0=全部）")
    parser.add_argument("--topk-ratio", type=float, default=0.10, help="top-k阈值")
    parser.add_argument("--skip-existing", action="store_true", help="跳过已完成的实验")
    args = parser.parse_args()

    sample_dir = Path(args.sample_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 发现类别
    categories = args.categories or discover_categories(sample_dir)
    if not categories:
        print(f"错误: 在 {sample_dir} 中未找到类别目录")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"批量阶段1实验配置")
    print(f"{'='*60}")
    print(f"样本目录: {sample_dir}")
    print(f"输出目录: {output_dir}")
    print(f"类别: {categories}")
    print(f"Prompt层次: {args.prompt_level}")
    print(f"使用mock: {args.use_mock}")
    print(f"每类最大图像: {args.max_images_per_cat or '全部'}")
    print(f"{'='*60}\n")

    # 记录所有结果
    all_results = {}
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for category in categories:
        cat_dir = sample_dir / category
        if not cat_dir.exists():
            print(f"跳过不存在的类别: {category}")
            continue

        # 获取该类别的prompts
        prompts_to_test = []
        for level in args.prompt_level:
            if level == "all":
                prompts_to_test.extend(get_prompts_for_category(category, level="all"))
            else:
                prompts_to_test.extend(get_prompts_for_category(category, level=level))

        # 去重
        prompts_to_test = list(dict.fromkeys(prompts_to_test))

        print(f"\n【{category}】({len(prompts_to_test)} 个prompts)")

        cat_results = {}
        for i, prompt in enumerate(prompts_to_test):
            prompt_key = prompt[:50] + "..." if len(prompt) > 50 else prompt
            print(f"  [{i+1}/{len(prompts_to_test)}] Prompt: {prompt_key}")

            # 为每个prompt创建独立的结果目录
            safe_prompt = prompt.replace(" ", "_").replace("/", "_")[:30]
            save_dir = output_dir / category / safe_prompt
            save_dir.mkdir(parents=True, exist_ok=True)

            # 检查是否已存在结果
            summary_file = save_dir / "stage1_summary.json"
            if args.skip_existing and summary_file.exists():
                print(f"    跳过（已存在结果）")
                with open(summary_file) as f:
                    cat_results[prompt_key] = json.load(f)
                continue

            # 运行实验
            result = run_stage1_calibration(
                data_dir=cat_dir,
                save_dir=save_dir,
                prompt=prompt,
                max_images=args.max_images_per_cat,
                use_mock=args.use_mock,
                topk_ratios=[args.topk_ratio] if args.topk_ratio else None,
            )

            cat_results[prompt_key] = result

            if result["status"] == "success":
                iou_str = f"IoU={result.get('iou', 'N/A')}" if "iou" in result else "校准完成"
                print(f"    ✓ {iou_str}")
            else:
                print(f"    ✗ 失败")

        all_results[category] = cat_results

    # 保存汇总结果
    summary_path = output_dir / f"batch_summary_{timestamp}.json"
    summary_data = {
        "timestamp": timestamp,
        "config": {
            "sample_dir": str(sample_dir),
            "categories": categories,
            "prompt_levels": args.prompt_level,
            "use_mock": args.use_mock,
            "max_images_per_cat": args.max_images_per_cat,
            "topk_ratio": args.topk_ratio,
        },
        "results": all_results,
    }
    summary_path.write_text(json.dumps(summary_data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n{'='*60}")
    print(f"批量实验完成！汇总结果: {summary_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
