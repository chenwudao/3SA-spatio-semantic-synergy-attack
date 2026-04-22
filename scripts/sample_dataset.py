"""
分层采样脚本：从MM-SafetyBench每个类别按比例/固定数量抽取图像。

用法:
    # 每类固定50张
    python scripts/sample_dataset.py --data-dir "./data/MM-SafetyBench(imgs)" --per-class 50 --output data/sample_50

    # 按比例5%抽样
    python scripts/sample_dataset.py --data-dir "./data/MM-SafetyBench(imgs)" --ratio 0.05 --output data/sample_5pct

    # 生成prompts.csv
    python scripts/sample_dataset.py --data-dir "./data/MM-SafetyBench(imgs)" --per-class 50 --output data/sample_50 --gen-prompts
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def discover_images(data_dir: Path) -> dict[str, list[Path]]:
    """按类别发现图像，返回 {category_name: [image_paths]}."""
    categories = {}
    for item in sorted(data_dir.iterdir()):
        if not item.is_dir():
            continue
        # 跳过镜像根目录的子目录（如 MM-SafetyBench(imgs)/MM-SafetyBench(imgs)/）
        if item.name == data_dir.name:
            continue
        images = []
        for img_path in item.rglob("*"):
            if img_path.is_file() and img_path.suffix.lower() in IMAGE_EXTENSIONS:
                images.append(img_path)
        if images:
            categories[item.name] = sorted(images)
    return categories


def sample_images(
    categories: dict[str, list[Path]],
    per_class: int | None = None,
    ratio: float | None = None,
    seed: int = 42,
) -> dict[str, list[Path]]:
    """按固定数量或比例从每个类别抽样."""
    rng = random.Random(seed)
    sampled = {}

    for cat_name, images in categories.items():
        if per_class is not None:
            k = min(per_class, len(images))
        elif ratio is not None:
            k = max(1, int(len(images) * ratio))
        else:
            raise ValueError("必须指定 per_class 或 ratio")

        sampled[cat_name] = rng.sample(images, k)

    return sampled


def copy_sampled_images(
    sampled: dict[str, list[Path]],
    output_dir: Path,
    use_symlinks: bool = True,
) -> None:
    """将抽样图像复制到输出目录，保持类别结构."""
    output_dir.mkdir(parents=True, exist_ok=True)

    for cat_name, images in sampled.items():
        cat_dir = output_dir / cat_name
        cat_dir.mkdir(parents=True, exist_ok=True)
        for img_path in images:
            dest = cat_dir / img_path.name
            if use_symlinks:
                if not dest.exists():
                    dest.symlink_to(img_path.resolve())
            else:
                if not dest.exists():
                    shutil.copy2(img_path, dest)


def generate_prompts_csv(
    sampled: dict[str, list[Path]],
    output_dir: Path,
    prompt_templates: dict[str, list[str]],
    seed: int = 42,
) -> None:
    """为抽样图像生成prompts.csv，随机分配不同层次的prompt."""
    rng = random.Random(seed)
    csv_path = output_dir / "prompts.csv"

    rows = []
    for cat_name, images in sampled.items():
        # 获取该类别的prompt模板
        templates = prompt_templates.get(cat_name, prompt_templates.get("_default", ["Describe this image."]))

        for img_path in images:
            # image_id 使用类别/文件名的格式（因为symlink指向原始路径）
            relative_id = f"{cat_name}/{img_path.stem}"
            # 随机选择一个prompt
            prompt = rng.choice(templates)
            rows.append(f"{relative_id},{prompt}")

    csv_path.write_text("image_id,prompt\n" + "\n".join(rows) + "\n", encoding="utf-8")
    print(f"Generated prompts.csv with {len(rows)} entries at {csv_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="分层采样MM-SafetyBench数据集")
    parser.add_argument("--data-dir", required=True, help="MM-SafetyBench根目录")
    parser.add_argument("--output", default=None, help="抽样结果输出目录（--summary-only时可选）")
    parser.add_argument("--per-class", type=int, default=None, help="每个类别抽取的固定数量")
    parser.add_argument("--ratio", type=float, default=None, help="每个类别的抽样比例（0-1）")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--symlinks", action="store_true", help="使用符号链接而非复制（节省空间）")
    parser.add_argument("--gen-prompts", action="store_true", help="同时生成prompts.csv")
    parser.add_argument("--summary-only", action="store_true", help="仅打印数据分布摘要，不执行采样")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"数据目录不存在: {data_dir}")

    # 发现图像
    print("正在扫描数据目录...")
    categories = discover_images(data_dir)

    # 打印分布摘要
    total = sum(len(imgs) for imgs in categories.values())
    print(f"\n{'类别':<30} {'数量':>6} {'占比':>6}")
    print("-" * 45)
    for cat_name, images in categories.items():
        pct = len(images) / total * 100
        print(f"{cat_name:<30} {len(images):>6} {pct:>5.1f}%")
    print("-" * 45)
    print(f"{'总计':<30} {total:>6} {'100%':>6}")

    if args.summary_only:
        return

    if args.output is None:
        parser.error("--output is required when not using --summary-only")

    if args.per_class is None and args.ratio is None:
        parser.error("必须指定 --per-class 或 --ratio")

    # 执行采样
    print(f"\n正在采样 (per_class={args.per_class}, ratio={args.ratio}, seed={args.seed})...")
    sampled = sample_images(categories, per_class=args.per_class, ratio=args.ratio, seed=args.seed)

    sampled_total = sum(len(imgs) for imgs in sampled.values())
    print(f"\n抽样结果:")
    for cat_name, images in sampled.items():
        print(f"  {cat_name}: {len(images)} 张")
    print(f"  总计: {sampled_total} 张")

    # 复制/链接图像
    output_dir = Path(args.output)
    print(f"\n正在输出到 {output_dir}...")
    copy_sampled_images(sampled, output_dir, use_symlinks=args.symlinks)

    # 生成prompts.csv
    if args.gen_prompts:
        import sys
        sys.path.insert(0, str(Path(__file__).parent))
        from prompt_templates import get_prompt_templates
        templates = get_prompt_templates()
        generate_prompts_csv(sampled, output_dir, templates, seed=args.seed)

    # 保存采样元信息
    meta = {
        "per_class": args.per_class,
        "ratio": args.ratio,
        "seed": args.seed,
        "categories": {cat: len(imgs) for cat, imgs in sampled.items()},
        "total": sampled_total,
    }
    meta_path = output_dir / "sample_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n采样元信息已保存到 {meta_path}")
    print("完成!")


if __name__ == "__main__":
    main()
