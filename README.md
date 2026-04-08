# 3SA Research Scaffold

This repository provides a modular PyTorch project scaffold for the four-stage 3SA experiment plan.

1. Attention overlap and IoU analysis across surrogate paradigms.
2. Core 3SA attack with dynamic masks and PCGrad.
3. Batch result aggregation and Pareto frontier visualization.
4. Defense-pipeline evaluation with spatial and denoising transforms.

## Project layout

```text
src/threesa/
  analysis/        CSV aggregation and plotting
  attacks/         PCGrad and 3SA attack loop
  data/            Image-text dataset loading
  defenses/        Evaluation-time defense transforms
  models/          Surrogate encoder interfaces and mock backends
scripts/
  stage1_iou.py
  stage2_attack.py
  stage3_pareto.py
  stage4_defense_eval.py
```

## Install

```bash
pip install -e .
```

## Usage

```bash
python scripts/stage1_iou.py --data-dir "./data/MM-SafetyBench(imgs)" --use-mock --max-images 10 --prompt "Describe the weapon."
python scripts/stage2_attack.py --data-dir "./data/MM-SafetyBench(imgs)" --output-dir ./outputs/adv
python scripts/stage3_pareto.py --csv ./outputs/results.csv --output ./outputs/pareto.png
python scripts/stage4_defense_eval.py --data-dir ./outputs/adv
```

## Notes

- The default surrogate registry includes a deterministic mock backend so the pipeline can run before real checkpoints are wired in.
- `stage1_iou.py` now scans image folders recursively, so it can run on nested category datasets such as MM-SafetyBench.
- Stage 1 accepts an optional `prompts.csv` with columns `image_id,prompt`, where `image_id` can be either a relative path like `01-Illegal_Activitiy/SD/0` or a simple stem like `0`.
- Replace the `MockVisionLanguageSurrogate` instances in `scripts/` with concrete Hugging Face model wrappers once your compute environment is ready.
- The attack code is intentionally modular: the data path, mask strategy, and gradient aggregation can be swapped independently for ablations.
