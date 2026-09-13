# Seed replicates: what was changed in the training notebook

The 37-run ablation was trained at a single seed (1337), which is its weakest
point. Three of the runs were repeated at two further seeds so that the trend in
α carries an error bar. This records the four changes that were made to the
notebook in `notebook/` to produce them.

## The trap this avoids

The run directory is derived from the run slug:

```python
def run_dir(run):
    d = os.path.join(CFG.paths.experiments, run.slug())
```

`run.slug()` is `run_{run_id:03d}_{loss_name}{params}{weight}{cls}` and carries
no seed. Changing the seed and re-running therefore writes to the same directory
and overwrites the original result. `CFG.paths.experiments_repeated` exists and
its directory is created, but nothing uses it; step 2 below is what puts it to
work.

## 1. The training seed

In the `CFG` / `TrainConfig` cell:

```python
seed: int = 1337        # 42 for the first repeat, 62 for the second
```

`split_seed` and `negative_seed` stay at 1337. They define the dataset — the
patient train/validation/test division and the negative-slice selection — so
changing them would change the test set and the runs would no longer be
replicates of the originals but a different experiment.

## 2. A separate output tree

Added after `CFG` is built and before `run_pipeline()` is called:

```python
import os
SEED_TAG = f"seed_{CFG.train.seed}"
_repeat_root = os.path.join(CFG.paths.experiments_repeated, SEED_TAG)

CFG.paths.experiments = _repeat_root
CFG.paths.registry    = os.path.join(_repeat_root, "registry")
CFG.paths.reports     = os.path.join(_repeat_root, "reports")
CFG.paths.figures     = os.path.join(_repeat_root, "figures")

for _d in (CFG.paths.experiments, CFG.paths.registry,
           CFG.paths.reports, CFG.paths.figures):
    os.makedirs(_d, exist_ok=True)

assert "experiments_repeated" in CFG.paths.experiments
```

The data paths (`data_dir`, `processed_dir`, `local_media`, `media_archive`,
`dataset_yaml`) are left alone so the slices are bit-identical to the originals
and the 3.7 GB of data is not re-downloaded. `sam_shared` is also unchanged: the
ground-truth-box SAM cache does not depend on the seed.

## 3. Which runs

```python
RUN_IDS = [9, 20, 1]
for _rid in RUN_IDS:
    RUN_MODE, SELECTED_RUN_ID, RUN_RANGE = "single", _rid, None
    run_pipeline()
```

The claim these support is that α controls box coverage (α → C_p, r = −0.94;
α → C_t, r = +0.88). The ends of the trend carry it, and the intermediate values
inherit their confidence from the ends.

| run | configuration | why |
|---|---|---|
| 9 | IC-Arb α = 0.00 | the coverage end |
| 20 | IC-Arb α = 1.00 | the IoU end, plain IoU |
| 1 | GIoU | baseline anchor |

Seeds 42 and 62 match the LoRA adapter seeds, so the paper can refer to
{1337, 42, 62} throughout.

## 4. SAM evaluation

```python
SAM_EVALUATION_POLICY = "deferred_all_runs"
```

The replicates need the detection and geometry metrics — precision, recall, F1,
mAP, box IoU, and `pred_coverage` / `target_coverage` — all of which come from
the detection evaluation step. Skipping SAM saves substantial time per run and
costs nothing for the α-to-coverage claim; whether a different segmenter raises
the ceiling is measured separately, in the oracle-box arm.

## What each replicate produces

Two files per run are enough for the analysis:

```
experiments_repeated/seed_42/run_009_.../summary/run_summary.json
experiments_repeated/seed_42/run_009_.../config/requested_config.json
```

The checkpoints (66 MB each) are not needed and are not kept.

## Checklist

- [ ] `seed` set to 42 or 62; `split_seed` and `negative_seed` still 1337
- [ ] output redirected under `experiments_repeated/seed_XX/`, assertion passes
- [ ] data paths unchanged
- [ ] `RUN_IDS = [9, 20, 1]`
- [ ] `SAM_EVALUATION_POLICY` set if SAM is to be skipped
- [ ] after the first run, the original 37 runs' `summary/status.json`
      timestamps are unchanged
