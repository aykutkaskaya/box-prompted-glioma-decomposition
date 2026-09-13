# Separating detector from segmenter loss in box-prompted glioma segmentation under domain shift

Analysis code and per-patient results for the manuscript. Everything reported in
Methods and Results is derived from the files in `reports/` by the scripts in
`src/backend/scripts/`; nothing was transcribed by hand.

## What is here

```
src/backend/core/           volume loading, normalisation, metrics, segmenter wrappers
src/backend/scripts/        evaluation, analysis, verification and figure scripts
src/segmenters/             the MedSAM3 sidecar the detector-free arm runs through
reports/validation/*.jsonl  per-patient results, one line per case per run
reports/*.json              the compiled summaries the manuscript quotes
data/processed/metadata/    the patient split, its summary, and the exclusion list
experiments*/               per-run summaries and per-patient metrics
notebook/                   the training notebook (Colab) and the seed-replicate note
```

## What is not here, and where to get it

Nothing in this list is needed to reproduce a number in the paper. It is listed
because scripts that train or re-run inference will ask for it.

| what | where |
|---|---|
| BraTS 2020 imaging | the BraTS challenge, after its data-use agreement |
| RHUH-GBM imaging | TCIA, `RHUH-GBM` collection (CC BY 4.0) |
| BraTS-Africa imaging | TCIA, BraTS 2023 release |
| SAM 1 and SAM 2.1 checkpoints | the Segment Anything repositories |
| MedSAM3 base checkpoint and LoRA adapters | the adapter project |
| trained RT-DETR detector weights | not distributed; Appendix A.1 gives the configuration |

Set these when a script needs them:

| variable | meaning |
|---|---|
| `DRIVE_ROOT` | project root; defaults to this checkout |
| `MEDSAM3_DIR` | the adapter project checkout, for the adapter-side split checks |
| `MEDSAM3_PROJECT`, `MEDSAM3_SAM3_SRC`, `MEDSAM3_CHECKPOINT`, `MEDSAM3_LORA_DIR` | sidecar paths |
| `RAW_DIR` | cohort imaging root, when re-running inference |
| `PY`, `MPY` | interpreters for the two virtual environments |
| `RHUH_CLIENT_ID`, `RHUH_CONTEXT` | TCIA public-link credentials, see below |
| `CROSSREF_MAILTO` | your address, for Crossref's polite pool |

No credential of any kind is stored in this repository. `scripts/fetch_rhuh.sh`
downloads the RHUH-GBM package over Aspera and needs the client id and context
blob that TCIA's own public download link carries: open the RHUH-GBM page on
TCIA, follow its Download link, and read `client_id` and `state` out of the
faspex URL the browser lands on. TCIA rotates these, so a stored copy would go
stale as well as being a credential.

## Reproducing the numbers

No imaging and no GPU is needed for this part. It reads the stored per-patient
results.

```bash
pip install -r src/backend/requirements.txt
cd src/backend
python scripts/audit_numbers.py          # recompute every value the paper states
```

The manuscript itself is not redistributed here. `audit_numbers.py` therefore
recomputes each value from the stored results and prints it, labelled, so it can
be compared against the published article.

Save the article's text as `PAPER_DRAFT.md` at the repository root and the same
script checks it automatically in both directions, exiting non-zero if either
fails: that every recomputed value appears in the text, and that every decimal
quoted in the abstract, introduction, discussion or limitations matches
something computed in Methods or Results. `scripts/check_claims.py` then adds
the claim-level and cross-reference checks.

The adapter-side split checks need `MEDSAM3_DIR` and are skipped without it.

```bash
python scripts/verify_split_hashes.py    # the two split manifests
python scripts/verify_rhuh_md5.py        # the 720-file RHUH-GBM manifest
python scripts/oracle_components.py      # the per-component oracle comparison
python scripts/rhuh_bgfix_compare.py     # the RHUH-GBM normalisation sensitivity
```

## Re-running inference

Needs the imaging, the checkpoints and a GPU. `scripts/full_validation.py` is
the entry point; the `*.sh` files beside it are the queues that produced each
cohort's results. They resolve the project root from their own location, so a
clone runs without editing paths.

## Notebooks

`notebook/` holds the Colab notebook for the detector ablation, and
`SEED_REPLICATES.md` records the four changes made to it to produce the seed
replicates without overwriting the original runs. The results in the paper are
reproducible from the released per-patient files without retraining, and the
detector configuration is given in Appendix A.1 of the article.

## Related repository

The project's browser interface for inspecting studies is archived separately at
`box-prompted-glioma-viewer`. It plays no part in reproducing anything here.

## Licence

Code is MIT (see `LICENSE`). The per-patient result files are derived
measurements and are released under CC BY 4.0. The imaging is not redistributed
here and remains under the terms of the collections listed above.
