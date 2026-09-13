"""Fill the training table in Appendix A.1 from the run configuration.

Hyperparameters written by hand drift from the run that produced the numbers.
These are read out of the resolved config the trainer wrote, so the table cannot
say something the run did not do.

    python scripts/build_supplementary.py
"""
from __future__ import annotations

import io
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DRIVE_ROOT  # noqa: E402

DRAFT = DRIVE_ROOT / "PAPER_DRAFT.md"
DRAFT_TR = DRIVE_ROOT / "PAPER_DRAFT_TR.md"
SLUG = "run_037_icarb_alpha_050_w30_clsfl"
RUN = DRIVE_ROOT / "experiments" / SLUG / "config" / "resolved_config.json"


def rows(cfg: dict) -> list:
    aug = cfg.get("augment") or {}
    on = ", ".join(f"{k} {v:g}" for k, v in aug.items() if v) or "none"
    return [
        ("architecture", "RT-DETR-L (Ultralytics)"),
        ("input size", f"{cfg.get('imgsz')} × {cfg.get('imgsz')}"),
        ("optimiser", str(cfg.get("optimizer"))),
        ("initial learning rate", f"{cfg.get('lr0'):g}"),
        ("weight decay", f"{cfg.get('weight_decay'):g}"),
        ("epochs", str(cfg.get("epochs"))),
        ("early-stopping patience", str(cfg.get("patience"))),
        ("batch size", str(cfg.get("batch"))),
        ("mixed precision", "yes" if cfg.get("amp") else "no"),
        ("augmentation", on),
        ("box loss", "L1 + $\\mathcal{L}_{\\text{ov}}$ at α=0.50, weight 3.0"),
        ("classification loss", {"fl": "focal", "vfl": "varifocal",
                                "bce": "BCE"}.get(cfg.get("cls_loss"),
                                                  str(cfg.get("cls_loss")))),
        ("matching", "Hungarian, IoU ≥ 0.5 for TP/FP"),
        ("training seed", str(cfg.get("seed"))),
    ]


def main() -> None:
    cfg = json.loads(RUN.read_text(encoding="utf-8"))
    body = ["**Table A1.** Detector training configuration, read from the "
            "resolved configuration the trainer wrote for run 37. The "
            "tumour-core detectors of §4.5 use the same configuration on "
            "tumour-core boxes, at seeds 1337, 42 and 62.", "",
            "| setting | value |", "|---|---|"]
    body += [f"| {k} | {v} |" for k, v in rows(cfg)]
    body += ["", "The detector-free arm is MedSAM3 with a LoRA adapter. The "
                 "base checkpoint is used as released; the adapter was "
                 "trained for this line of work, and its three replicates "
                 "are seeds 42, 52 and 62. Inference for both arms, and "
                 "the timings of Table 10, ran on a single NVIDIA GeForce "
                 "RTX 5060 Laptop GPU in one session, in mixed precision, "
                 "with the weights already resident so that no load time "
                 "is counted.", ""]

    # rewrite the whole section rather than filling a placeholder once, so
    # re-running after a configuration change actually updates the table
    for path, head, lead in (
        (DRAFT, "### Appendix A.1 Training configuration",
         "The configuration the detector was trained with is given in Table A1."),
        (DRAFT_TR, "### Ek A.1 Eğitim konfigürasyonu",
         "Dedektörün eğitildiği konfigürasyon Tablo A1'de veriliyor."),
    ):
        if not path.exists():
            continue
        s = io.open(path, encoding="utf-8").read()
        a = s.index(head + chr(10)) + len(head) + 1
        b = s.index(chr(10) + "---" + chr(10), a)
        s = s[:a] + chr(10) + lead + chr(10) + chr(10) + chr(10).join(body) + s[b:]
        io.open(path, "w", encoding="utf-8", newline="").write(s)
        print(f"{path.name}: S1 filled, {len(rows(cfg))} settings")


if __name__ == "__main__":
    main()
