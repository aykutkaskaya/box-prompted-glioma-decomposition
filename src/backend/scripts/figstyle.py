"""Shared style, bilingual captions and data loaders for the paper figures.

The manuscript language is not fixed yet, so every figure is generated twice
from one code path. Nothing is drawn from a hard-coded number: the loaders below
read the same result files the report writers use, so a figure cannot drift away
from the text that cites it.

Two colours carry meaning throughout and nothing else competes with them:
localisation is amber, segmentation is teal. Everything structural is neutral
slate. The reader should be able to tell the two halves of the decomposition
apart from across a room, which is the whole point of the paper.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DRIVE_ROOT  # noqa: E402
from core.rhuh import FIX, corrected  # noqa: E402

V = DRIVE_ROOT / "reports" / "validation"
R = DRIVE_ROOT / "reports"
OUT = DRIVE_ROOT / "reports" / "figures"
SEG = "sam2.1_l"
COHORTS = ["clean", "rhuh", "brats_africa"]

# ----------------------------------------------------------------- palette
LOC = "#B4690E"        # localisation -- amber, warm, the failing half
SEG_C = "#0E6E6E"      # segmentation -- teal, cool, the stable half
INK = "#1A1D21"
MUTED = "#5B6169"
FAINT = "#9AA1A9"
RULE = "#D8DCE0"
PAPER = "#FFFFFF"
WARN = "#9E2B25"
ORACLE = "#3C4650"


def rc() -> dict:
    """Publication defaults: hairlines, real typography, no chartjunk.

    Sizes are set for a figure reduced to one journal column, which is the
    worst case; at full width they are simply comfortable.
    """
    return {
        "figure.facecolor": PAPER, "axes.facecolor": PAPER,
        "savefig.facecolor": PAPER, "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "mathtext.fontset": "dejavusans",
        "font.size": 9.5, "axes.labelsize": 9.5, "axes.titlesize": 10.5,
        "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "legend.fontsize": 8.5,
        "axes.edgecolor": MUTED, "axes.linewidth": 0.7,
        "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "xtick.major.width": 0.7, "ytick.major.width": 0.7,
        "xtick.major.size": 3, "ytick.major.size": 3,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": False, "grid.color": RULE, "grid.linewidth": 0.6,
        "legend.frameon": False, "lines.linewidth": 1.4,
        "pdf.fonttype": 42, "ps.fonttype": 42,   # embed real glyphs, not paths
    }


# ------------------------------------------------------------- vocabulary
# Keys are stable; only the strings differ. Turkish is written for a reader who
# knows the field, not translated word for word from the English.
L = {
    "en": {
        "cohort": "cohort",
        "clean": "BraTS2020\n(shared held-out set)",
        # RT-DETR stage names
        "s_bb": "CNN" + chr(10) + "backbone",
        "s_enc": "hybrid encoder" + chr(10) + "AIFI + CCFF",
        "s_qs": "uncertainty-min." + chr(10) + "query selection",
        "s_dec": "transformer" + chr(10) + "decoder",
        "s_box": "boxes", "s_mask": "SAM 2.1-L" + chr(10) + "mask",
        # short forms, for figures where the full label will not fit
        "clean_s": "BraTS2020", "rhuh_s": "RHUH-GBM",
        "brats_africa_s": "BraTS-Africa",
        "rhuh": "RHUH-GBM",
        "brats_africa": "BraTS-Africa",
        "dice": "volumetric Dice",
        "dice_wt": "whole-tumour Dice",
        "dice_tc": "tumour-core Dice",
        "loc": "detector stage",
        "seg": "residual",
        "loc_term": "detector-stage term\n(oracle-box − pipeline)",
        "seg_term": "residual term\n(detector-free − oracle-box)",
        "pipeline": "pipeline\n(detector + box-prompted segmenter)",
        "oracle": "oracle-box\n(ground-truth box + same segmenter)",
        "direct": "detector-free\n(Medical SAM 3 + LoRA)",
        "pipeline_s": "pipeline", "oracle_s": "oracle-box", "direct_s": "detector-free",
        "f1": "detection F1",
        "gap": "Dice difference",
        "total_gap": "total gap (detector-free − pipeline)",
        "ahead_pipe": "pipeline ahead", "ahead_free": "detector-free ahead",
        "alpha": r"$\alpha$  (0 = pure coverage, 1 = pure IoU)",
        "cp": r"$C_p$ — prediction coverage",
        "ct": r"$C_t$ — target coverage",
        "seedbar": "bars: seed-to-seed spread",
        "patients": "patients", "patient": "patient",
        "seed": "detector seed", "seeds_n": "detector seeds",
        "spread": "spread across seeds",
        "gt": "ground truth", "pred": "prediction",
        "boxes_n": "boxes", "box_1": "box", "nobox": "no box proposed",
        "fig7_note": "one axial slice per patient — same segmenter, same slice, only the box source differs",
        "gtbox": "ground-truth\nbox", "detbox": "RT-DETR\nbox",
        "tight": "tight, inside", "exact": "exact", "inflated": "inflated, contains GT",
        "ring": "enhancing tumour (ET)", "disc": "tumour core (TC)",
        "samebox": "identical bounding box",
        "volerr": "volume error of the oracle mask",
        "wt": "WT", "tc": "TC", "et": "ET",
        "nearzero": "patients scoring < 0.05",
        "median": "median", "mean": "mean",
        "figs": {
            1: "The decomposition. Given the same volume, three arms differ only in where the box comes from; the two error terms sum exactly to the pipeline's deficit against the detector-free model.",
            2: "The two faces of the intersection. IoU and prediction coverage agree when the prediction contains the target and separate when it sits inside it — the regime the blend parameter arbitrates.",
            3: "Only one half degrades. As the cohort moves away from the training distribution the localisation term grows by an order of magnitude while the segmentation term stays small and changes sign.",
            4: "Why a box cannot ask for a rim. The tight rectangle around an annulus is the tight rectangle around the disc it encloses, so no detector — trained or oracular — can supply a prompt that distinguishes them.",
            5: "Instability grows with distance. Detectors that differ by seed alone diverge further out of domain, and the divergence is carried by a few patients that fail to near zero rather than by a uniform shift.",
            6: "Where each system was modified. Grey is used as published; amber marks what this study changed.",
        },
    },
    "tr": {
        "cohort": "kohort",
        "clean": "BraTS2020\n(ortak dışarıda bırakılmış)",
        # RT-DETR asama adlari
        "s_bb": "CNN" + chr(10) + "omurga",
        "s_enc": "hibrit kodlayıcı" + chr(10) + "AIFI + CCFF",
        "s_qs": "belirsizlik-min." + chr(10) + "sorgu seçimi",
        "s_dec": "transformer" + chr(10) + "kod çözücü",
        "s_box": "kutular", "s_mask": "SAM 2.1-L" + chr(10) + "maskesi",
        # short forms, for figures where the full label will not fit
        "clean_s": "BraTS2020", "rhuh_s": "RHUH-GBM",
        "brats_africa_s": "BraTS-Africa",
        "rhuh": "RHUH-GBM",
        "brats_africa": "BraTS-Africa",
        "dice": "hacimsel Dice",
        "dice_wt": "tüm tümör Dice",
        "dice_tc": "tümör çekirdeği Dice",
        "loc": "dedektör aşaması",
        "seg": "artık",
        "loc_term": "dedektör aşaması terimi\n(oracle-kutu − boru hattı)",
        "seg_term": "artık terim\n(dedektörsüz − oracle-kutu)",
        "pipeline": "boru hattı\n(dedektör + kutu-yönlendirmeli segmenter)",
        "oracle": "oracle-kutu\n(yer gerçeği kutusu + aynı segmenter)",
        "direct": "dedektörsüz\n(Medical SAM 3 + LoRA)",
        "pipeline_s": "boru hattı", "oracle_s": "oracle-box", "direct_s": "dedektörsüz",
        "f1": "tespit F1",
        "gap": "Dice farkı",
        "total_gap": "toplam fark (dedektörsüz − boru hattı)",
        "ahead_pipe": "boru hattı önde", "ahead_free": "dedektörsüz önde",
        "alpha": r"$\alpha$  (0 = saf kapsama, 1 = saf IoU)",
        "cp": r"$C_p$ — öngörü kapsaması",
        "ct": r"$C_t$ — hedef kapsaması",
        "seedbar": "çubuklar: tohumlar arası yayılım",
        "patients": "hasta", "patient": "hasta",
        "seed": "dedektör tohumu", "seeds_n": "dedektör tohumu",
        "spread": "tohumlar arası yayılım",
        "gt": "yer gerçeği", "pred": "öngörü",
        "boxes_n": "kutu", "box_1": "kutu", "nobox": "kutu önerilmedi",
        "fig7_note": "hasta başına tek aksiyel dilim — aynı segmenter, aynı dilim, yalnızca kutu kaynağı farklı",
        "gtbox": "yer gerçeği\nkutusu", "detbox": "RT-DETR\nkutusu",
        "tight": "dar, içeride", "exact": "birebir", "inflated": "şişkin, YG'yi kapsıyor",
        "ring": "kontrastlanan tümör (ET)", "disc": "tümör çekirdeği (TC)",
        "samebox": "aynı sınırlayıcı kutu",
        "volerr": "oracle maskesinin hacim hatası",
        "wt": "WT", "tc": "TC", "et": "ET",
        "nearzero": "0.05 altında kalan hasta",
        "median": "medyan", "mean": "ortalama",
        "figs": {
            1: "Ayrıştırma. Aynı hacim verildiğinde üç kol yalnızca kutunun nereden geldiği bakımından farklıdır; iki hata terimi, boru hattının dedektörsüz modele karşı açığına tam olarak toplanır.",
            2: "Kesişimin iki yüzü. IoU ile öngörü kapsaması, öngörü hedefi kapsadığında uyuşur, hedefin içinde kaldığında ayrışır — karışım parametresinin hakemlik ettiği bölge budur.",
            3: "Yalnızca bir yarı bozuluyor. Kohort eğitim dağılımından uzaklaştıkça lokalizasyon terimi bir büyüklük mertebesi artarken segmentasyon terimi küçük kalıyor ve işaret değiştiriyor.",
            4: "Kutu neden halka isteyemez. Bir halkanın dar sınırlayıcı dikdörtgeni, kapsadığı diskin dar sınırlayıcı dikdörtgeniyle aynıdır; bu yüzden hiçbir dedektör — eğitilmiş ya da kusursuz — ikisini ayıran bir istem üretemez.",
            5: "Kararsızlık uzaklıkla büyüyor. Yalnızca tohumu farklı dedektörler alan dışında daha çok ayrışıyor ve bu ayrışmayı, tekdüze bir kayma değil, sıfıra yakın çöken birkaç hasta taşıyor.",
            6: "Her sistemde nereye dokunuldu. Gri, yayınlandığı hâliyle; amber, bu çalışmanın değiştirdiği yerler.",
        },
    },
}


def label(lang: str, key: str) -> str:
    return L[lang][key]




def _corrected(path: Path) -> Path:
    return corrected(path)


def _read(path: Path) -> dict:
    path = _corrected(path)
    out = {}
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except Exception:
            continue
        if "error" not in rec:
            out[rec["patient"]] = rec.get("arms", {})
    return out


def wt_decomposition() -> dict:
    """cohort -> arrays for the whole-tumour three-arm comparison.

    Patients are intersected across arms so every term in a row is computed on
    the same set; a mean over different patient sets is not a decomposition.
    """
    out = {}
    for c in COHORTS:
        box = _read(V / f"{c}_box.jsonl")
        direct = _read(V / f"{c}_direct.jsonl")
        ps = sorted(p for p in box
                    if f"oracle:{SEG}" in box[p] and f"pipeline:{SEG}" in box[p]
                    and "direct:seed_42" in direct.get(p, {}))
        out[c] = {
            "patients": ps,
            "pipeline": np.array([box[p][f"pipeline:{SEG}"]["vol_dice"] for p in ps]),
            "oracle": np.array([box[p][f"oracle:{SEG}"]["vol_dice"] for p in ps]),
            "direct": np.array([direct[p]["direct:seed_42"]["regions"]["WT"]["vol_dice"] for p in ps]),
        }
    return out


def tc_summary() -> dict:
    p = R / "tc_detector_summary.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def oracle_regions() -> dict:
    """cohort -> region -> oracle Dice and volume ratio, for the ET argument."""
    out = {}
    for c in COHORTS:
        recs = _read(V / f"{c}_oracle_regions.jsonl")
        key = f"oracle:{SEG}"
        rows = [r[key]["regions"] for r in recs.values() if key in r]
        if not rows:
            continue
        out[c] = {}
        for reg in ("WT", "TC", "ET"):
            vals = [x[reg] for x in rows if reg in x]
            if vals:
                out[c][reg] = {
                    "dice": float(np.mean([v["vol_dice"] for v in vals])),
                    # Median, matching write_report_external.py. ET has
                    # patients whose reference is a few dozen voxels, and a
                    # relative error over those dominates any mean: the mean
                    # here is 0.98 against a median of 0.46 on the same data.
                    "rel_vol_diff": (float(np.median(rvd)) if (rvd := [v["rel_vol_diff"] for v in vals
                                     if v.get("rel_vol_diff") is not None]) else None),
                    "n": len(vals),
                }
    return out


def alpha_sweep() -> dict:
    """Run id -> alpha, C_p, C_t for the twelve single-seed ablation runs, plus
    the replicate spread at the two endpoints."""
    exp = DRIVE_ROOT / "experiments"
    rep = DRIVE_ROOT / "experiments_repeated"
    rows = []
    for d in sorted(exp.iterdir()):
        cfg = d / "config" / "requested_config.json"
        sm = d / "summary" / "run_summary.json"
        if not (cfg.exists() and sm.exists()):
            continue
        c = json.loads(cfg.read_text(encoding="utf-8"))
        if c.get("loss_name") != "icarb" or c.get("overlap_loss_weight") != 3.0:
            continue
        if c.get("experiment_group") != "A_alpha_sweep":
            continue
        a = (c.get("loss_params") or {}).get("alpha")
        if a is None:
            continue
        g = json.loads(sm.read_text(encoding="utf-8"))["geometry"]
        rows.append({"run": c["run_id"], "alpha": float(a),
                     "cp": g["pred_coverage"], "ct": g["target_coverage"],
                     "slug": d.name})
    rows.sort(key=lambda r: r["alpha"])

    spread = {}
    for r in rows:
        if r["alpha"] not in (0.0, 1.0):
            continue
        cps, cts = [r["cp"]], [r["ct"]]
        for s in (42, 62):
            sm = rep / f"seed_{s}" / r["slug"] / "summary" / "run_summary.json"
            if sm.exists():
                g = json.loads(sm.read_text(encoding="utf-8"))["geometry"]
                cps.append(g["pred_coverage"])
                cts.append(g["target_coverage"])
        spread[r["alpha"]] = {"cp": (min(cps), max(cps)), "ct": (min(cts), max(cts)),
                              "n": len(cps)}
    return {"runs": rows, "spread": spread}


def detection_f1() -> dict:
    """Per-cohort detection F1 of the whole-tumour detector, from the box arms."""
    out = {}
    for c in COHORTS:
        box = _read(V / f"{c}_box.jsonl")
        key = f"pipeline:{SEG}"
        vals = [r[key] for r in box.values() if key in r]
        tp = sum(v["tp"] for v in vals)
        fp = sum(v["fp"] for v in vals)
        fn = sum(v["fn"] for v in vals)
        out[c] = 2 * tp / (2 * tp + fp + fn) if tp else float("nan")
    return out


def save(fig, name: str, lang: str) -> list:
    """Write PDF for the manuscript and PNG for everything else."""
    OUT.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext, kw in (("pdf", {}), ("png", {"dpi": 400})):
        p = OUT / f"{name}_{lang}.{ext}"
        fig.savefig(p, **kw)
        paths.append(p)
    return paths
