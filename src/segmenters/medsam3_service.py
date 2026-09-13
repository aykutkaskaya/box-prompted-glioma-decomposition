"""MedSAM3 / MedSAM3+LoRA sidecar.

Two reasons this is a separate process rather than part of the console:

  * MedSAM3 needs the bundled SAM3 code and a dependency set that does not agree
    with the console's; and
  * the LoRA pipeline is the author's own research code, which is marked
    "ported VERBATIM … nothing here may be improved, reordered or simplified".
    So this file *imports* that code rather than reimplementing any of it. The
    numerics belong to the research; only process plumbing lives here.

Reused verbatim from the MedSAM3 research project:
    app.imaging.preprocessing   normalize_volume, build_pseudo_rgb  (FLAIR/T1ce/T2, z-score ±3)
    app.inference.research_core build_medical_sam3, inject_lora, restore_lora, predict_volume

Two operating modes, both selectable per request:
    box   — plain MedSAM3 prompted with a detector box; directly comparable with
            SAM1/SAM2 under the same GT-box protocol.
    text  — the research pipeline: detector-free, one text prompt per region
            (WT/TC/ET). This is what the LoRA adapters were trained for; the
            research explicitly treats box prompts as an oracle upper bound and
            excludes them from this path.

Run with the isolated venv:
    drive/models/medical_sam3/venv/Scripts/python.exe drive/src/segmenters/medsam3_service.py
"""
from __future__ import annotations

import base64
import io
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
DRIVE = Path(os.environ.get("DRIVE_ROOT", HERE.parent.parent))

# The research project supplies both the pipeline code and its pinned SAM3 checkout.
# no default: the research project is a separate checkout, and guessing a
# path only turns a clear error into a confusing one
PROJECT = Path(os.environ.get("MEDSAM3_PROJECT")
               or os.environ.get("MEDSAM3_DIR", "medsam3-not-set"))
SAM3_SRC = Path(os.environ.get("MEDSAM3_SAM3_SRC", PROJECT / "external" / "Medical-SAM3"))
BASE_CKPT = Path(os.environ.get("MEDSAM3_CHECKPOINT", DRIVE / "models" / "medical_sam3" / "checkpoint_2D.pt"))
LORA_DIR = Path(os.environ.get("MEDSAM3_LORA_DIR", DRIVE / "models" / "lora"))
PORT = int(os.environ.get("MEDSAM3_PORT", 8020))

for p in (SAM3_SRC, PROJECT / "demo" / "backend"):
    if p.is_dir():
        sys.path.insert(0, str(p))

import torch  # noqa: E402
from fastapi import FastAPI, HTTPException  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

app = FastAPI(title="MedSAM3 sidecar", version="1.0.0")
_S: Dict[str, object] = {"model": None, "adapter": None, "error": None, "regions": None}


def _log(m: str) -> None:
    print(f"[medsam3] {m}", flush=True)


def _core():
    from app.inference import research_core as rc
    return rc


def adapters() -> List[dict]:
    """LoRA checkpoints on disk. seed_52 is experimental in the research config."""
    out = []
    if LORA_DIR.is_dir():
        for d in sorted(LORA_DIR.iterdir()):
            ck = sorted(d.glob("*.pt")) if d.is_dir() else []
            if ck:
                out.append({
                    "id": d.name, "path": str(ck[0]),
                    "size_mb": round(ck[0].stat().st_size / 1048576, 1),
                    "experimental": d.name == "seed_52",
                })
    return out


def _release() -> None:
    """Drop the model and hand the memory back, so a retry can start clean."""
    _S["model"] = None
    _S["adapter"] = None
    _S.pop("pristine", None)
    try:
        import gc
        gc.collect()
        torch.cuda.empty_cache()
    except Exception:
        pass
    _log("model released; GPU cache cleared")


def ensure_model(adapter_id: Optional[str]):
    """Load the base once; swap the LoRA weights in place when the adapter changes.

    `restore_lora` overwrites only the LoRA parameters, so switching adapters
    costs a tensor copy rather than a 9.4 GB reload. Base weights are untouched.
    """
    rc = _core()
    want = adapter_id or None

    if _S["model"] is None:
        if not BASE_CKPT.exists():
            raise HTTPException(500, f"MedSAM3 base checkpoint missing: {BASE_CKPT}")
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        free_note = ""
        if dev == "cuda":
            try:
                free_b, total_b = torch.cuda.mem_get_info()
                free_note = f" ({free_b/2**30:.1f} of {total_b/2**30:.1f} GiB free)"
                if free_b < 5 * 2**30:
                    _log(f"WARNING: only {free_b/2**30:.1f} GiB free — MedSAM3 needs about 5 GiB. "
                         f"Another process is probably holding the card.")
            except Exception:
                pass
        _log(f"building MedSAM3 base on {dev} from {BASE_CKPT.name}{free_note}")
        t0 = time.time()
        try:
            model, missing, unexpected = rc.build_medical_sam3(str(BASE_CKPT), device=dev, train_mode=True)
        except torch.cuda.OutOfMemoryError as e:
            _release()
            raise HTTPException(507, f"Out of GPU memory while loading MedSAM3: {e}") from e
        if missing or unexpected:
            _log(f"WARNING: state dict missing={len(missing)} unexpected={len(unexpected)}")
        model, trainable, non_lora = rc.inject_lora(model)
        if non_lora:
            raise HTTPException(500, f"{len(non_lora)} non-LoRA parameters are trainable; base not frozen")
        # Keep the freshly-initialised (zero-effect) LoRA state so "plain MedSAM3"
        # is reachable again after an adapter has been loaded.
        _S["pristine"] = rc.lora_state(model)
        model.eval()
        _S["model"] = model
        _S["adapter"] = None
        _S["device"] = dev
        _log(f"base ready in {time.time()-t0:.1f}s ({trainable:,} LoRA params)")

    if _S["adapter"] == want:
        return _S["model"]

    model = _S["model"]
    if want is None:
        rc.restore_lora(model, _S["pristine"])
        _S["adapter"] = None
        _log("adapter detached (plain MedSAM3)")
        return model

    entry = next((a for a in adapters() if a["id"] == want), None)
    if entry is None:
        raise HTTPException(404, f"No LoRA adapter '{want}' under {LORA_DIR}")
    ck = torch.load(entry["path"], map_location="cpu", weights_only=False)
    sd = ck["lora_state_dict"]
    stored = ck.get("lora_state_sha256")
    computed = rc.lora_state_hash(sd)
    if stored and stored != computed:
        raise HTTPException(500, f"{want}: lora_state_sha256 mismatch — checkpoint may be corrupt")
    rc.restore_lora(model, {k: v for k, v in sd.items()})
    _S["adapter"] = want
    _log(f"adapter {want} active (seed={ck.get('seed')}, step={ck.get('step')}, hash ok)")
    return model


class BoxRequest(BaseModel):
    image_png_b64: str = Field(..., description="RGB slice PNG, base64")
    box: List[float] = Field(..., description="xyxy pixels")
    adapter: Optional[str] = None


class VolumeRequest(BaseModel):
    """Detector-free path: the research's own text-per-region pipeline."""
    prgb_npy_b64: str = Field(..., description="float32 (3,H,W,Z) pseudo-RGB .npy, base64")
    adapter: Optional[str] = Field(None, description="seed_42 | seed_62 | seed_52 | null")
    regions: List[str] = Field(default_factory=lambda: ["WT"], description="subset of WT/TC/ET")
    batch_size: int = Field(2, ge=1, le=8)
    threshold: Optional[float] = None


class ComposedRequest(BaseModel):
    """The adapter project's recommended prompt protocol, for a sensitivity check.

    The deployed path asks for WT/TC/ET by name, and two of those three names are
    absent from the model's training query vocabulary. The source project instead
    recommends prompting for the three raw BraTS classes, whose names are all in
    vocabulary, and composing the regions from them. Nothing else differs: same
    model, same adapter, same interpolate-sigmoid-threshold order.
    """
    prgb_npy_b64: str
    adapter: Optional[str] = None
    batch_size: int = Field(2, ge=1, le=8)
    threshold: Optional[float] = None
    queries: List[str] = Field(
        default_factory=lambda: ["Enhancing tissue",
                                 "Non-enhancing tumor core",
                                 "Surrounding non-enhancing FLAIR hyperintensity"])
    # region -> indices into `queries` whose masks are unioned
    compose: dict = Field(default_factory=lambda: {"ET": [0], "TC": [0, 1],
                                                   "WT": [0, 1, 2]})


@app.post("/segment/volume_composed")
def segment_volume_composed(req: ComposedRequest):
    """Same inference as /segment_volume with a different prompt protocol."""
    import torch
    import numpy as _np
    rc = _core()
    model = ensure_model(req.adapter)
    prgb = np.load(io.BytesIO(base64.b64decode(req.prgb_npy_b64)), allow_pickle=False)
    if prgb.ndim != 4 or prgb.shape[0] != 3:
        raise HTTPException(400, f"pseudo-RGB must be (3,H,W,Z); got {prgb.shape}")
    prgb = prgb.astype(_np.float32)
    H, W, Z = prgb.shape[1], prgb.shape[2], prgb.shape[3]
    thr = req.threshold if req.threshold is not None else rc.THRESHOLD
    dev = str(_S.get("device", "cuda"))
    q = list(req.queries)
    raw = _np.zeros((len(q), H, W, Z), dtype=bool)
    fin = True
    t0 = time.time()
    with torch.inference_mode():
        for s0 in range(0, Z, req.batch_size):
            ii = list(range(s0, min(s0 + req.batch_size, Z)))
            imgs = torch.stack([rc.model_input_for_slice(prgb, k) for k in ii]).to(dev)
            with rc._autocast(dev):
                out = rc.unwrap_out(model(rc.make_text_batch(imgs, q, device=dev)))
                sem = out["semantic_seg"]
                lo = sem.view(imgs.shape[0], len(q), *sem.shape[-2:])
            lf = lo.float()
            if not bool(torch.isfinite(lf).all()):
                fin = False
            pr = (torch.nn.functional.interpolate(
                lf, (H, W), mode="bilinear", align_corners=False).sigmoid() > thr)
            pr = pr.cpu().numpy()
            for j, k in enumerate(ii):
                for qi in range(len(q)):
                    raw[qi, :, :, k] = pr[j, qi]
            del imgs, lo, lf, pr
    payload = {}
    for region, idx in req.compose.items():
        m = _np.zeros((H, W, Z), dtype=bool)
        for qi in idx:
            m |= raw[qi]
        buf = io.BytesIO()
        np.save(buf, np.packbits(m, axis=None), allow_pickle=False)
        payload[region] = {
            "packed_npy_b64": base64.b64encode(buf.getvalue()).decode("utf-8"),
            "shape": list(m.shape), "voxels": int(m.sum()),
        }
    return {
        "regions": payload, "all_logits_finite": bool(fin),
        "elapsed_ms": int((time.time() - t0) * 1000), "adapter": _S["adapter"],
        "threshold": thr, "mode": "composed_per_raw_label",
        "queries": q, "compose": req.compose,
        "note": "detector-free; regions composed by union of raw-class queries",
    }


class BoxVolumeRequest(BaseModel):
    """MedSAM3 given a box per slice, so the missing fourth arm can be scored.

    The deployed detector-free path feeds `input_boxes` a zero-length tensor:
    the slot is wired and fed nothing. This fills it. Boxes arrive as
    [cx, cy, w, h] normalised to [0, 1], the convention the processor's own
    `add_geometric_prompt` documents, one box per slice; slices with no box get
    an empty prediction rather than a text-only fallback, so the arm measures
    box prompting and not a mixture.
    """
    prgb_npy_b64: str
    boxes: dict          # {slice index (str) -> [cx, cy, w, h]}
    adapter: Optional[str] = None
    query: str = "Whole tumor"
    batch_size: int = Field(2, ge=1, le=8)
    threshold: Optional[float] = None


@app.post("/segment/volume_box")
def segment_volume_box(req: BoxVolumeRequest):
    import torch
    import numpy as _np
    rc = _core()
    model = ensure_model(req.adapter)
    prgb = np.load(io.BytesIO(base64.b64decode(req.prgb_npy_b64)), allow_pickle=False)
    prgb = prgb.astype(_np.float32)
    H, W, Z = prgb.shape[1], prgb.shape[2], prgb.shape[3]
    thr = req.threshold if req.threshold is not None else rc.THRESHOLD
    dev = str(_S.get("device", "cuda"))
    out = _np.zeros((H, W, Z), dtype=bool)
    boxes = {int(k): v for k, v in req.boxes.items()}
    fin = True
    t0 = time.time()

    from sam3.model.data_misc import BatchedDatapoint, FindStage
    with torch.inference_mode():
        for z in sorted(boxes):
            img = rc.model_input_for_slice(prgb, z).unsqueeze(0).to(dev)
            b = torch.tensor(boxes[z], device=dev, dtype=torch.float32).view(1, 1, 4)
            fs = FindStage(
                img_ids=torch.zeros(1, dtype=torch.long, device=dev),
                text_ids=torch.zeros(1, dtype=torch.long, device=dev),
                input_boxes=b,
                # padding mask: 1 means MASKED OUT (collator.py:310 'Masked-out
                # regions indicated by 1s'; it reaches attention as
                # key_padding_mask). A valid box therefore carries 0, which is
                # what the upstream append_boxes fills in when mask is None.
                input_boxes_mask=torch.zeros(1, 1, dtype=torch.bool, device=dev),
                input_boxes_label=torch.ones(1, 1, dtype=torch.long, device=dev),
                input_points=torch.zeros(0, 1, 2, device=dev),
                input_points_mask=torch.zeros(1, 0, dtype=torch.bool, device=dev),
                object_ids=None)
            dp = BatchedDatapoint(img_batch=img, find_text_batch=[req.query],
                                  find_inputs=[fs], find_targets=[None],
                                  find_metadatas=[None])
            with rc._autocast(dev):
                o = rc.unwrap_out(model(dp))
                sem = o["semantic_seg"]
                lo = sem.view(1, 1, *sem.shape[-2:])
            lf = lo.float()
            if not bool(torch.isfinite(lf).all()):
                fin = False
            pr = (torch.nn.functional.interpolate(
                lf, (H, W), mode="bilinear", align_corners=False).sigmoid() > thr)
            out[:, :, z] = pr[0, 0].cpu().numpy()
            del img, lo, lf, pr

    buf = io.BytesIO()
    np.save(buf, np.packbits(out, axis=None), allow_pickle=False)
    return {"packed_npy_b64": base64.b64encode(buf.getvalue()).decode("utf-8"),
            "shape": list(out.shape), "voxels": int(out.sum()),
            "slices_prompted": len(boxes), "all_logits_finite": bool(fin),
            "adapter": _S["adapter"], "threshold": thr, "query": req.query,
            "elapsed_ms": int((time.time() - t0) * 1000)}


@app.get("/health")
async def health():
    return {
        "status": "error" if _S["error"] else ("ready" if _S["model"] else "idle"),
        "error": _S["error"],
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "base_checkpoint": str(BASE_CKPT), "base_present": BASE_CKPT.exists(),
        "project": str(PROJECT), "project_present": (PROJECT / "demo" / "backend").is_dir(),
        "sam3_src": str(SAM3_SRC), "sam3_present": SAM3_SRC.is_dir(),
        "active_adapter": _S["adapter"],
        "adapters": adapters(),
        "modes": ["box", "text"],
    }


@app.post("/segment/box")
def segment_box(req: BoxRequest):
    """Box-prompted MedSAM3 — the mode that lines up with SAM1/SAM2."""
    try:
        ensure_model(req.adapter)
    except HTTPException:
        raise
    except Exception as e:
        _S["error"] = str(e)
        raise HTTPException(500, f"MedSAM3 could not be prepared: {e}")

    from sam3_inference import SAM3Model  # noqa: F401  (import guard: repo present)
    raise HTTPException(
        501,
        "Box-prompted MedSAM3 is not wired to the research model wrapper yet. "
        "Use /segment/volume (the pipeline the LoRA adapters were trained for), "
        "or run plain MedSAM3 through the reference inference script.")


@app.post("/segment/volume")
def segment_volume(req: VolumeRequest):   # sync: FastAPI runs it in a threadpool, keeping /health alive
    """The research pipeline verbatim: text prompt per region, no detector, no ground truth."""
    rc = _core()
    try:
        model = ensure_model(req.adapter)
    except HTTPException:
        raise
    except Exception as e:
        _S["error"] = str(e)
        _log(f"load failed: {e}")
        raise HTTPException(500, f"MedSAM3 could not be prepared: {e}")

    prgb = np.load(io.BytesIO(base64.b64decode(req.prgb_npy_b64)), allow_pickle=False)
    if prgb.ndim != 4 or prgb.shape[0] != 3:
        raise HTTPException(400, f"pseudo-RGB must be (3,H,W,Z); got {prgb.shape}")

    t0 = time.time()
    try:
        vols, finite = rc.predict_volume(
            model, prgb.astype(np.float32),
            device=str(_S.get("device", "cuda")),
            batch_size=req.batch_size,
            threshold=req.threshold if req.threshold is not None else rc.THRESHOLD,
        )
    except torch.cuda.OutOfMemoryError:
        _release()
        raise HTTPException(507, "Out of GPU memory during inference; the model was released. "
                                 "Free the card or lower batch_size, then retry.")
    except RuntimeError as e:
        # An illegal memory access poisons the CUDA context for this process:
        # nothing works afterwards, so drop everything rather than serve garbage.
        if "CUDA" in str(e):
            _release()
            raise HTTPException(507, f"CUDA failure; the model was released: {e}") from e
        raise
    elapsed = int((time.time() - t0) * 1000)

    wanted = [r for r in req.regions if r in rc.REGION_ORDER] or ["WT"]
    payload = {}
    for r in wanted:
        buf = io.BytesIO()
        np.save(buf, np.packbits(vols[r], axis=None), allow_pickle=False)
        payload[r] = {
            "packed_npy_b64": base64.b64encode(buf.getvalue()).decode("utf-8"),
            "shape": list(vols[r].shape),
            "voxels": int(vols[r].sum()),
        }

    return {
        "regions": payload, "all_logits_finite": bool(finite),
        "elapsed_ms": elapsed, "adapter": _S["adapter"],
        "threshold": req.threshold if req.threshold is not None else rc.THRESHOLD,
        "mode": "text_per_region",
        "note": "detector-free; WT/TC/ET are independent binaries, no hierarchy enforcement",
    }


if __name__ == "__main__":
    import uvicorn
    _log(f"project   = {PROJECT} ({'ok' if (PROJECT / 'demo' / 'backend').is_dir() else 'MISSING'})")
    _log(f"sam3 src  = {SAM3_SRC} ({'ok' if SAM3_SRC.is_dir() else 'MISSING'})")
    _log(f"base ckpt = {BASE_CKPT} ({'ok' if BASE_CKPT.exists() else 'MISSING'})")
    _log(f"adapters  = {[a['id'] for a in adapters()]}")
    uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="warning")
