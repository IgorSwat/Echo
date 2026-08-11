#!/usr/bin/env python3
"""Evaluate an EchoFM checkpoint and write the results to JSON for comparison.

The conditioning is rebuilt from the codecs every run rather than read off disk,
so a result stays reproducible whatever distils happen to be around:

    clean  ground-truth codec -> Mimi -> BlueCodec      (in-distribution, oracle)
    ar     free-running EchoAR -> Mimi -> BlueCodec     (what the pipeline gets)

Both are scored: a checkpoint can look strong on the oracle input and regress on
the real one, and that gap *is* the train/inference mismatch. Three fixed anchors
(the recording, its codec round trip, the conditioning alone) keep runs from
different days comparable even if the ASR model or codec changes underneath.

Usage:
    python scripts/eval/fm.py --model checkpoints/ljspeech/echo_fm_best.pt --tag baseline
    python scripts/eval/fm.py --model ... --tag bridge --norm instance
    python scripts/eval/fm.py --compare checkpoints/ljspeech/eval/*.json
"""

from __future__ import annotations

import sys
from pathlib import Path

# The shared helpers (__common__, __style__, ...) sit one level up, in scripts/.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import json
import subprocess
import warnings
from datetime import datetime, timezone

import librosa
import numpy as np
import torch
import torchaudio
from scipy import linalg
from torch.utils.data import random_split

from __common__ import (
    BLUE_SR,
    MIMI_FPS,
    MIMI_SR,
    REPO_ROOT,
    decode_mimi,
    load_checkpoint,
    load_pairs_csv,
    load_tokenizer,
    select_device,
)
from __eval__ import ASR, edit_ops, normalize_text
from __style__ import Colors, print_header, print_info, print_section, print_separator

from echo import config
from echo.ar_model import EchoAR
from echo.fm_model import EchoFM

# Sampler settings scored on every run. Keeping the list fixed is what lets two
# checkpoints be compared at matching settings later; each model can still be
# read at its own best row.
_CONFIGS = [
    {"name": "s8",      "steps": 8,  "cfg": 1.0},
    {"name": "s16",     "steps": 16, "cfg": 1.0},
    {"name": "s32",     "steps": 32, "cfg": 1.0},
    {"name": "s8_cfg3", "steps": 8,  "cfg": 3.0},
    {"name": "s8_mid",  "steps": 8,  "cfg": 1.0, "solver": "midpoint"},
]

_ANCHORS = ("real", "codec_roundtrip", "ar_stage", "conditioning_clean", "conditioning_ar")


# ---------
# Metrics
# ---------

def _mel(w: np.ndarray, sr: int) -> np.ndarray:
    """Log-mel with a *fixed* reference, so values compare across runs."""
    if sr != BLUE_SR:
        w = librosa.resample(np.asarray(w, np.float32), orig_sr=sr, target_sr=BLUE_SR)
    spec = librosa.feature.melspectrogram(
        y=np.asarray(w, np.float32), sr=BLUE_SR, n_mels=80, hop_length=256
    )

    return librosa.power_to_db(spec, ref=1e-6)


def _mel_dist(a: np.ndarray, sr_a: int, b: np.ndarray, dtw: bool) -> float:
    """Mean absolute log-mel difference; DTW-aligned when timing is free-running."""
    A, B = _mel(a, sr_a), _mel(b, BLUE_SR)
    if dtw:
        _, wp = librosa.sequence.dtw(A, B, metric="euclidean")
        return float(np.mean([np.abs(A[:, i] - B[:, j]).mean() for i, j in wp]))

    T = min(A.shape[1], B.shape[1])

    return float(np.abs(A[:, :T] - B[:, :T]).mean())


def _frechet(A: np.ndarray, B: np.ndarray) -> float:
    """Frechet distance between two frame sets — sees blur, which MSE rewards."""
    mu1, mu2 = A.mean(0), B.mean(0)
    c1, c2 = np.cov(A, rowvar=False), np.cov(B, rowvar=False)
    cc, _ = linalg.sqrtm(c1.dot(c2), disp=False)

    return float(((mu1 - mu2) ** 2).sum() + np.trace(c1 + c2 - 2 * np.real(cc)))


def _feats(a: np.ndarray) -> np.ndarray:
    """Frames stacked with their first differences: marginals *and* temporal detail."""

    return np.concatenate([a[1:], a[1:] - a[:-1]], axis=1)


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       cwd=REPO_ROOT, text=True).strip()
    except Exception:                                                # noqa: BLE001
        return "unknown"


def _val_files(num: int) -> list[str]:
    """The first `num` files of the flow-matching val split, deterministically."""
    cfg = config.training.fm
    names = [name for name, _ in load_pairs_csv(REPO_ROOT / cfg.data_dir / "phonemes.csv")]
    n = len(names)
    val_len = int(n * cfg.val_ratio)
    _, va = random_split(range(n), [n - val_len, val_len],
                         generator=torch.Generator().manual_seed(cfg.seed))

    return [names[i] for i in list(va)[:num]]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate an EchoFM checkpoint.")
    parser.add_argument("--model", type=str, help="Path to an EchoFM checkpoint (.pt).")
    parser.add_argument("--ar-model", type=str,
                        default="checkpoints/ljspeech/echo_ar_best_ctc02_corr.pt",
                        help="EchoAR checkpoint used to build the pipeline conditioning.")
    parser.add_argument("--tag", type=str, default=None,
                        help="Short name for this result (default: the checkpoint stem).")
    parser.add_argument("--seed", type=int, default=0,
                        help="Seeds the source dither, so a run is reproducible (the transport "
                             "is stochastic whenever fm_model.source_noise > 0).")
    parser.add_argument("--num-files", type=int, default=12,
                        help="Val-split utterances to score (default: 12).")
    parser.add_argument("--norm", choices=("dataset", "instance"), default=None,
                        help="Normalization the checkpoint was TRAINED with. Defaults to "
                             "config.latent_norm, which may not match an older checkpoint — "
                             "pass it explicitly when they differ.")
    parser.add_argument("--out-dir", type=str, default=None,
                        help="Where to write <tag>.json (default: <output_dir>/eval).")
    parser.add_argument("--save-audio", action="store_true",
                        help="Also write the generated wavs next to the JSON.")
    parser.add_argument("--compare", nargs="+", default=None,
                        help="Print a comparison of existing result JSONs and exit.")
    args = parser.parse_args()

    if not args.compare and not args.model:
        parser.error("--model is required (or use --compare)")

    return args


def main() -> None:
    args = _parse_args()

    if args.compare:
        _print_comparison(args.compare)
        return

    warnings.filterwarnings("ignore")
    device = select_device()
    norm = args.norm or config.latent_norm
    tag = args.tag or Path(args.model).stem
    out_dir = (Path(args.out_dir) if args.out_dir
               else REPO_ROOT / config.training.fm.output_dir / "eval")
    out_dir.mkdir(parents=True, exist_ok=True)

    from bluecodec import BlueCodec                                  # noqa: PLC0415
    from transformers import MimiModel                               # noqa: PLC0415

    data_dir = REPO_ROOT / config.training.fm.data_dir
    print_header("EchoFM - Evaluation")
    print_separator()
    print_section("Setup")
    print_info("Checkpoint", args.model, Colors.OKCYAN)
    print_info("Tag", tag)
    print_info("Device", str(device))
    print_info("Source noise", f"sigma {config.fm_model.source_noise:g}"
               + ("" if config.fm_model.source_noise > 0 else "  (deterministic transport)"))
    print_info("Normalization", f"{norm}"
               + ("" if args.norm else "  (from config; pass --norm if the checkpoint differs)"),
               Colors.OKCYAN if args.norm else Colors.WARNING)

    # --- Data ---------------------------------------------------------------
    stats = np.load(data_dir / "latents/latent_stats.npz")
    d_mean = torch.tensor(stats["mean"], device=device).float().view(1, 1, -1)
    d_std = torch.tensor(stats["std"], device=device).float().view(1, 1, -1)

    phon = dict(load_pairs_csv(data_dir / "phonemes.csv"))
    transcripts = {name.replace(".wav", ""): text
                   for name, text in load_pairs_csv(data_dir / "metadata.csv")}
    files = _val_files(args.num_files)
    print_info("Utterances",
               f"{len(files)} from the val split (seed {config.training.fm.seed})")

    # --- Models -------------------------------------------------------------
    tokenizer = load_tokenizer()
    fm = EchoFM().to(device).eval()
    ckpt = load_checkpoint(fm, args.model, device, "EchoFM", strict=False)
    ar = EchoAR().to(device).eval()
    load_checkpoint(ar, REPO_ROOT / args.ar_model, device, "EchoAR")
    mimi = MimiModel.from_pretrained("kyutai/mimi").to(device).eval()
    blue = BlueCodec.from_pretrained("notmax123/blue-codec", device=str(device))
    asr = ASR("auto", None, "en", device)
    print_info("Trained", f"epoch {ckpt.get('epoch', '?')}, "
                          f"val_loss {ckpt.get('val_loss', float('nan')):.4f}")

    def normalize(raw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """(normalized, mean, std) — instance mode takes its numbers from the distil."""
        if norm == "instance":
            m = raw.mean(dim=-2, keepdim=True)
            s = raw.std(dim=-2, keepdim=True).clamp_min(1e-5)
        else:
            m, s = d_mean, d_std

        return (raw - m) / s, m, s

    # --- Score --------------------------------------------------------------
    per_file: list[dict] = []
    pools: dict[str, list[np.ndarray]] = {}
    target_pool: list[np.ndarray] = []
    print_section("Scoring")

    for idx, name in enumerate(files):
        stem = name.replace(".npz", "")
        row: dict = {"stem": stem}
        real, _ = librosa.load(str(data_dir / "audio" / f"{stem}.wav"), sr=BLUE_SR, mono=True)
        ref_words = normalize_text(transcripts.get(stem, ""))
        row["ref_words"] = len(ref_words)

        with np.load(data_dir / "latents" / name) as z:
            target_raw = torch.from_numpy(z["latents"]).float()[None].to(device)
        with np.load(data_dir / "codecs" / name) as z:
            gt_codes = torch.from_numpy(
                z["codes"][: EchoAR.NUM_TOKEN_LAYERS].astype(np.int64))[None].to(device)
        text_ids = torch.tensor([tokenizer.tokenize(phon[name])], dtype=torch.long, device=device)

        def score(wav: np.ndarray, sr: int, dtw: bool) -> dict:
            hyp = normalize_text(asr.transcribe(wav, sr))
            s_, d_, i_ = edit_ops(ref_words, hyp)
            return {"sub": s_, "del": d_, "ins": i_,
                    "wer": (s_ + d_ + i_) / max(len(ref_words), 1),
                    "mel": _mel_dist(wav, sr, real, dtw)}

        with torch.no_grad():
            # --- anchors, so results stay comparable across days -------------
            row["anchor_real"] = score(real, BLUE_SR, False)
            rt = blue.decode(blue.encode(torch.from_numpy(real)[None].to(device)))
            row["anchor_codec_roundtrip"] = score(rt.reshape(-1).float().cpu().numpy(),
                                                  BLUE_SR, False)

            # --- conditioning: clean (oracle) and AR (the real pipeline) -----
            clean_wav = decode_mimi(mimi, gt_codes)
            clean_raw = blue.encode(torchaudio.functional.resample(
                clean_wav, MIMI_SR, BLUE_SR)[..., : len(real)]).transpose(1, 2).float()

            ar_codes = ar.generate(text_ids, max_frames=1000)
            ar_frames = int(ar_codes.shape[1])
            ar_wav = decode_mimi(mimi, ar_codes.transpose(1, 2))
            ar_raw = blue.encode(torchaudio.functional.resample(
                ar_wav, MIMI_SR, BLUE_SR)).transpose(1, 2).float()
            row["ar_frames"], row["gt_frames"] = ar_frames, int(gt_codes.shape[2])
            row["anchor_ar_stage"] = score(ar_wav.reshape(-1).float().cpu().numpy(),
                                           MIMI_SR, True)

            for src, raw, dtw in (("clean", clean_raw, False), ("ar", ar_raw, True)):
                cond, m, s = normalize(raw)
                row[f"anchor_conditioning_{src}"] = score(
                    blue.decode(raw.transpose(1, 2)).reshape(-1).float().cpu().numpy(),
                    BLUE_SR, dtw)

                if src == "clean":              # aligned: latent metrics are meaningful
                    tgt = (target_raw.transpose(1, 2) - m) / s
                    T = min(tgt.shape[1], cond.shape[1])
                    target_pool.append(_feats(tgt[0, :T].cpu().numpy()))

                for spec in _CONFIGS:
                    gen = torch.Generator(device=device).manual_seed(args.seed)
                    out = fm.sample(text_ids, cond, spec["steps"], spec["cfg"],
                                    spec.get("solver", "euler"), generator=gen)
                    wav = blue.decode((out * s + m).transpose(1, 2)).reshape(-1)
                    wav = wav.float().cpu().numpy()
                    if src == "ar":             # trim to what the AR actually produced
                        wav = wav[: int(ar_frames / MIMI_FPS * BLUE_SR)]

                    key = f"{src}__{spec['name']}"
                    row[key] = score(wav, BLUE_SR, dtw)
                    if src == "clean":
                        o = out[0, :T].cpu().numpy()
                        row[key]["latent_mse"] = float(((out[:, :T] - tgt[:, :T]) ** 2).mean())
                        row[key]["detail"] = float(np.diff(o, axis=0).std())
                        row[key]["std"] = float(o.std())
                        pools.setdefault(key, []).append(_feats(o))
                    if args.save_audio:
                        import soundfile as sf                       # noqa: PLC0415

                        (out_dir / tag).mkdir(exist_ok=True)
                        sf.write(str(out_dir / tag / f"{stem}_{key}.wav"), wav, BLUE_SR)

        per_file.append(row)
        print_info(f"[{idx + 1}/{len(files)}] {stem}",
                   f"AR {row['anchor_ar_stage']['wer']:.3f} | "
                   f"clean/s8 {row['clean__s8']['wer']:.3f} | "
                   f"ar/s8 {row['ar__s8']['wer']:.3f}")

    # --- Aggregate ----------------------------------------------------------
    total_words = sum(r["ref_words"] for r in per_file)
    X = np.concatenate(target_pool, 0)
    dX = float(np.concatenate([t[:, 24:] for t in target_pool], 0).std())
    sX = float(np.concatenate([t[:, :24] for t in target_pool], 0).std())

    def agg(key: str) -> dict:
        rows = [r[key] for r in per_file if key in r]
        out = {
            "wer": sum(r["sub"] + r["del"] + r["ins"] for r in rows) / max(total_words, 1),
            "del_rate": sum(r["del"] for r in rows) / max(total_words, 1),
            "mel": float(np.mean([r["mel"] for r in rows])),
        }
        if key in pools:
            A = np.concatenate(pools[key], 0)
            out["frechet"] = _frechet(A, X)
            out["latent_mse"] = float(np.mean([r["latent_mse"] for r in rows]))
            out["detail_ratio"] = float(np.mean([r["detail"] for r in rows])) / dX
            out["std_ratio"] = float(np.mean([r["std"] for r in rows])) / sX
        return out

    keys = ([f"anchor_{k}" for k in _ANCHORS]
            + [f"{src}__{c['name']}" for src in ("clean", "ar") for c in _CONFIGS])
    summary = {k: agg(k) for k in keys}

    result = {
        "tag": tag,
        "checkpoint": str(args.model),
        "ar_checkpoint": args.ar_model,
        "epoch": ckpt.get("epoch"), "val_loss": ckpt.get("val_loss"),
        "normalization": norm,
        "source_noise": config.fm_model.source_noise,
        "seed": args.seed,
        "num_files": len(files), "files": [r["stem"] for r in per_file],
        "sampler_configs": _CONFIGS,
        "git_commit": _git_commit(),
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "latent_norm_config": config.latent_norm,
        "summary": summary,
        "per_file": per_file,
    }
    out_path = out_dir / f"{tag}.json"
    out_path.write_text(json.dumps(result, indent=1))

    print_section("Results")
    print(f"  {'variant':28s}{'WER':>8}{'mel':>8}{'Frechet':>10}{'detail':>8}{'MSE':>8}")
    for k in keys:
        v = summary[k]
        print(f"  {k:28s}{v['wer']:8.3f}{v['mel']:8.2f}"
              f"{v.get('frechet', float('nan')):10.3f}{v.get('detail_ratio', float('nan')):8.3f}"
              f"{v.get('latent_mse', float('nan')):8.3f}")
    print_separator()
    print_info("Saved", str(out_path), Colors.OKGREEN)


def _print_comparison(paths: list[str]) -> None:
    """Side-by-side of saved runs, aligned by sampler *spec* rather than by name.

    Two checkpoints may name their sampler settings differently, so matching on
    (steps, cfg, solver) compares equal compute on each model's own transport,
    which is the honest pairing; rows without a counterpart are listed apart.
    """
    runs = [json.loads(Path(p).read_text()) for p in paths]
    print_header("EchoFM - Comparison")
    for r in runs:
        print_info(r["tag"], f"{r['checkpoint']}  (epoch {r['epoch']}, "
                             f"val_loss {r['val_loss']:.4f}, norm {r['normalization']}, "
                             f"{r['num_files']} files, commit {r['git_commit']})")

    base = runs[0]
    for r in runs[1:]:
        if r["files"] != base["files"]:
            print_info("Warning", f"{r['tag']} scored different utterances than {base['tag']}; "
                                  "the numbers are not comparable", Colors.FAIL)

    def spec_key(c: dict) -> tuple:
        # `init` is deliberately not part of the key: where the transport starts
        # is the model's architecture, not a knob. `start_t` *is* included — it
        # marks a deliberately off-distribution run.
        return (c["steps"], c["cfg"], c.get("solver", "euler"), c.get("start_t", 0.0))

    # Anchors first: identical inputs, so any movement is measurement noise.
    anchors = [k for k in base["summary"] if k.startswith("anchor_")]
    rows: list[tuple[str, list[str | None]]] = [
        (a, [a if a in r["summary"] else None for r in runs]) for a in anchors
    ]
    shared, unmatched = [], []
    for c in base["sampler_configs"]:
        for src in ("clean", "ar"):
            keys = []
            for r in runs:
                match = next(
                    (rc for rc in r["sampler_configs"] if spec_key(rc) == spec_key(c)), None
                )
                keys.append(f"{src}__{match['name']}" if match else None)
            label = f"{src} · {c['steps']} steps, cfg {c['cfg']:g}, {c.get('solver', 'euler')}"
            (shared if all(keys) else unmatched).append((label, keys))
    rows += shared

    width = max(len(r["tag"]) for r in runs) + 2
    for metric in ("wer", "mel", "frechet", "detail_ratio"):
        printed = False
        for label, keys in rows:
            vals = [r["summary"].get(k, {}).get(metric) if k else None
                    for k, r in zip(keys, runs)]
            if all(v is None or (isinstance(v, float) and np.isnan(v)) for v in vals):
                continue
            if not printed:
                print_section(metric)
                print(f"  {'':38s}" + "".join(f"{r['tag']:>{width}s}" for r in runs))
                printed = True
            cells = "".join(
                f"{(v if v is not None else float('nan')):>{width}.3f}" for v in vals
            )
            print(f"  {label:38s}{cells}")

    if unmatched:
        print_section("no counterpart (listed for reference)")
        for label, keys in unmatched:
            have = [f"{r['tag']}:{k.split('__')[1]}" for k, r in zip(keys, runs) if k]
            print(f"  {label:38s}  only in {', '.join(have)}")


if __name__ == "__main__":
    main()
