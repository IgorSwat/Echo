#!/usr/bin/env python3
"""Frame-aligned phonemes from the AR stage's CTC head.

The AR model carries an auxiliary CTC head over its decoder states, upsampled to
``ar_model.ctc.upsample`` times the 12.5 Hz token grid. Scoring it against the
*known* phoneme string and taking the Viterbi path turns that head into a forced
aligner: every frame gets the phoneme being spoken, or the CTC blank.

The frame grid is the one the CTC loss uses at training time, so what comes out
here is exactly what the model was trained against:

    seq     = [BOS, token_0 ... token_{T-1}, EOS]      as scripts/train/ar.py
    inputs  = seq[:-1]                                  -> T + 1 frames
    ctc     = ctc_log_probs(hidden)                     -> u * (T + 1) frames

so an utterance of T codec tokens yields ``u * (T + 1)`` alignment frames at
``12.5 * u`` Hz. This reproduces the layout of the existing ctc_align.npz.

Output is one compressed npz holding every utterance end to end:

    flat     (N,)  int16   all alignments concatenated
    offsets  (M+1,) int64  utterance i is flat[offsets[i]:offsets[i+1]]
    names    (M,)   str    the row's name, as it appears in the csv
    blank    scalar int    the CTC blank id, one past the phoneme alphabet

Usage:
    python scripts/preprocess/ctc_align.py --ar-model checkpoints/ljspeech/echo_ar_best.pt
    python scripts/preprocess/ctc_align.py --ar-model ... --csv data/ljspeech/phonemes_test.csv \
        --out data/ljspeech/ctc_align_test.npz
"""

from __future__ import annotations

import sys
from pathlib import Path

# The shared helpers (__common__, __style__, ...) sit one level up, in scripts/.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import time

import numpy as np
import torch
import torchaudio.functional as AF

from __common__ import (
    REPO_ROOT,
    load_checkpoint,
    load_pairs_csv,
    load_tokenizer,
    print_run_summary,
    save_npz,
    select_device,
)
from __style__ import Colors, print_header, print_info, print_section, print_separator

from echo import config
from echo.ar_model import EchoAR


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Force-align phonemes to frames with the AR model's CTC head."
    )
    p.add_argument("--ar-model", type=str, required=True,
                   help="Path to an EchoAR checkpoint (.pt) trained with the CTC head.")
    p.add_argument("--csv", type=str, default="data/ljspeech/phonemes.csv",
                   help="Phoneme csv to align (default: data/ljspeech/phonemes.csv).")
    p.add_argument("--codec-dir", type=str, default=None,
                   help="Where the codec npz files live (default: <csv dir>/codecs).")
    p.add_argument("--out", type=str, default=None,
                   help="Output npz (default: <csv dir>/ctc_align.npz).")
    p.add_argument("--limit", type=int, default=None,
                   help="Align only the first N rows; for a quick check.")
    p.add_argument("--device", type=str, default=None,
                   help="Override the auto-selected device.")

    return p.parse_args()


@torch.no_grad()
def align_one(
    model: EchoAR,
    codes: torch.Tensor,                                    # (T,) long, layer 0
    phonemes: torch.Tensor,                                 # (P,) long
    device: torch.device,
) -> np.ndarray:
    """The Viterbi path over one utterance: (u * (T + 1),) of phoneme ids and blanks."""
    # The BOS-prefixed input half of the training sequence. EOS is only ever a
    # target, so it never enters the frame states the CTC head reads.
    inputs = torch.cat([
        torch.tensor([config.prosody_bos], device=device), codes.to(device)
    ]).unsqueeze(0)                                         # (1, T + 1)

    _, hidden = model(inputs, phonemes.unsqueeze(0).to(device), return_hidden=True)
    log_probs = model.ctc_log_probs(hidden)                 # (1, u * (T+1), V + 1)

    # forced_align has no MPS kernel, and the Viterbi pass is cheap next to the
    # forward, so it always runs on the host.
    path, _ = AF.forced_align(
        log_probs.float().cpu(), phonemes.unsqueeze(0).cpu(), blank=model.ctc_blank,
    )                                                       # (1, u * (T+1))

    return path[0].numpy().astype(np.int16)


def main() -> None:
    args = _parse_args()
    device = torch.device(args.device) if args.device else select_device()

    csv_path = REPO_ROOT / args.csv if not Path(args.csv).is_absolute() else Path(args.csv)
    data_dir = csv_path.parent
    codec_dir = Path(args.codec_dir) if args.codec_dir else data_dir / "codecs"
    out_path = Path(args.out) if args.out else data_dir / "ctc_align.npz"

    print_header("Echo - CTC Forced Alignment")
    print_separator()

    if not config.ar_model.ctc_enabled:
        raise SystemExit(
            "ar_model.ctc.enabled is false in models/config.json, so the checkpoint "
            "has no CTC head to align with."
        )

    tokenizer = load_tokenizer()
    model = EchoAR().to(device)
    load_checkpoint(model, args.ar_model, device, "EchoAR")
    model.eval()
    if model.ctc_head is None:
        raise SystemExit("the loaded EchoAR has no CTC head")
    # `load_weights` tolerates a checkpoint saved without the CTC branch, which
    # would leave the head at its initialization -- aligning with random weights
    # produces a plausible-looking file that means nothing, so it is refused.
    state = torch.load(args.ar_model, map_location="cpu")
    state = state.get("model", state)
    if not any(k.startswith("ctc_") for k in state):
        raise SystemExit(
            f"{args.ar_model} carries no CTC weights (no 'ctc_*' keys), so its head is "
            f"untrained. Align with a checkpoint trained with ctc_weight > 0."
        )

    rows = load_pairs_csv(csv_path)
    if args.limit:
        rows = rows[: args.limit]

    print_section("Setup")
    print_info("Device", str(device), Colors.OKCYAN)
    print_info("AR checkpoint", args.ar_model)
    print_info("Rows to align", f"{len(rows)} from {csv_path}")
    print_info("Frame rate", f"{12.5 * max(model.ctc_upsample, 1):g} Hz "
                             f"(12.5 Hz x {max(model.ctc_upsample, 1)})", Colors.OKCYAN)
    print_info("Output", str(out_path))
    print_separator()

    flat: list[np.ndarray] = []
    offsets = [0]
    names: list[str] = []
    n_fail = 0
    t_start = time.perf_counter()

    for i, (name, phon) in enumerate(rows):
        try:
            codes = np.load(codec_dir / name)["codes"][0]            # layer 0, (T,)
            ids = torch.tensor(tokenizer.tokenize(phon), dtype=torch.long)
            if ids.numel() == 0:
                raise ValueError("empty phoneme string")
            # CTC needs at least one frame per phoneme, plus a blank between any
            # repeated pair. Without that there is no valid path and the Viterbi
            # pass would fail rather than return a bad alignment.
            frames = max(model.ctc_upsample, 1) * (len(codes) + 1)
            need = len(ids) + int((ids[1:] == ids[:-1]).sum())
            if frames < need:
                raise ValueError(f"{frames} frames cannot hold {need} CTC states")

            path = align_one(model, torch.tensor(codes, dtype=torch.long), ids, device)
        except Exception as e:                                        # noqa: BLE001
            n_fail += 1
            print_info(f"skipped {name}", str(e), Colors.WARNING)
            continue

        flat.append(path)
        offsets.append(offsets[-1] + len(path))
        names.append(name)

        if (i + 1) % 500 == 0:
            print_info(f"{i + 1}/{len(rows)}",
                       f"{time.perf_counter() - t_start:.0f}s", Colors.OKGREEN)

    if not flat:
        raise SystemExit("nothing aligned")

    save_npz(
        out_path,
        flat=np.concatenate(flat).astype(np.int16),
        offsets=np.asarray(offsets, dtype=np.int64),
        names=np.asarray(names),
        blank=np.asarray(model.ctc_blank),
    )

    print_separator()
    print_info("Frames written", f"{offsets[-1]:,}", Colors.OKCYAN)
    print_run_summary(len(names), n_fail, time.perf_counter() - t_start)


if __name__ == "__main__":
    main()
