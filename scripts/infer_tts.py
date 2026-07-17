#!/usr/bin/env python3
"""Full AR + NAR text-to-speech inference (VALL-E style voice cloning).

In the default **half‑real** mode the reference recording is split: the
first half of its first codebook is used as an acoustic prompt, the AR
model generates the second half, and the two are concatenated to form the
full first codebook.  The NAR then fills in codebooks 1..15.

Steps:
  1. Phonemes: concatenate ``[ref_transcript, target_text]``.
  2. Acoustic prompt: keep the **first half** of the reference's first
     codebook (the second half is replaced by AR generation).
  3. AR stage: prefill ``[phonemes, <SEP>, ref_cb0[:Tref//2]]``, autoregress
     new tokens, then concatenate ``[ref_cb0[:Tref//2], gen0]``.
  4. NAR stage: given the full first codebook, iteratively predict codebooks
     1..15 to fill the ``(16, L)`` grid.
  5. Decode: run the grid through Mimi to obtain the waveform.

Pass ``--prompt-frames 0`` to use the full reference as prompt (gen‑only
mode, original behaviour).

``--text`` and ``--ref-text`` are **phoneme strings** in the same IPA format
as ``phonemes.csv``.  ``--ref-text`` may be omitted with ``--ref`` as a
dataset ``.npz`` (transcript is looked up from a sibling ``phonemes.csv``).

Usage::

    python scripts/infer_tts.py \\
        --ar-model  models/ar_decoder/best.pt \\
        --nar-model models/nar_decoder/best.pt \\
        --ref       data/norbi/audio/clone_0000.wav \\
        --ref-text  "wiː wɪl sˈɛnd ..." \\
        --text      "hɐlˈoʊ wˈɜːld." \\
        --output-dir tts_out
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import soundfile as sf  # noqa: E402

from echo.ar_decoder import ARDecoder, TokenType as ARTokenType  # noqa: E402
from echo.nar_decoder import NARDecoder  # noqa: E402
from echo.config import (  # noqa: E402
    AR_DROPOUT,
    CODEC_VOCAB_SIZE,
    EOS_TOKEN_ID,
    MAX_AUDIO_LENGTH,
    MAX_TEXT_LENGTH,
    NAR_DROPOUT,
    NUM_CODEBOOKS,
)
from __style__ import (  # noqa: E402
    Colors,
    print_header,
    print_section,
    print_info,
    print_separator,
    print_success,
    print_error,
)

_TARGET_SR = 24000
_AUDIO_EXTS = {".wav", ".flac", ".mp3", ".ogg", ".m4a", ".aiff", ".aif"}


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------


class PhonemeTokenizer:
    def __init__(self, stoi: dict[str, int]) -> None:
        self.stoi = stoi

    @classmethod
    def from_file(cls, path: Path | str) -> "PhonemeTokenizer":
        with open(path) as f:
            vocab = json.load(f)
        return cls(vocab["stoi"])

    def encode(self, text: str) -> list[int]:
        ids: list[int] = []
        for ch in text:
            idx = self.stoi.get(ch)
            if idx is None:
                raise KeyError(f"Unknown phoneme character {ch!r} (U+{ord(ch):04X})")
            if idx in (0, 1, 2):
                continue
            ids.append(idx)
        return ids


# ---------------------------------------------------------------------------
# Model loading (architecture inferred from the checkpoint)
# ---------------------------------------------------------------------------


def _infer_arch(state: dict) -> dict:
    d_model = state["text_embedding.weight"].shape[1]
    d_ff = state["blocks.0.ff.w1.weight"].shape[0]
    n_layers = 1 + max(int(k.split(".")[1]) for k in state if k.startswith("blocks."))
    n_heads = max(1, d_model // 64)
    return {"d_model": d_model, "d_ff": d_ff, "n_layers": n_layers, "n_heads": n_heads}


def _load_state(path: Path, device: torch.device) -> dict:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    return ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt


def load_ar_model(path: Path, device: torch.device) -> ARDecoder:
    state = _load_state(path, device)
    arch = _infer_arch(state)
    print_info("AR arch", f"d_model={arch['d_model']}, d_ff={arch['d_ff']}, "
                          f"n_layers={arch['n_layers']}, n_heads={arch['n_heads']}", Colors.OKCYAN)
    model = ARDecoder(dropout=AR_DROPOUT, **arch).to(device)
    model.load_state_dict(state)
    return model.eval()


def load_nar_model(path: Path, device: torch.device) -> NARDecoder:
    state = _load_state(path, device)
    arch = _infer_arch(state)
    print_info("NAR arch", f"d_model={arch['d_model']}, d_ff={arch['d_ff']}, "
                           f"n_layers={arch['n_layers']}, n_heads={arch['n_heads']}", Colors.OKCYAN)
    model = NARDecoder(dropout=NAR_DROPOUT, **arch).to(device)
    model.load_state_dict(state)
    return model.eval()


# ---------------------------------------------------------------------------
# Reference codes + Mimi
# ---------------------------------------------------------------------------


def load_reference_codes(ref: Path, device: torch.device) -> np.ndarray:
    """Return the reference codec grid ``(16, T)`` (int64)."""
    if ref.suffix.lower() == ".npz":
        return np.load(ref)["codes"].astype(np.int64)[:NUM_CODEBOOKS]

    if ref.suffix.lower() not in _AUDIO_EXTS:
        raise ValueError(f"Unsupported --ref type: {ref.suffix}")

    import librosa
    from transformers import MimiModel, AutoFeatureExtractor

    audio, _ = librosa.load(str(ref), sr=_TARGET_SR, mono=True)
    mimi = MimiModel.from_pretrained("kyutai/mimi").to(device).eval()
    fe = AutoFeatureExtractor.from_pretrained("kyutai/mimi")
    inputs = fe(raw_audio=[audio], sampling_rate=_TARGET_SR, return_tensors="pt")
    with torch.no_grad():
        enc = mimi.encode(inputs["input_values"].to(device))
    load_reference_codes._mimi = mimi  # type: ignore[attr-defined]
    return enc.audio_codes[0, :NUM_CODEBOOKS, :].detach().cpu().numpy().astype(np.int64)


def get_mimi(device: torch.device):
    cached = getattr(load_reference_codes, "_mimi", None)
    if cached is not None:
        return cached
    from transformers import MimiModel
    mimi = MimiModel.from_pretrained("kyutai/mimi").to(device).eval()
    load_reference_codes._mimi = mimi  # type: ignore[attr-defined]
    return mimi


def lookup_transcript(ref: Path) -> Optional[str]:
    """Look up the reference transcript from a sibling phonemes.csv."""
    csv = ref.parent.parent / "phonemes.csv"
    if not csv.is_file():
        return None
    target = ref.with_suffix(".npz").name
    with open(csv, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line or "|" not in line:
                continue
            name, phon = line.split("|", 1)
            if name == target:
                return phon
    return None


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


def sample_token(logits: torch.Tensor, temperature: float, top_k: int,
                 top_p: float, greedy: bool) -> int:
    if greedy or temperature <= 0.0:
        return int(logits.argmax().item())
    logits = logits / temperature
    if top_k and top_k > 0:
        k = min(top_k, logits.numel())
        kth = torch.topk(logits, k).values[-1]
        logits = torch.where(logits < kth, torch.full_like(logits, float("-inf")), logits)
    if top_p and top_p < 1.0:
        sl, si = torch.sort(logits, descending=True)
        probs = F.softmax(sl, dim=-1)
        cdf = torch.cumsum(probs, dim=-1)
        keep = cdf - probs <= top_p
        keep[0] = True
        sl = torch.where(keep, sl, torch.full_like(sl, float("-inf")))
        logits = torch.full_like(logits, float("-inf")).scatter(0, si, sl)
    probs = F.softmax(logits, dim=-1)
    return int(torch.multinomial(probs, 1).item())


# ---------------------------------------------------------------------------
# AR stage: generate first codebook given phonemes + acoustic prompt
# ---------------------------------------------------------------------------


@torch.no_grad()
def ar_generate(model: ARDecoder, text_ids: list[int], prompt_codes0: np.ndarray,
                device: torch.device, max_steps: int, temperature: float,
                top_k: int, top_p: float, greedy: bool,
                min_steps: int = 0) -> tuple[list[int], bool]:
    """Autoregress new first-codebook tokens after an acoustic prompt.

    Returns ``(generated_tokens, hit_eos)`` -- the reference prompt and the EOS
    token are NOT included in ``generated_tokens``.  ``min_steps`` forbids EOS
    until at least that many tokens have been produced (useful when a plain
    text->audio AR treats a complete acoustic prompt as "already finished" and
    would otherwise emit EOS immediately).
    """
    Tt = len(text_ids)
    Tp = len(prompt_codes0)

    # --- prefill: [text, <SEP>, ref_codebook0] ------------------------------
    ids = torch.tensor([text_ids + [0] + prompt_codes0.tolist()],
                       dtype=torch.long, device=device)
    types = torch.tensor([[ARTokenType.TEXT] * Tt + [ARTokenType.SEP]
                          + [ARTokenType.AUDIO] * Tp], dtype=torch.long, device=device)
    positions = torch.tensor([list(range(Tt)) + [0] + list(range(1, Tp + 1))],
                             dtype=torch.long, device=device)
    logits, kv = model.forward_step(ids, types, positions, kv_cache=None)
    next_logits = logits[0, -1, :]  # last prompt token predicts first new token

    gen: list[int] = []
    hit_eos = False
    for _ in range(max_steps):
        step_logits = next_logits
        if len(gen) < min_steps:
            step_logits = next_logits.clone()
            step_logits[EOS_TOKEN_ID] = float("-inf")  # forbid early EOS
        tok = sample_token(step_logits, temperature, top_k, top_p, greedy)
        if tok == EOS_TOKEN_ID:
            hit_eos = True
            break
        gen.append(tok)
        pos = min(Tp + len(gen), MAX_AUDIO_LENGTH)  # ref occupies audio pos 1..Tp
        tok_t = torch.tensor([[tok]], dtype=torch.long, device=device)
        typ_t = torch.tensor([[ARTokenType.AUDIO]], dtype=torch.long, device=device)
        pos_t = torch.tensor([[pos]], dtype=torch.long, device=device)
        logits, kv = model.forward_step(tok_t, typ_t, pos_t, kv_cache=kv)
        next_logits = logits[0, -1, :]

    return gen, hit_eos


# ---------------------------------------------------------------------------
# NAR stage: fill in codebooks 1..15
# ---------------------------------------------------------------------------


@torch.no_grad()
def nar_generate(model: NARDecoder, text_ids: list[int], codes0: np.ndarray,
                 device: torch.device) -> np.ndarray:
    """Iteratively predict codebooks 1..15 from the generated first codebook.

    Returns the full ``(16, L)`` grid (int64).
    """
    Tt = len(text_ids)
    L = len(codes0)
    text_t = torch.tensor([text_ids], dtype=torch.long, device=device)
    # stack: (1, n_layers_so_far, L); starts with the AR-generated layer 0.
    stack = torch.tensor(codes0, dtype=torch.long, device=device).view(1, 1, L)

    for i in range(1, NUM_CODEBOOKS):
        # No key-padding mask needed (batch size 1, no padding) -> fully
        # bidirectional attention over [text, <SEP>, audio].
        logits = model(text_t, stack, layer_idx=i)          # (1, Tt+1+L, V)
        audio_logits = logits[0, Tt + 1: Tt + 1 + L, :]     # (L, V)
        layer_i = audio_logits.argmax(dim=-1)               # (L,) greedy
        stack = torch.cat([stack, layer_i.view(1, 1, L)], dim=1)

    return stack[0].detach().cpu().numpy().astype(np.int64)  # (16, L)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Full AR + NAR TTS inference (voice cloning).")
    p.add_argument("--ar-model", type=str, required=True)
    p.add_argument("--nar-model", type=str, required=True)
    p.add_argument("--ref", type=str, required=True,
                   help="Reference audio file OR precomputed .npz codec grid.")
    p.add_argument("--ref-text", type=str, default=None,
                   help="Reference transcript phonemes. If omitted and --ref is a "
                        "dataset .npz, looked up from a sibling phonemes.csv.")
    p.add_argument("--text", type=str, required=True,
                   help="Target phoneme string to synthesize.")
    p.add_argument("--nar-text", choices=["target", "full"], default="target",
                   help="Phonemes fed to the NAR: 'target' (aligned with the generated "
                        "audio, default) or 'full' (reference transcript + target).")
    p.add_argument("--phoneme-vocab", type=str, default="checkpoints/phoneme_vocab.json")
    p.add_argument("--output-dir", type=str, default="tts_out")
    p.add_argument("--max-steps", type=int, default=256)
    p.add_argument("--min-steps", type=int, default=0,
                   help="Forbid <EOS> until at least N first-codebook tokens are "
                        "generated (avoids immediate EOS when the AR treats the "
                        "acoustic prompt as a finished utterance).")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-k", type=int, default=0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--greedy", action="store_true", help="Greedy AR decoding.")
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--device", type=str, default="auto",
                   choices=["auto", "cuda", "mps", "cpu"])
    return p.parse_args()


def select_device(prefer: str) -> torch.device:
    if prefer == "cpu":
        return torch.device("cpu")
    if prefer == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if prefer == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> None:
    args = parse_args()
    print_header("Echo TTS - AR + NAR Inference")
    print_separator()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = select_device("auto" if args.device == "auto" else args.device)

    print_section("Setup")
    print_info("Device", str(device), Colors.OKCYAN)
    print_info("Max AR steps", str(args.max_steps))
    print_info("AR decoding",
               "greedy" if args.greedy else
               f"sample (T={args.temperature}, top_k={args.top_k}, top_p={args.top_p})")

    ref = Path(args.ref)
    if not ref.is_file():
        print_error(f"Reference not found: {ref}")
        sys.exit(1)

    # ---- phonemes ----------------------------------------------------------
    ref_text = args.ref_text
    if ref_text is None:
        ref_text = lookup_transcript(ref)
        if ref_text is None:
            print_error("No --ref-text and could not look it up from phonemes.csv.")
            sys.exit(1)
    tokenizer = PhonemeTokenizer.from_file(args.phoneme_vocab)
    try:
        ref_ids = tokenizer.encode(ref_text)
        target_ids = tokenizer.encode(args.text)
    except KeyError as e:
        print_error(str(e))
        sys.exit(1)
    if len(target_ids) == 0:
        print_error("Empty target phoneme sequence.")
        sys.exit(1)

    ar_text_ids = ref_ids + target_ids                       # [transcript, text]
    nar_text_ids = target_ids if args.nar_text == "target" else ar_text_ids
    if len(ar_text_ids) > MAX_TEXT_LENGTH:
        print_error(f"Combined phonemes too long ({len(ar_text_ids)} > {MAX_TEXT_LENGTH}).")
        sys.exit(1)
    print_section("Phonemes")
    print_info("Reference transcript tokens", str(len(ref_ids)))
    print_info("Target text tokens", str(len(target_ids)))
    print_info("AR conditioning tokens ([transcript, text])", str(len(ar_text_ids)))

    # ---- models + reference -----------------------------------------------
    print_section("Loading")
    ar_model = load_ar_model(Path(args.ar_model), device)
    nar_model = load_nar_model(Path(args.nar_model), device)
    codes = load_reference_codes(ref, device)                # (16, T_ref)
    prompt_codes0 = codes[0]                                 # reference first codebook
    print_info("Reference frames", str(codes.shape[1]))
    if prompt_codes0.min() < 0 or prompt_codes0.max() >= CODEC_VOCAB_SIZE:
        print_error("Reference first-codebook token out of range.")
        sys.exit(1)

    # ---- AR stage ----------------------------------------------------------
    print_section("AR stage (first codebook)")
    gen0, hit_eos = ar_generate(
        ar_model, ar_text_ids, prompt_codes0, device, args.max_steps,
        args.temperature, args.top_k, args.top_p, args.greedy,
        min_steps=args.min_steps,
    )
    L = len(gen0)
    print_info("Generated frames", str(L))
    print_info("Stopped on <EOS>", str(hit_eos))
    if not hit_eos:
        print_info("Note", f"hit max steps ({args.max_steps}) without EOS", Colors.WARNING)
    if L == 0:
        print_error("AR generated zero frames (immediate EOS). Nothing to synthesize.")
        sys.exit(1)
    gen0_arr = np.asarray(gen0, dtype=np.int64)

    # ---- NAR stage ---------------------------------------------------------
    print_section("NAR stage (codebooks 2..16)")
    print_info("NAR conditioning", f"{args.nar_text} ({len(nar_text_ids)} tokens)")
    grid = nar_generate(nar_model, nar_text_ids, gen0_arr, device)   # (16, L)
    print_info("Full grid", f"{grid.shape[0]} x {grid.shape[1]}")

    # ---- Mimi decode (generated part only) ---------------------------------
    print_section("Mimi decode")
    mimi = get_mimi(device)
    with torch.no_grad():
        out = mimi.decode(torch.tensor(grid[None], dtype=torch.long, device=device))
    wav = out.audio_values[0, 0].detach().cpu().numpy().astype(np.float32)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{ref.stem}__tts"
    wav_path = out_dir / f"{stem}.wav"
    codes_path = out_dir / f"{stem}.codes.npz"
    sf.write(wav_path, wav, _TARGET_SR)
    np.savez_compressed(codes_path, codes=grid)

    dur = len(wav) / _TARGET_SR
    print_separator("═", 60)
    print_info("Output audio", str(wav_path), Colors.OKGREEN)
    print_info("Duration", f"{dur:.2f}s")
    print_info("Generated codes", str(codes_path))
    print_success("Done.")
    print_separator("═", 60)


if __name__ == "__main__":
    main()
