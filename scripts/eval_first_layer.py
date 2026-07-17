#!/usr/bin/env python3
"""Evaluate the AR decoder by resynthesising a reference sample.

The idea is to isolate the quality of the **first Mimi codebook** produced by
the AR model.  Given a reference audio sample (or its precomputed codec grid)
and its phoneme transcription:

1. Encode the reference to a ``(16, T)`` Mimi codec grid (or load it if a
   ``.npz`` is supplied).
2. **Generate the entire first codebook** autoregressively with the AR model,
   conditioned on the phonemes.
3. **Copy codebooks 1..15 verbatim from the reference** and stack them under
   the generated first codebook.
4. Decode the hybrid grid back to a waveform with Mimi.

Because only the first codebook is model-generated and the acoustic detail
layers are the real ones, the resulting audio directly reflects how good the
model's first-codebook prediction is: if it captures the right content and
prosody, the resynthesis will be intelligible and natural.

For reference, the original grid is also decoded (all 16 real codebooks) so
you can A/B ``hybrid.wav`` against ``reference.wav``.

``--text`` is the **phoneme string** in the same IPA format as
``data/<...>/phonemes.csv`` (that is what the model was trained on).  If it is
omitted and ``--ref`` points at a dataset ``.npz``, the phonemes are looked up
from a sibling ``phonemes.csv``.

Usage:
    python scripts/eval_first_layer.py \
        --model models/ar_decoder/best.pt \
        --ref data/norbi/audio/clone_0000.wav \
        --text "wiː wɪl sˈɛnd juː ɐ ɹᵻnjˈuːəl nˈOɾɪs ɪnðə pˈOst ɪn ɑːktˈObɚ." \
        --output-dir eval_out

    # length taken from the model's own <EOS> instead of the reference
    python scripts/eval_first_layer.py --model ... --ref clip.wav \
        --text "..." --length-mode eos
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

# Make the ``echo`` package and ``__style__`` importable when running this
# script directly, regardless of the current working directory.
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

from echo.ar_decoder import ARDecoder, TokenType  # noqa: E402
from echo.config import (  # noqa: E402
    AR_D_FF,
    AR_D_MODEL,
    AR_DROPOUT,
    AR_N_HEADS,
    AR_N_LAYERS,
    CODEC_VOCAB_SIZE,
    EOS_TOKEN_ID,
    MAX_AUDIO_LENGTH,
    MAX_TEXT_LENGTH,
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
# Tokenizer (identical mapping to training)
# ---------------------------------------------------------------------------


class PhonemeTokenizer:
    """Map a phoneme string to token ids, one per Unicode code point."""

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
            if idx in (0, 1, 2):  # skip <pad>/<bos>/<eos>
                continue
            ids.append(idx)
        return ids


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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


def _infer_arch(state: dict) -> dict:
    """Infer (d_model, d_ff, n_layers, n_heads) from a checkpoint state dict.

    This keeps eval independent of the current ``echo.config`` values, which
    may have drifted since the checkpoint was trained.  ``n_heads`` is derived
    from the repo's ``head_dim = 64`` convention.
    """
    d_model = state["text_embedding.weight"].shape[1]
    d_ff = state["blocks.0.ff.w1.weight"].shape[0]
    n_layers = 1 + max(int(k.split(".")[1]) for k in state if k.startswith("blocks."))
    n_heads = max(1, d_model // 64)
    return {"d_model": d_model, "d_ff": d_ff, "n_layers": n_layers, "n_heads": n_heads}


def load_model(path: Path, device: torch.device) -> ARDecoder:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    state = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
    arch = _infer_arch(state)
    print_info("Model arch (from checkpoint)",
               f"d_model={arch['d_model']}, d_ff={arch['d_ff']}, "
               f"n_layers={arch['n_layers']}, n_heads={arch['n_heads']}", Colors.OKCYAN)
    model = ARDecoder(dropout=AR_DROPOUT, **arch).to(device)
    model.load_state_dict(state)
    model.eval()
    if isinstance(ckpt, dict):
        step = ckpt.get("step")
        val = ckpt.get("best_val_loss")
        if step is not None:
            print_info("Checkpoint step", str(step))
        if val is not None:
            print_info("Checkpoint best val loss", f"{val:.4f}", Colors.OKCYAN)
    return model


def load_reference_codes(ref: Path, layers: int,
                         device: torch.device) -> np.ndarray:
    """Return the reference codec grid ``(num_layers, T)`` (int64).

    ``ref`` may be a precomputed ``.npz`` (with a ``codes`` array) or an audio
    file, in which case it is encoded with Mimi on the fly.
    """
    if ref.suffix.lower() == ".npz":
        codes = np.load(ref)["codes"].astype(np.int64)
        return codes[:layers]

    if ref.suffix.lower() not in _AUDIO_EXTS:
        raise ValueError(f"Unsupported --ref type: {ref.suffix} "
                         f"(expected .npz or an audio file)")

    import librosa
    from transformers import MimiModel, AutoFeatureExtractor

    audio, _ = librosa.load(str(ref), sr=_TARGET_SR, mono=True)
    mimi = MimiModel.from_pretrained("kyutai/mimi").to(device).eval()
    fe = AutoFeatureExtractor.from_pretrained("kyutai/mimi")
    inputs = fe(raw_audio=[audio], sampling_rate=_TARGET_SR, return_tensors="pt")
    with torch.no_grad():
        enc = mimi.encode(inputs["input_values"].to(device))
    codes = enc.audio_codes[0, :layers, :].detach().cpu().numpy().astype(np.int64)
    # Cache the model so decode() can reuse it.
    load_reference_codes._mimi = mimi  # type: ignore[attr-defined]
    return codes


def get_mimi(device: torch.device):
    """Return a (cached) Mimi model for decoding."""
    cached = getattr(load_reference_codes, "_mimi", None)
    if cached is not None:
        return cached
    from transformers import MimiModel
    mimi = MimiModel.from_pretrained("kyutai/mimi").to(device).eval()
    load_reference_codes._mimi = mimi  # type: ignore[attr-defined]
    return mimi


def sample_token(logits: torch.Tensor, temperature: float, top_k: int,
                 top_p: float, greedy: bool) -> int:
    """Sample one token id from a ``(V,)`` logit vector."""
    if greedy or temperature <= 0.0:
        return int(logits.argmax().item())

    logits = logits / temperature

    if top_k and top_k > 0:
        k = min(top_k, logits.numel())
        kth = torch.topk(logits, k).values[-1]
        logits = torch.where(logits < kth, torch.full_like(logits, float("-inf")), logits)

    if top_p and top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        probs = F.softmax(sorted_logits, dim=-1)
        cdf = torch.cumsum(probs, dim=-1)
        # Keep tokens up to and including the one that crosses top_p.
        keep = cdf - probs <= top_p
        keep[0] = True  # always keep the top token
        sorted_logits = torch.where(keep, sorted_logits, torch.full_like(sorted_logits, float("-inf")))
        logits = torch.full_like(logits, float("-inf")).scatter(0, sorted_idx, sorted_logits)

    probs = F.softmax(logits, dim=-1)
    return int(torch.multinomial(probs, 1).item())


@torch.no_grad()
def generate_first_layer(model: ARDecoder, text_ids: list[int], device: torch.device,
                         max_len: int, forbid_eos: bool, temperature: float,
                         top_k: int, top_p: float, greedy: bool,
                         prompt_codes0: Optional[np.ndarray] = None
                         ) -> tuple[list[int], bool]:
    """Autoregressively generate first-codebook tokens.

    Returns ``(tokens, hit_eos)`` — the **full** layer (prompt + generated).
    When ``prompt_codes0`` is ``None`` or empty, generation starts from the
    ``<SEP>`` logits (scratch generation).
    When ``prompt_codes0`` is provided those frames are used as an acoustic
    prefix (prefill includes them), the model generates the remaining tokens,
    and the returned list is ``prompt_codes0 + generated``.

    ``forbid_eos`` masks the EOS logit so the model always runs ``max_len``
    total frames (reference‑length mode); otherwise it may stop early.
    """
    Tt = len(text_ids)
    Tp = len(prompt_codes0) if prompt_codes0 is not None else 0

    # --- build prefill sequence ----------------------------------------------
    ids_list = text_ids + [0]  # dummy id for SEP
    types_list = [TokenType.TEXT] * Tt + [TokenType.SEP]
    pos_list = list(range(Tt)) + [0]

    if Tp > 0:
        ids_list.extend(prompt_codes0.tolist())
        types_list.extend([TokenType.AUDIO] * Tp)
        pos_list.extend(range(1, Tp + 1))

    ids = torch.tensor([ids_list], dtype=torch.long, device=device)
    types = torch.tensor([types_list], dtype=torch.long, device=device)
    positions = torch.tensor([pos_list], dtype=torch.long, device=device)

    logits, kv = model.forward_step(ids, types, positions, kv_cache=None)
    next_logits = logits[0, -1, :]  # last prompt token predicts first new token

    gen_remaining = max(0, max_len - Tp)

    tokens: list[int] = []
    hit_eos = False
    for _ in range(gen_remaining):
        step_logits = next_logits.clone()
        if forbid_eos:
            step_logits[EOS_TOKEN_ID] = float("-inf")
        tok = sample_token(step_logits, temperature, top_k, top_p, greedy)
        if tok == EOS_TOKEN_ID:
            hit_eos = True
            break
        tokens.append(tok)
        # Position: audio position starts after the prompt.
        pos = min(Tp + len(tokens), MAX_AUDIO_LENGTH)
        tok_t = torch.tensor([[tok]], dtype=torch.long, device=device)
        typ_t = torch.tensor([[TokenType.AUDIO]], dtype=torch.long, device=device)
        pos_t = torch.tensor([[pos]], dtype=torch.long, device=device)
        logits, kv = model.forward_step(tok_t, typ_t, pos_t, kv_cache=kv)
        next_logits = logits[0, -1, :]

    if Tp > 0:
        return prompt_codes0.tolist() + tokens, hit_eos
    return tokens, hit_eos


@torch.no_grad()
def teacher_forced_metrics(model: ARDecoder, text_ids: list[int],
                           ref_layer0: np.ndarray, device: torch.device
                           ) -> tuple[float, float]:
    """Unsmoothed per-token CE and top-1 accuracy of the reference layer-0.

    This is the clean quantitative signal (same quantity the val loss tracks,
    but for this one sample), independent of sampling.
    """
    text = torch.tensor([text_ids], dtype=torch.long, device=device)
    audio = torch.from_numpy(np.asarray(ref_layer0)[None]).long().to(device)
    T = audio.shape[1]
    logits = model(text, audio)  # (1, Tt+1+T, V)
    Tt = len(text_ids)
    pred = logits[:, Tt:Tt + T + 1, :]  # SEP..a_{T-1} -> predict a_0..a_{T-1}, EOS
    targets = torch.cat([audio[0], torch.tensor([EOS_TOKEN_ID], device=device)])
    ce = F.cross_entropy(pred[0], targets, reduction="mean").item()
    # Top-1 accuracy over the audio-token positions only (exclude EOS step).
    top1 = (pred[0, :T].argmax(dim=-1) == audio[0]).float().mean().item()
    return ce, top1


def maybe_lookup_phonemes(ref: Path) -> Optional[str]:
    """If ``ref`` is a dataset .npz, try to find its phonemes in phonemes.csv."""
    # data/<set>/codecs/<name>.npz  -> data/<set>/phonemes.csv keyed by <name>.npz
    csv = ref.parent.parent / "phonemes.csv"
    if not csv.is_file():
        return None
    target = ref.name  # e.g. clone_0000.npz
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
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Resynthesise a reference sample using a "
                                            "model-generated first codebook.")
    p.add_argument("--model", type=str, required=True, help="Path to an AR decoder checkpoint.")
    p.add_argument("--ref", type=str, required=True,
                   help="Reference audio file OR precomputed .npz codec grid.")
    p.add_argument("--text", type=str, default=None,
                   help="Phoneme string (IPA, same format as phonemes.csv). If omitted "
                        "and --ref is a dataset .npz, it is looked up from phonemes.csv.")
    p.add_argument("--phoneme-vocab", type=str, default="checkpoints/phoneme_vocab.json")
    p.add_argument("--output-dir", type=str, default="eval_out")
    p.add_argument("--layers", type=int, default=16, help="Number of codec layers to use.")
    p.add_argument("--length-mode", choices=["reference", "eos"], default="reference",
                   help="'reference': generate exactly as many frames as the reference "
                        "(EOS masked, cleanest A/B). 'eos': let the model stop at its own EOS.")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-k", type=int, default=0, help="0 disables top-k.")
    p.add_argument("--top-p", type=float, default=1.0, help="1.0 disables nucleus sampling.")
    p.add_argument("--greedy", action="store_true", help="Argmax decoding (ignores sampling args).")
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--device", type=str, default="auto",
                   choices=["auto", "cuda", "mps", "cpu"])
    return p.parse_args()


def main() -> None:
    args = parse_args()
    print_header("AR Decoder - First-Layer Resynthesis")
    print_separator()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = select_device("auto" if args.device == "auto" else args.device)

    print_section("Setup")
    print_info("Device", str(device), Colors.OKCYAN)
    print_info("Length mode", args.length_mode)
    if args.greedy:
        print_info("Decoding", "greedy (argmax)")
    else:
        print_info("Decoding", f"sample (T={args.temperature}, top_k={args.top_k}, top_p={args.top_p})")

    ref = Path(args.ref)
    if not ref.is_file():
        print_error(f"Reference not found: {ref}")
        sys.exit(1)

    # ---- phonemes ----------------------------------------------------------
    text = args.text
    if text is None:
        text = maybe_lookup_phonemes(ref)
        if text is None:
            print_error("No --text supplied and could not look up phonemes from a "
                        "sibling phonemes.csv. Pass --text explicitly.")
            sys.exit(1)
        print_info("Phonemes (looked up)", text[:60] + ("..." if len(text) > 60 else ""))
    else:
        print_info("Phonemes", text[:60] + ("..." if len(text) > 60 else ""))

    tokenizer = PhonemeTokenizer.from_file(args.phoneme_vocab)
    try:
        text_ids = tokenizer.encode(text)
    except KeyError as e:
        print_error(str(e))
        sys.exit(1)
    if len(text_ids) == 0:
        print_error("Empty phoneme sequence after tokenisation.")
        sys.exit(1)
    if len(text_ids) > MAX_TEXT_LENGTH:
        print_error(f"Phoneme sequence too long ({len(text_ids)} > {MAX_TEXT_LENGTH}).")
        sys.exit(1)

    # ---- model + reference codes ------------------------------------------
    print_section("Loading")
    model = load_model(Path(args.model), device)
    codes = load_reference_codes(ref, args.layers, device)  # (L, T)
    num_layers, T_ref = codes.shape
    print_info("Reference frames (T)", str(T_ref))
    print_info("Codec layers", str(num_layers))
    if num_layers < 1:
        print_error("Reference has no codec layers.")
        sys.exit(1)
    # Sanity: first-codebook range.
    if codes[0].min() < 0 or codes[0].max() >= CODEC_VOCAB_SIZE:
        print_error("Reference first-codebook token out of range.")
        sys.exit(1)

    # ---- teacher-forced metric (sampling-independent) ----------------------
    print_section("Teacher-forced metrics (reference layer-0)")
    ce, top1 = teacher_forced_metrics(model, text_ids, codes[0], device)
    print_info("Cross-entropy (unsmoothed)", f"{ce:.4f} nats  (ppl {np.exp(ce):.2f})", Colors.OKCYAN)
    print_info("Top-1 accuracy", f"{top1 * 100:.2f}%", Colors.OKCYAN)

    # ---- generate first codebook (first half real, second half model) ------
    print_section("Generating first codebook (half-real)")
    T_half = T_ref // 2
    prompt = codes[0, :T_half]          # keep first half as acoustic prompt
    forbid_eos = args.length_mode == "reference"
    max_len = T_ref if forbid_eos else MAX_AUDIO_LENGTH
    gen, hit_eos = generate_first_layer(
        model, text_ids, device, max_len=max_len, forbid_eos=forbid_eos,
        temperature=args.temperature, top_k=args.top_k, top_p=args.top_p, greedy=args.greedy,
        prompt_codes0=prompt,
    )
    L = len(gen)
    print_info("Real prefix frames", str(T_half))
    print_info("Generated frames (total)", str(L))
    if not forbid_eos:
        print_info("Stopped on <EOS>", str(hit_eos))
        if not hit_eos:
            print_info("Note", "hit max length without EOS", Colors.WARNING)
    if L <= T_half:
        print_error("Model generated no new frames (immediate EOS or zero output).")
        sys.exit(1)

    # Length to actually decode: min(generated total, reference).
    Lc = min(L, T_ref)
    gen_layer0 = np.asarray(gen[:Lc], dtype=np.int64)

    # Free-running exact-match against the reference (only on the generated half).
    gen_half = gen_layer0[T_half:Lc] if Lc > T_half else np.array([], dtype=np.int64)
    ref_half = codes[0, T_half:Lc] if Lc > T_half else np.array([], dtype=np.int64)
    match = float((gen_half == ref_half).mean()) if len(gen_half) > 0 else 0.0
    print_info("Free-running match vs reference (generated half)", f"{match * 100:.2f}%")

    # ---- build hybrid grid: generated layer-0 + real layers 1..N ----------
    hybrid = codes[:, :Lc].copy()
    hybrid[0] = gen_layer0

    # ---- decode ------------------------------------------------------------
    print_section("Decoding with Mimi")
    mimi = get_mimi(device)

    def decode(grid: np.ndarray) -> np.ndarray:
        t = torch.tensor(grid[None], dtype=torch.long, device=device)
        with torch.no_grad():
            out = mimi.decode(t)
        wav = out.audio_values[0, 0].detach().cpu().numpy().astype(np.float32)
        return wav

    hybrid_wav = decode(hybrid)
    reference_wav = decode(codes)          # all real layers -> upper-bound reconstruction
    gen_only_wav = decode(gen_layer0[None])  # generated first codebook ALONE (1 quantizer)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = ref.stem
    hybrid_path = out_dir / f"{stem}.hybrid.wav"
    ref_path = out_dir / f"{stem}.reference.wav"
    gen_only_path = out_dir / f"{stem}.gen_only.wav"
    codes_path = out_dir / f"{stem}.gen_codes.npz"
    sf.write(hybrid_path, hybrid_wav, _TARGET_SR)
    sf.write(ref_path, reference_wav, _TARGET_SR)
    sf.write(gen_only_path, gen_only_wav, _TARGET_SR)
    np.savez_compressed(codes_path, generated_layer0=gen_layer0, hybrid=hybrid)

    print_separator("═", 60)
    print_info("Hybrid (gen layer-0 + real rest)", str(hybrid_path), Colors.OKGREEN)
    print_info("Gen-only (gen layer-0, 1 codebook)", str(gen_only_path), Colors.OKGREEN)
    print_info("Reference (all real layers)", str(ref_path), Colors.OKCYAN)
    print_info("Generated codes", str(codes_path))
    print_success("Done. A/B the wavs to judge first-codebook quality.")
    print_separator("═", 60)


if __name__ == "__main__":
    main()
