from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch
from transformers import MimiModel, AutoFeatureExtractor

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from scripts.__style__ import Colors, print_header, print_section, print_info, print_separator, print_success
from echo.model import Echo
from echo import config
from echo.tokenizer import Tokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Echo TTS inference")
    parser.add_argument("--text", type=str, required=True, help="Target phoneme string to synthesize")
    parser.add_argument("--transcript", type=str, default="", help="Phoneme transcript of the reference audio")
    parser.add_argument(
        "--audio", type=str, default=None,
        help="Path to reference audio file (requires Mimi to encode it)",
    )
    parser.add_argument(
        "--codec", type=str, default=None,
        help="Path to pre-computed reference codec (.npz with 'codes' array of shape (L, T))",
    )
    parser.add_argument("--model", type=str, default=None, help="Path to Echo checkpoint (random init if omitted)")
    parser.add_argument("--output", type=str, default="output.wav", help="Output audio file")
    parser.add_argument("--max-steps", type=int, default=config.max_audio_length, help="Max generation steps")
    parser.add_argument("--min-steps", type=int, default=0, help="Min generation steps before EOS allowed")
    parser.add_argument("--layers", type=int, default=16, help="Mimi codec layers")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature (0 = greedy)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print_header("Echo TTS")
    print_info("Device", str(device))
    print()

    if args.audio is None and args.codec is None:
        parser.error("one of --audio or --codec is required")
    if args.audio is not None and args.codec is not None:
        parser.error("--audio and --codec are mutually exclusive")

    # --- Mimi codec (only needed if decoding audio or encoding on-the-fly) ----
    print_section("Loading Mimi codec")
    t0 = time.perf_counter()
    mimi = MimiModel.from_pretrained("kyutai/mimi").to(device).eval()
    feature_extractor = AutoFeatureExtractor.from_pretrained("kyutai/mimi")
    sr = feature_extractor.sampling_rate
    print_info("Sample rate", f"{sr} Hz")
    print_info("Load time", f"{time.perf_counter() - t0:.1f}s")

    # --- Reference codec -------------------------------------------------------
    print_section("Loading reference codec")
    if args.codec:
        codec_path = Path(args.codec)
        if not codec_path.is_file():
            print(f"Codec file not found: {codec_path}", file=sys.stderr)
            sys.exit(1)
        codes = np.load(codec_path)["codes"]                    # (L, T)
        ref_codes = torch.from_numpy(codes[:args.layers].T).long()  # (T, L)
        ref_codes = ref_codes.unsqueeze(0)                       # (1, T, L)
        print_info("Source", str(codec_path))
    else:
        audio, _ = librosa.load(args.audio, sr=sr, mono=True)
        print_info("Duration", f"{len(audio) / sr:.1f}s")
        inputs = feature_extractor(raw_audio=audio, sampling_rate=sr, return_tensors="pt")
        with torch.no_grad():
            enc = mimi.encode(inputs["input_values"].to(device))
        ref_codes = enc.audio_codes[:, : args.layers, :]         # (1, L, T)
        ref_codes = ref_codes.permute(0, 2, 1)                   # (1, T, L)
        print_info("Source", args.audio)
    print_info("Codec shape", str(list(ref_codes.shape)))

    # --- Echo model -----------------------------------------------------------
    print_section("Loading Echo model")
    model = Echo()
    if args.model:
        ckpt = torch.load(args.model, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])
        print_info("Checkpoint", args.model)
    else:
        print_info("Checkpoint", "(random init — no checkpoint provided)")
    model = model.to(device).eval()

    # --- Text → tokens --------------------------------------------------------
    tokenizer = Tokenizer(_REPO_ROOT / "models" / "phoneme_vocab.json")
    ref_text_ids = tokenizer.tokenize(args.transcript)
    text_ids = tokenizer.tokenize(args.text)
    ref_text_tensor = torch.tensor([ref_text_ids], dtype=torch.long, device=device)
    text_tensor = torch.tensor([text_ids], dtype=torch.long, device=device)
    print_info("Ref phonemes", (args.transcript[:80] + "...") if len(args.transcript) > 80 else args.transcript)
    print_info("Ref text tokens", str(len(ref_text_ids)))
    print_info("Target phonemes", (args.text[:80] + "...") if len(args.text) > 80 else args.text)
    print_info("Target text tokens", str(len(text_ids)))

    # --- Generate -------------------------------------------------------------
    print_section("Generating")
    t_gen = time.perf_counter()
    with torch.no_grad():
        frames, lengths = model.generate(
            ref_text_tensor,
            ref_codes.to(device),
            text_tensor,
            max_steps=args.max_steps,
            min_steps=args.min_steps,
            temperature=args.temperature,
        )
    gen_time = time.perf_counter() - t_gen
    print_info("Frames generated", str(lengths.item()))
    print_info("Time", f"{gen_time:.1f}s")
    print_info("Real-time factor", f"{gen_time / (lengths.item() * 0.0125):.2f}×")

    # --- Decode → audio -------------------------------------------------------
    print_section("Decoding")
    t_dec = time.perf_counter()
    codes_for_mimi = frames[0].T.unsqueeze(0)                  # (1, L, T) for Mimi
    with torch.no_grad():
        audio_out = mimi.decode(codes_for_mimi.to(device))[0]
    dec_time = time.perf_counter() - t_dec
    print_info("Time", f"{dec_time:.1f}s")

    # --- Save -----------------------------------------------------------------
    print_section("Saving")
    waveform = audio_out.squeeze().detach().cpu().numpy()
    sf.write(args.output, waveform, sr)
    print_info("File", args.output, Colors.OKCYAN)

    total = time.perf_counter() - t0
    print_separator("═", 60)
    print_info("Total time", f"{total:.1f}s", Colors.OKGREEN)
    print_success("Done.")


if __name__ == "__main__":
    main()