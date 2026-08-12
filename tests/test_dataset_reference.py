"""Same-speaker reference sampling in EchoDataset, and its collation."""

import numpy as np
import pytest
import torch

from echo import config
from echo.ar_model import EchoAR
from echo.tokenizer import Tokenizer
from echo.training.collate import collate_fn
from echo.training.dataset import EchoDataset

LAYERS = 2
VOCAB = "models/phoneme_vocab.json"


@pytest.fixture
def corpus(tmp_path):
    """Three speakers x four utterances, with codec files of known length."""

    codec_dir = tmp_path / "codecs"
    codec_dir.mkdir()
    rng = np.random.default_rng(0)

    lines, lengths = [], {}
    for speaker in ("A", "B", "C"):
        for i, frames in enumerate((80, 12, 55, 90)):        # one below any floor
            name = f"{speaker}_{i}.npz"
            np.savez_compressed(
                codec_dir / name,
                codes=rng.integers(0, 2048, (16, frames), dtype=np.int32),
            )
            lines.append(f"{name}|hɛlˈO wˈɜrld {i}|{speaker}")
            lengths[name] = frames

    manifest = tmp_path / "phonemes.csv"
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return {"manifest": manifest, "codec_dir": codec_dir, "lengths": lengths}


def _dataset(corpus, **kwargs):
    return EchoDataset(
        corpus["manifest"], None, None, Tokenizer(VOCAB),
        codec_dir=corpus["codec_dir"], codec_layers=LAYERS,
        load_latent=False, load_distil=False, **kwargs,
    )


# --------------------------------------------------------------------------
# Manifest parsing
# --------------------------------------------------------------------------

def test_speaker_column_is_not_folded_into_the_transcript(corpus):
    """`|` is phoneme id 16, so a bad split appends a token to every sample."""

    tok = Tokenizer(VOCAB)
    assert tok.tokenize("|") == [16]                             # the trap

    line = corpus["manifest"].read_text(encoding="utf-8").splitlines()[0]
    assert 16 in tok.tokenize(line.split("|", 1)[1])             # the old parse

    _, phonemes, _ = EchoDataset._load_index(corpus["manifest"])[0]
    assert 16 not in tok.tokenize(phonemes)                      # the new one


def test_load_index_reads_three_columns(corpus):
    samples = EchoDataset._load_index(corpus["manifest"])
    name, phonemes, speaker = samples[0]
    assert name == "A_0.npz" and speaker == "A"
    assert "|" not in phonemes and speaker not in phonemes


def test_load_index_tolerates_a_two_column_manifest(tmp_path):
    manifest = tmp_path / "two.csv"
    manifest.write_text("a.npz|hɛlˈO\nb.npz|wˈɜrld\n", encoding="utf-8")
    assert EchoDataset._load_index(manifest) == [
        ("a.npz", "hɛlˈO", None), ("b.npz", "wˈɜrld", None),
    ]


def test_load_index_normalizes_bare_stems(tmp_path):
    """The corpus manifests carry stems; every artifact dir uses `<stem>.npz`."""

    manifest = tmp_path / "stems.csv"
    manifest.write_text("100_121669_000001_000000|hɛlˈO|100\n", encoding="utf-8")
    assert EchoDataset._load_index(manifest)[0][0] == "100_121669_000001_000000.npz"


def test_text_carries_no_trailing_separator(corpus):
    sample = _dataset(corpus)[0]
    assert 16 not in sample["text"].tolist()


# --------------------------------------------------------------------------
# The length index
# --------------------------------------------------------------------------

def test_length_cache_is_written_and_reused(corpus):
    _dataset(corpus, load_reference=True, min_ref_frames=50)
    cache = corpus["codec_dir"] / EchoDataset.CODEC_LENGTH_CACHE
    assert cache.is_file()

    cached = np.load(cache)
    got = dict(zip(cached["names"].tolist(), cached["lengths"].tolist()))
    assert got == corpus["lengths"]

    # A second build must not need the codec files at all -- everything but the
    # cache itself goes away, and construction still succeeds.
    hidden = [p for p in corpus["codec_dir"].glob("*.npz") if p.name != cache.name]
    for npz in hidden:
        npz.rename(npz.with_suffix(".hidden"))
    try:
        _dataset(corpus, load_reference=True, min_ref_frames=50)
    finally:
        for npz in corpus["codec_dir"].glob("*.hidden"):
            npz.rename(npz.with_suffix(".npz"))


# --------------------------------------------------------------------------
# Reference sampling
# --------------------------------------------------------------------------

def test_reference_is_same_speaker_long_enough_and_never_itself(corpus):
    ds = _dataset(corpus, load_reference=True, min_ref_frames=50)
    names = [name for name, _, _ in ds._samples]

    for idx in range(len(ds)):
        speaker = ds._samples[idx][2]
        for _ in range(50):
            ref = ds._reference_index(idx, speaker)
            assert ref != idx
            assert ds._samples[ref][2] == speaker
            assert corpus["lengths"][names[ref]] >= 50


def test_max_ref_frames_bounds_the_pool(corpus):
    ds = _dataset(corpus, load_reference=True, min_ref_frames=50, max_ref_frames=80)
    names = [name for name, _, _ in ds._samples]
    for pool in ds._by_speaker.values():
        assert all(50 <= corpus["lengths"][names[i]] <= 80 for i in pool)


def test_unreachable_speaker_is_a_construction_error(corpus):
    with pytest.raises(ValueError, match="no utterance"):
        _dataset(corpus, load_reference=True, min_ref_frames=95)


def test_reference_needs_a_speaker_column(tmp_path, corpus):
    manifest = tmp_path / "two.csv"
    manifest.write_text("A_0.npz|hɛlˈO\n", encoding="utf-8")
    with pytest.raises(ValueError, match="speaker column"):
        EchoDataset(
            manifest, None, None, Tokenizer(VOCAB), codec_dir=corpus["codec_dir"],
            load_latent=False, load_distil=False, load_reference=True,
        )


def test_seed_makes_references_reproducible(corpus):
    a = _dataset(corpus, load_reference=True, reference_seed=7)
    b = _dataset(corpus, load_reference=True, reference_seed=7)
    for idx in range(len(a)):
        spk = a._samples[idx][2]
        assert a._reference_index(idx, spk) == b._reference_index(idx, spk)


def test_unseeded_references_vary(corpus):
    ds = _dataset(corpus, load_reference=True)
    draws = {ds._reference_index(0, "A") for _ in range(60)}
    assert len(draws) > 1


def test_getitem_emits_both_reference_fields(corpus):
    sample = _dataset(corpus, load_reference=True)[0]
    assert set(sample) == {"text", "codec", "ref_text", "ref_codec"}
    assert sample["ref_codec"].shape[1] == LAYERS
    assert sample["ref_codec"].shape[0] >= 50
    assert sample["ref_text"].dtype == torch.long


def test_reference_is_off_by_default(corpus):
    assert "ref_text" not in _dataset(corpus)[0]


# --------------------------------------------------------------------------
# Collation
# --------------------------------------------------------------------------

def test_collate_pads_references_with_their_own_masks(corpus):
    ds = _dataset(corpus, load_reference=True)
    batch = [ds[i] for i in range(6)]
    out = collate_fn(batch)

    for name in ("text", "codec", "ref_text", "ref_codec"):
        mask = out[f"{name}_key_padding_mask"]
        assert out[name].shape[0] == len(batch)
        assert out[name].shape[1] == max(s[name].size(0) for s in batch)
        assert mask.sum(1).tolist() == [s[name].size(0) for s in batch]

    # Padded tails carry the right fill, so the model's merge can find them.
    row = int(out["ref_codec_key_padding_mask"][0].sum())
    assert (out["ref_codec"][0, row:] == config.prosody_pad).all()
    row = int(out["ref_text_key_padding_mask"][0].sum())
    assert (out["ref_text"][0, row:] == config.text_pad).all()


def test_collate_without_references_is_unchanged(corpus):
    out = collate_fn([_dataset(corpus)[i] for i in range(4)])
    assert set(out) == {"text", "text_key_padding_mask",
                        "codec", "codec_key_padding_mask"}


# --------------------------------------------------------------------------
# Feeding a batch to the model
# --------------------------------------------------------------------------

def _rows(lengths, width):
    return torch.arange(width)[None, :] < torch.tensor(lengths)[:, None]


def test_merged_frame_counts_is_not_the_sum_of_padded_widths():
    """The two agree only when one row is longest on both sides."""

    x = torch.zeros(2, 6, LAYERS, dtype=torch.long)
    ref = torch.zeros(2, 5, LAYERS, dtype=torch.long)
    offset, frames = EchoAR.merged_frame_counts(
        x, _rows([6, 2], 6), ref, _rows([1, 5], 5),
    )
    assert offset.tolist() == [1, 5]
    assert frames == 7                                   # max(1+6, 5+2)
    assert frames != x.shape[1] + ref.shape[1]           # the naive 11


def test_merged_frame_counts_without_a_reference(corpus):
    x = torch.zeros(2, 6, LAYERS, dtype=torch.long)
    offset, frames = EchoAR.merged_frame_counts(x, _rows([6, 4], 6))
    assert offset.tolist() == [0, 0] and frames == 6


def test_a_collated_batch_feeds_the_model(corpus):
    ds = _dataset(corpus, load_reference=True, reference_seed=1)
    b = collate_fn([ds[i] for i in range(6)])

    offset, frames = EchoAR.merged_frame_counts(
        b["codec"], b["codec_key_padding_mask"],
        b["ref_codec"], b["ref_codec_key_padding_mask"],
    )
    assert offset.tolist() == b["ref_codec_key_padding_mask"].sum(1).tolist()

    model = EchoAR().eval()
    with torch.no_grad():
        logits = model(
            b["codec"], b["text"], b["codec_key_padding_mask"],
            b["text_key_padding_mask"],
            cond_tokens=torch.zeros(6, frames, LAYERS - 1, dtype=torch.long),
            ref_codec=b["ref_codec"],
            ref_codec_padding_mask=b["ref_codec_key_padding_mask"],
            ref_text=b["ref_text"],
            ref_text_padding_mask=b["ref_text_key_padding_mask"],
        )
    assert logits.shape == (6, frames, LAYERS, config.prosody_vocab_size)
    assert torch.isfinite(logits).all()


def test_wrong_cond_tokens_length_is_reported_clearly(corpus):
    ds = _dataset(corpus, load_reference=True, reference_seed=1)
    b = collate_fn([ds[i] for i in range(6)])
    naive = b["codec"].shape[1] + b["ref_codec"].shape[1]     # the trap
    _, frames = EchoAR.merged_frame_counts(
        b["codec"], b["codec_key_padding_mask"],
        b["ref_codec"], b["ref_codec_key_padding_mask"],
    )
    assert naive != frames

    with pytest.raises(ValueError, match="merged_frame_counts"):
        EchoAR().eval()(
            b["codec"], b["text"], b["codec_key_padding_mask"],
            b["text_key_padding_mask"],
            cond_tokens=torch.zeros(6, naive, LAYERS - 1, dtype=torch.long),
            ref_codec=b["ref_codec"],
            ref_codec_padding_mask=b["ref_codec_key_padding_mask"],
            ref_text=b["ref_text"],
            ref_text_padding_mask=b["ref_text_key_padding_mask"],
        )
