#!/usr/bin/env bash
# Generate 10 speech samples using the Echo model.
# Usage: bash scripts/generate_samples.sh [--temperature 0.0] [--max-steps 200] [--min-steps 10]

set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"

TEMPERATURE="${1:-0.0}"
MAX_STEPS="${2:-200}"
MIN_STEPS="${3:-10}"
OUTDIR="outputs"
CHECKPOINT="models/checkpoint_epoch020.pt"
REF_CODEC="data/norbi/codecs/clone_0000.npz"
REF_TRANSCRIPT="wiː wɪl sˈɛnd juː ɐ ɹᵻnjˈuːəl nˈOɾɪs ɪnðə pˈOst ɪn ɑːktˈObɚ."

mkdir -p "$OUTDIR"

echo "Generating 10 samples with temperature=$TEMPERATURE"
echo "Reference: clone_0000 ($REF_TRANSCRIPT)"
echo "Output dir: $OUTDIR"
echo ""

samples=(
  "wˌʌt ɑːɹ ðə kˈiː ˈɛlɪmənts əvə tɹˈæʤɪk plˈA?"
  "ʃiː ɡˈAv mˌiː ɐ θˈɔːtfəl ɡˈɪft. wˌʌt ɐ kˈInd ʤˈɛsʧɚ!"
  "ɪf wiː fˈWnd ˈAliən lˈIf, hˌW wʊd səsˈIəɾi ɹɪˈækt?"
  "tɹˈævəl wˈIdli."
  "ɪz hjˈuːmən nˈAʧɚ fˈɪkst ɔːɹ mˈæləbəl?"
  "sˈɪmpli mˈɑːɹvələs!"
  "ðə kˈɪʧən ˈɔːlsO hɐz ɐ pˈæntɹi."
  "ɐ ɡˈʊd bˈʊk tɹˈænspɔːɹts juː ˈɪnstəntli."
  "ɪmˈæʤɪn ɐ wˈɜːld wɪð pˈɜːfɛkt mˈɛmɚɹi."
  "hˈOld ɪt ˈWt ðɛɹ! ɛɡzˈɛkjuTˌɔɹʧ ɪz kˈʌmɪŋ. wˌi nˈid tə stˈA stɹˈɔŋ ænd fˈOkəst."
  "bˈɑɹTɛk, wˌʌt ðə fˈʌk ɑɹ ju dˈuɪŋ ðɛɹ wɪð ðɪs ʤˈɛmə? wˈI ɪz ɪt skɹˈOlɪŋ tˈɪktˈɑks ˈOvəɹ ænd ˈOvəɹ əɡˈɛn? jˌu mˈʌst stˈɑp, ˈʌðəɹwˌIz mˈɪstəɹ ɛɡzˈɛkjuTˌɔɹʧ wɪl kˈʌm tʊ ju."
  "jˌu mˌIt θˈɪŋk, ɛɡzˈɛkjuTˌɔɹʧ ɪz ˈɔl wi nˈid. bˈʌt ðˌæts nˌɑt tɹˈu. wˌi ɑɹ mˈʌʧ bˈɪɡəɹ ðən ðɪs mˈʌðəɹfˌʌkəɹ nˈAmd ɛɡzˈɛkjuTˌɔɹʧ. ˌænd wi wɪl pɹˈuv ɪt!"
)

labels=(
  "elements_of_tragedy"
  "thoughtful_gift"
  "alien_life"
  "travel_widely"
  "human_nature"
  "marvelous"
  "kitchen_pantry"
  "good_book"
  "perfect_memory"
  "hold_it_right"
  "bartek"
  "executorch"
)

for i in "${!samples[@]}"; do
  text="${samples[$i]}"
  label="${labels[$i]}"
  outfile="$OUTDIR/${label}.wav"

  echo "[$((i+1))/10] Generating: $label"
  .venv/bin/python scripts/run.py \
    --text "$text" \
    --transcript "$REF_TRANSCRIPT" \
    --codec "$REF_CODEC" \
    --model "$CHECKPOINT" \
    --output "$outfile" \
    --max-steps "$MAX_STEPS" \
    --min-steps "$MIN_STEPS" \
    --temperature "$TEMPERATURE" \
    2>&1 | tail -5
  echo ""
done

echo "Done. Generated files:"
ls -1 "$OUTDIR"/*.wav