#!/usr/bin/env bash
#
# Fourteen greedy generations from a multispeaker EchoAR checkpoint, each on a
# different target text and (mostly) a different reference speaker.
#
# Decoding is greedy throughout, so every run is deterministic and the only
# things varying are the reference and the text. What the texts sweep: length
# (two words to a 40-word sentence), sentence type (statement, question,
# imperative, multi-sentence), and the front-end's harder cases -- digits,
# abbreviations, comma-heavy lists, dense sibilants.
#
# Each setup writes TWO files so you can compare the voices directly:
#
#     NN_slug_ref.wav   the reference recording that prompted it
#     NN_slug_gen.wav   what the model produced
#
# Both are 24 kHz mono, so they line up in any editor. Note the reference is a
# clean recording while the generation is Mimi decoded from two codebooks --
# some of the gap you hear is the codec, not the model. To hear that ceiling,
# encode a reference and decode it back through the same two layers.
#
# Every --ref_text is the true transcript of its --ref_audio, from
# data/librispeech/metadata.csv. Speakers are from the *unseen* test split
# except 3032, which was in training.
#
#   ./scripts/run/ar_examples.sh            # run all of them
#   ./scripts/run/ar_examples.sh 4          # run one
#   ./scripts/run/ar_examples.sh 1 2 3      # run several
#   ./scripts/run/ar_examples.sh --list     # just print what each one is
#
# Override with env vars:
#   MODEL=... OUT=... PY=... ./scripts/run/ar_examples.sh

set -uo pipefail
cd "$(dirname "$0")/../.."

MODEL="${MODEL:-checkpoints/librispeech/echo_ar_multispeaker.pt}"
OUT="${OUT:-samples/ar}"
PY="${PY:-.venv/bin/python}"
AUDIO="data/librispeech/audio"

[ -x "$PY" ] || PY=python

# --- Reference recordings, with their true transcripts -----------------------

REF_6965="$AUDIO/6965_291718_000018_000015.wav"          # 109f  8.7s  unseen
TXT_6965="There are some houses in which I lived, before I came to live with you, about which I could tell strange stories if I chose."

REF_5393="$AUDIO/5393_19218_000013_000000.wav"           #  85f  6.8s  unseen
TXT_5393="The marriage took place, therefore, in accordance with this now superannuated fashion, at m Gillenormand's house."

REF_8401="$AUDIO/8401_280380_000069_000001.wav"          #  85f  6.8s  unseen
TXT_8401="They were the Fates, deities whose duty it was to thread the days of all mortals who appeared on earth, were it but for an instant."

REF_7720="$AUDIO/7720_105167_000049_000000.wav"          #  85f  6.8s  unseen
TXT_7720="Her lips moved but whether they voiced consent or refusal he did not know, for the words were drowned in the whir of the propeller."

REF_353="$AUDIO/353_128309_000072_000005.wav"            #  87f  7.0s  unseen
TXT_353="She was-and He was-before the slow years of the North Tower-ages ago."

REF_868="$AUDIO/868_131295_000013_000010.wav"            #  88f  7.0s  unseen
TXT_868="From our ashes springs the phoenix of celestial hope, out of the freedom comes a higher realisation of manhood."

REF_3009="$AUDIO/3009_10327_000027_000004.wav"           #  87f  7.0s  unseen
TXT_3009="Therefore it would follow that the matter of spiritual things is subject to quantity; which cannot be."

REF_3032="$AUDIO/3032_19361_000059_000001.wav"           # 118f  9.4s  SEEN in training
TXT_3032="He proceeded to see if by making the planet librate, or the plane of its orbit tilt up and down, anything could be done."

REF_6828="$AUDIO/6828_64738_000013_000002.wav"           #  55f  4.4s  unseen, at the training floor
TXT_6828="When Beauty found herself alone, she began to feel pity for the poor beast."

REF_3370="$AUDIO/3370_756_000019_000005.wav"             #  26f  2.1s  unseen, BELOW the floor
TXT_3370="So much has happened to him since that day."

REF_329="$AUDIO/329_861_000043_000001.wav"               # 250f 20.0s  unseen, longest in the corpus
TXT_329="The assailants, however, did not disarm, but after a sufficient time had elapsed to allow the garrison to lay aside their armour two strong parties attacked the principal gates of the town, while Sir Walter Manny and the Earl of Oxford moved round to the opposite side with ladders for an escalade. The plan was successful."

REF_3307="$AUDIO/3307_145145_000024_000003.wav"          # 109f  8.7s  unseen
TXT_3307="In my urchin days, days free from prejudices in regard to what one ate, I, like many others, was able to appreciate that dainty."

REF_203="$AUDIO/203_132073_000032_000004.wav"            # 109f  8.7s  unseen
TXT_203="They were asking his name and bestowing upon him wellbred thanks for his heroic deed, and the Scotch cap was especially babbling and insistent."

# Setup 1 synthesizes the line speaker 6965 actually recorded next, so a real
# waveform exists to check both timbre and duration against.
GROUND_TRUTH_TEXT="And now, my dear Helen, I hope I have prepared you to see me looking perfectly hideous."
GROUND_TRUTH_WAV="$AUDIO/6965_291718_000019_000003.wav"  # 80 frames, 6.4s

# The sentence setups 12, 13 and 14 share, so the reference is the only variable.
SHARED_TEXT="It was the best of times, it was the worst of times."

DESC_1="Reconstruction, 15 words. Ground truth exists -- expect ~6.4s and a voice you can match exactly."
DESC_2="Two words. Does it stop, or ramble?"
DESC_3="Short statement, 6 words."
DESC_4="Question, 10 words. Interrogative contour."
DESC_5="Two short imperatives. Sentence boundary plus emphasis."
DESC_6="Comma-heavy list, 13 words. Five phrase breaks in one breath."
DESC_7="Digits and a year, 13 words. Exercises the front end's number expansion."
DESC_8="Abbreviations (Dr., St., Mrs.), 13 words. Seen speaker 3032."
DESC_9="Dense sibilants, 6 words, on a 4.4s reference at the training floor."
DESC_10="Two full sentences, on a 2.1s reference BELOW the floor. Should warn."
DESC_11="Long multi-clause narrative, 40 words, on the 20s reference. Premature-EOS stress test."
DESC_12="Shared sentence, voice A (speaker 3307)."
DESC_13="Shared sentence, voice B (speaker 203). 12 vs 13 isolates the reference's effect."
DESC_14="Shared sentence, NO reference. Out of distribution -- the control. No _ref.wav for this one."

# gen <index> <slug> <ref_audio> <ref_text> <target_text>
gen() {
  local i="$1" slug="$2" ref_audio="$3" ref_text="$4" target="$5"
  local stem
  stem="$(printf '%02d_%s' "$i" "$slug")"

  if [ -n "$ref_audio" ]; then
    cp "$ref_audio" "$OUT/${stem}_ref.wav"
    $PY scripts/run/ar.py \
      --model "$MODEL" \
      --ref_audio "$ref_audio" \
      --ref_text "$ref_text" \
      --text "$target" \
      --output "$OUT/${stem}_gen.wav"
  else
    $PY scripts/run/ar.py \
      --model "$MODEL" \
      --text "$target" \
      --output "$OUT/${stem}_gen.wav"
  fi
}

run_1()  { gen 1  reconstruction "$REF_6965" "$TXT_6965" "$GROUND_TRUTH_TEXT"
           cp "$GROUND_TRUTH_WAV" "$OUT/01_reconstruction_truth.wav"
           echo "    also wrote 01_reconstruction_truth.wav - the real recording of this line"; }
run_2()  { gen 2  two_words     "$REF_5393" "$TXT_5393" "Good morning."; }
run_3()  { gen 3  statement     "$REF_8401" "$TXT_8401" "The train arrives at four o'clock."; }
run_4()  { gen 4  question      "$REF_7720" "$TXT_7720" "Where did you put the letter I gave you yesterday?"; }
run_5()  { gen 5  imperative    "$REF_353"  "$TXT_353"  "Stop. Do not touch that."; }
run_6()  { gen 6  comma_list    "$REF_868"  "$TXT_868"  "She packed bread, cheese, two apples, a knife, and a bottle of water."; }
run_7()  { gen 7  numbers       "$REF_3009" "$TXT_3009" "In 1847 the population was 3,200, and by 1861 it had nearly doubled."; }
run_8()  { gen 8  abbreviations "$REF_3032" "$TXT_3032" "Dr. Adams left for St. Louis on Tuesday, and Mrs. Hale went with him."; }
run_9()  { gen 9  sibilants     "$REF_6828" "$TXT_6828" "She sells seashells by the seashore."; }
run_10() { gen 10 two_sentences "$REF_3370" "$TXT_3370" "He opened the door slowly. Nobody was there."; }
run_11() { gen 11 long_narrative "$REF_329" "$TXT_329"  "In the middle of the afternoon, when the light had begun to slant across the floorboards, she put down the letter she had been reading and walked to the window, wondering whether the answer she wanted was one anybody could give her."; }
run_12() { gen 12 voice_3307    "$REF_3307" "$TXT_3307" "$SHARED_TEXT"; }
run_13() { gen 13 voice_203     "$REF_203"  "$TXT_203"  "$SHARED_TEXT"; }
run_14() { gen 14 no_reference  ""          ""          "$SHARED_TEXT"; }

# -----------------------------------------------------------------------------

N=14

list() {
  echo "Setups (scripts/run/ar_examples.sh <n>), all greedy:"
  for i in $(seq 1 $N); do
    desc="DESC_$i"
    printf "  %2d  %s\n" "$i" "${!desc}"
  done
}

if [ "${1:-}" = "--list" ] || [ "${1:-}" = "-l" ]; then
  list
  exit 0
fi

if [ ! -f "$MODEL" ]; then
  echo "checkpoint not found: $MODEL" >&2
  echo "set it with MODEL=path/to/checkpoint.pt $0" >&2
  exit 1
fi

mkdir -p "$OUT"
targets=("$@")
[ ${#targets[@]} -eq 0 ] && targets=($(seq 1 $N))

failed=()
for i in "${targets[@]}"; do
  desc="DESC_$i"
  if [ -z "${!desc:-}" ]; then
    echo "no setup $i (1-$N)" >&2
    continue
  fi
  echo
  echo "═══ $i/$N  ${!desc}"
  if ! "run_$i"; then
    failed+=("$i")
    echo "    setup $i FAILED" >&2
  fi
done

echo
if [ ${#failed[@]} -eq 0 ]; then
  echo "done - *_ref.wav / *_gen.wav pairs in $OUT/"
else
  echo "done with failures in setups: ${failed[*]} - other output in $OUT/" >&2
  exit 1
fi
