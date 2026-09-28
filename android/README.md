# PlantGuard Android app

Kotlin + Jetpack Compose + CameraX + Navigation Compose + Room, fully offline,
`minSdk 24`. The app now runs against the real fine-tuned model (see "The
model contract" below) and has a redesigned three-tier result flow — read
"Why three tiers, not two" first if you're new to this codebase, since it
shapes almost everything downstream of a prediction.

## What's in here right now

- **Home screen** (the app's start destination, after the one-time
  disclaimer): the app name, a leaf mark, and four large cards — Identify a
  Plant, History, About, and How to Take a Good Photo.
- **Camera screen**: CameraX preview with a subtle framing-guide overlay
  showing roughly the square the classifier will actually see (it
  centre-crops every photo), a shutter, a gallery picker
  (`ActivityResultContracts.PickVisualMedia` — needs no storage permission
  on any Android version), and a scanning animation while inference runs
  (which also blocks the shutter — the previous overlay let a second tap
  start a second inference mid-flight).
- **Result screen — three tiers, not two.** See the section below; this
  replaced a binary "confident diagnosis or unclear photo" gate that was
  measurably misleading for 41% of real field photos.
- **Disease detail screen**: full symptoms/management/citation for one
  class, reached by tapping a candidate in the "possible matches" tier.
- **History screen**: past predictions, stored locally in Room, each row
  carrying the same tier badge the result screen uses.
- **About screen**: what the app does, its 38-class/14-crop coverage, that
  it's offline, and its measured field accuracy.
- **Photo tips screen**: concrete framing guidance — fill the frame with
  one leaf, get close, keep it sharp, avoid harsh shadow/glare, avoid
  overlapping leaves. Not filler: better framing measurably raises the
  share of photos that land in the confident tier.
- **Disclaimer**: shown once, on first launch, then lands on Home.
- **No `INTERNET` permission anywhere** — check `AndroidManifest.xml`. This
  app cannot reach the network even if something in it tried to.

## Why three tiers, not two

The old app had a binary gate: above the OOD threshold it named a disease,
below it said "Unclear photo, please retake." Measured on PlantDoc's
236-image official test split, that single message was wrong roughly 90% of
the time it fired: of the 31 images scoring below 0.50 confidence, 28 are
**sharp** (Laplacian variance ≥ 100) and only 3 are actually blurry. The
photo is usually fine — the plant simply isn't one of the 38 classes the
model knows. The old gate also threw away the entire 0.50–0.95 confidence
band (41.1% of field photos), where the correct answer is among the top
three about three times in four — which is the direct cause of the
complaint that clear leaf photos were being rejected.

The result screen now reads `confidence_bands` from `model_metadata.json`
(`confident_min` / `possible_min`) and shows one of three tiers —
[ml/ConfidenceTier.kt](app/src/main/java/com/plantguard/app/ml/ConfidenceTier.kt)
is the single place a probability becomes a tier, used for both fresh
predictions and history rows read back later:

1. **Confident** (`>= confident_min`): names one condition, styled in green.
2. **Possible matches** (`possible_min`–`confident_min`): shows the top
   three candidates with their confidences, headed "Possible matches — not
   confident." Deliberately never worded as "most likely" — in this band
   the single best guess is right only about 41% of the time, while the
   correct answer is among the three about 75% of the time.
3. **Not recognised** (`< possible_min`): "This plant may not be one of the
   38 we cover..." — leads with the out-of-taxonomy explanation because
   that's what's usually true, and mentions retaking only second. No blur
   detector was built for this: log10(Laplacian variance) correlates with
   model confidence at -0.013, effectively zero.

A deliberate consequence: `PlantClassifier.classify()` now returns the top
**three** candidates (`ClassCandidate`), not just the winner, and
`HistoryEntry` stores all three (`topCandidatesJson`) so a saved result can
be re-tiered correctly if the model's bands are ever re-tuned. `AppDatabase`
went from schema v1 to v2 with a real `Migration(1, 2)`
([data/history/Migrations.kt](app/src/main/java/com/plantguard/app/data/history/Migrations.kt))
— existing history rows are preserved, not wiped, since `minSdk 24`'s
SQLite predates `ALTER TABLE ... DROP COLUMN`.

## The model contract

Two bundled assets define the entire model contract — nothing about them is
hardcoded in Kotlin:

- `app/src/main/assets/model.tflite` — the real fine-tuned export
  (MobileNetV3Large, PlantDoc fine-tune), **float16-quantized (float32
  input, float32 output)**. See `src/export/to_tflite.py`'s docstring for
  the exact contract the export must match.
- `app/src/main/assets/model_metadata.json` — `class_names` (copied verbatim
  from `src/config.py:PLANTVILLAGE_CLASS_NAMES`, same order), `image_size`,
  `confidence_threshold` (the OOD-rejection cutoff the Python side tuned by
  Youden's J), and `confidence_bands` (`confident_min` / `possible_min`,
  which drive the three result tiers above). The app reads class order and
  every boundary from this file — **never** from Kotlin source.
  `ModelMetadata.kt` requires `confidence_bands` to be present and valid
  (`0 < possible_min < confident_min < 1`) and throws otherwise, rather than
  guessing a default.

**Two things to know if you touch this file next:**

- **`confidence_bands` was hand-added to this asset.**
  `src/export/metadata.py` does not currently emit it — check before
  assuming a fresh Python-side export will carry it forward. If it's
  missing, the app throws at launch (by design, per CLAUDE.md rule 1) rather
  than silently mistiering every result.
- **The `notes` field is stale.** It still describes the old untrained
  uint8 placeholder even though `"placeholder": false` and
  `"quantization": "float16"` describe the real deployed model. Also
  Python-generated, also not fixed here.

**Why float16, not full-integer (INT8).** A real Kaggle export of the
selected model (MobileNetV3Large) measured, on the same validation split the
float model was scored on: INT8 macro-F1 0.5545 (a 0.4133 absolute drop from
0.9678 — the known hard-swish/squeeze-excite INT8 quantization pathology for
this architecture family, not something a larger representative dataset
fixes), float16 macro-F1 0.9682 (effectively lossless). `src/export/
to_tflite.py` still builds and reports INT8 as a comparison artifact
(genuinely useful for the project report) but never deploys it —
`PlantClassifier.kt` requires float32 input/output tensors and crashes
loudly at startup on anything else, rather than silently producing garbage
predictions from a mismatched model.tflite.

The app feeds the model raw pixel values (0-255, resized to
`image_size`×`image_size`, cast straight to float32 — no normalization or
scaling on the app side) and reads the output directly as float32
probabilities — no dequantization step, since float16 quantization only
compresses weights and leaves the I/O tensors as plain float32.
`PlantClassifier.build()` asserts both tensors are actually `FLOAT32` before
use, so an INT8 `model.tflite` dropped in by mistake fails loudly at startup
instead of silently mis-predicting.

### Swapping in a future model

1. Overwrite `app/src/main/assets/model.tflite` with the new (float16)
   export.
2. Overwrite `app/src/main/assets/model_metadata.json` with the new one
   (same schema, including `confidence_bands` — see the caveat above).
3. Rebuild. No Kotlin changes required, *provided* the export is
   float16-quantized (float32 I/O), its output class count matches
   `model_metadata.json`'s `class_names` length, and `confidence_bands` is
   present and valid. `PlantClassifier` and `ModelMetadata` assert all of
   this loudly at startup rather than silently mis-mapping predictions.

## Field accuracy: `assets/field_metrics.json`

The always-visible accuracy notice, the About screen, and each result
tier's "how often is this right" line all read from
`app/src/main/assets/field_metrics.json` — never a hardcoded number in
Kotlin or in `strings.xml` (CLAUDE.md rule 5). Its `source` field records
where the figures come from: **as of this redesign, they were
hand-transcribed** from a Kaggle fine-tune run's reported PlantDoc results
(236-image official test split, accuracy 0.5720, macro-F1 0.5444; per-band
share/top-1/top-3 figures), because `artifacts/results.json` in this repo
predates that run (commit `37211d6`, threshold 0.98, 2578-image split) and
`src/export/` does not currently emit this file. Re-sync this asset by hand
whenever the model is re-evaluated, until the Python export pipeline
generates it directly.

## Disease content

`app/src/main/assets/disease_info.json` has real, researched content for
all 38 classes, each with a citation actually retrieved and checked (not
guessed) against ICAR, an Indian state agricultural university (Tamil Nadu
Agricultural University's Agritech Portal, in particular), a KVK
publication, or an established university plant pathology extension
service (Cornell, UC IPM/ANR, Ohio State, NC State, Clemson, Penn State's
PlantVillage, and others) where Indian-specific literature was thin.
`tests/test_disease_info.py` enforces this going forward:

- All 38 keys must exactly match `src/config.py:PLANTVILLAGE_CLASS_NAMES`
  (no fuzzy matching — a typo'd key fails loudly instead of silently
  producing a class with no info).
- No entry may contain a number immediately followed by a dosage/
  concentration-style unit (`%`, `g/L`, `ml`, `kg`, `ppm`, etc.) — this is
  deliberately broad (no non-chemical number+unit either, e.g. leaf sizes
  or sun-hours), specifically so an agrochemical dose can never creep back
  in unnoticed (CLAUDE.md rule 8).
- `management` stays generic and non-chemical (no product/brand names, no
  doses, no application schedules) — the app's own "confirm with your local
  KVK" line is not a substitute for this rule, it's in addition to it.
- Every entry must have `"status": "verified"` — a real source was fetched
  and its exact supporting text confirmed for everything in the entry.
  There is no "pending" status: this text is shown directly to farmers, and
  a pending flag isn't visible to them, so an unsourced claim is dropped at
  write time rather than kept with a flag. Two entries are shorter than the
  others for exactly this reason — `Grape___Leaf_blight_(Isariopsis_Leaf_Spot)`
  keeps only its sourced symptom description (no management claim could be
  confirmed), and `Soybean___healthy` keeps only its sourced leaf
  description (no cultivation-practice claim could be confirmed); both say
  so honestly in `management` instead of making an unsourced claim. If you
  find a real source for either, add the practice back with a citation.

## Versions used, and why

This redesign added no new Gradle dependencies — `navigation-compose`,
`material-icons-extended`, Compose's animation APIs, and `junit` were
already declared and present in the local Gradle cache. The table below is
unchanged from the original build:

| Library | Version |
|---|---|
| Gradle wrapper | 9.5.0 |
| Android Gradle Plugin (AGP) | 9.3.1 |
| Kotlin / Compose compiler plugin | 2.2.10 |
| KSP | 2.2.10-2.0.2 |
| Compose BOM | 2026.02.01 |
| CameraX | 1.5.1 |
| Room | 2.8.4 |
| Navigation Compose | 2.9.7 |
| **LiteRT** | `com.google.ai.edge.litert:litert:2.1.6` |

**On LiteRT specifically**: LiteRT is Google's current name for what used to
be "TensorFlow Lite." Version 2.1.6 is current stable. It has a newer
`CompiledModel` API (mainly for GPU acceleration), but this app uses the
classic `org.tensorflow.lite.Interpreter` API instead — LiteRT fully
supports it, it's CPU-only (which is all this tiny model needs), and it's
the same well-documented pattern every "TensorFlow Lite" tutorial already
uses. See `ml/PlantClassifier.kt`.

Two AGP-9-era quirks you'll see and can ignore (both already worked around
in `gradle.properties`, with comments there):

- **No `org.jetbrains.kotlin.android` plugin is applied.** AGP 9+ compiles
  Kotlin itself now ("built-in Kotlin support"); applying the classic
  plugin on top of it fails the build.
- **`android.uniquePackageNames=false`** — LiteRT's `litert` artifact and
  its own transitive `litert-api` dependency ship the same manifest
  namespace, which is a known upstream LiteRT packaging bug against AGP 9's
  stricter duplicate-namespace check
  ([google-ai-edge/LiteRT#6965](https://github.com/google-ai-edge/LiteRT/issues/6965)).
  This is Google's own documented temporary mitigation.

If Android Studio's sync ever suggests a different version for something
above, take its suggestion — that's a loud, visible fix, not a silent one.

## Opening, building, and running this (assumes only "Hello World" experience)

1. Open **Android Studio**.
2. **File → Open...** and select the `android` folder inside this repo
   (`...\plantguard-v2\android` — the folder that directly contains
   `settings.gradle.kts`, not the repo root).
3. Android Studio will show "Gradle sync" running in the status bar. Wait
   for it to finish (first time may take a few minutes). If it reports a
   version-mismatch error, click whatever "Fix" / "Upgrade" button it
   offers, then let it sync again — that's expected and fine.
4. Plug in your Android phone via USB. If prompted on the phone, allow USB
   debugging.
5. In the toolbar, pick your phone from the device dropdown (next to the
   green Run ▶ button). If it doesn't appear, make sure USB debugging is
   enabled on the phone (Settings → About phone → tap "Build number" 7
   times → Developer options → USB debugging).
6. Click **Run ▶** (or Shift+F10).
7. On the phone: you'll see the one-time disclaimer, then Home. Tap
   "Identify a Plant" for the camera permission prompt (grant it, or use
   the gallery icon — both work without it).

**What to expect now**: the bundled model is the real fine-tuned export, so
predictions are meaningful. On a clear, well-framed leaf photo you should
mostly land in the confident (green) tier; on a harder field photo, expect
the amber "possible matches" tier with three candidates; on something
outside the 38 covered classes (a hand, soil, a whole plant from far away)
expect "Not recognised," not "Unclear photo." If you see "Unclear photo"
anywhere, that string should no longer exist in this codebase — it was
replaced entirely by the three tiers above.

## Command-line build (optional, what was used to verify this before commit)

From the `android/` folder:

```
gradlew.bat assembleDebug
gradlew.bat test
```

The first confirms the project actually compiles; the second runs
`ConfidenceTierTest`
([app/src/test/java/.../ml/ConfidenceTierTest.kt](app/src/test/java/com/plantguard/app/ml/ConfidenceTierTest.kt)),
a plain JVM unit test pinning the tier boundaries and the "malformed bands
throw" behaviour — no device or emulator needed for either command. You
don't need to run these yourself, Android Studio's Run ▶ button does the
equivalent of the first.
