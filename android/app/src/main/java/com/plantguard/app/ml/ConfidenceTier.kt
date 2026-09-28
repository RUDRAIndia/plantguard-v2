package com.plantguard.app.ml

/**
 * How much the app is willing to claim about one prediction.
 *
 * This replaced a binary "confident / unclear photo" gate, which was wrong in
 * practice. Measured on PlantDoc's 236-image official test split: of the 31
 * images scoring below 0.50 confidence, 28 are sharp (Laplacian variance >= 100)
 * and only 3 are actually blurry — so telling the user "unclear photo, please
 * retake" was the wrong explanation about nine times in ten. The photo is
 * usually fine; the plant simply isn't one of the 38 classes the model knows.
 *
 * The old gate also threw away the whole middle band, which is 41.1% of field
 * photos and where the correct answer is among the top three about three times
 * in four. That band is now [POSSIBLE] instead of being rejected.
 *
 * Blur detection is deliberately not implemented: log10(Laplacian variance)
 * correlates with model confidence at -0.013 (effectively zero) and median
 * sharpness is near-identical across all three bands, so a blur detector cannot
 * separate these cases. The fix was wording and tiering.
 */
enum class ConfidenceTier {
    /** >= confident_min. A diagnosis worth naming; top-1 accuracy 0.8148 on field photos. */
    CONFIDENT,

    /** possible_min..confident_min. Show three candidates, do not claim a winner. */
    POSSIBLE,

    /** < possible_min. Probably not one of the 38 covered classes at all. */
    UNRECOGNISED,
}

/**
 * The single place a probability becomes a tier. Both fresh predictions and
 * rows read back out of history go through this, so a stored result is always
 * displayed under the boundaries currently shipped in model_metadata.json —
 * re-tuning the threshold re-tiers past results consistently rather than
 * leaving them labelled with boundaries that no longer exist.
 *
 * Boundaries are inclusive at the bottom of each band: a probability of exactly
 * `confident_min` counts as [ConfidenceTier.CONFIDENT], matching the Python
 * side's `>=` comparison against the tuned threshold.
 */
fun tierFor(confidence: Float, bands: ConfidenceBands): ConfidenceTier = when {
    confidence >= bands.confidentMin -> ConfidenceTier.CONFIDENT
    confidence >= bands.possibleMin -> ConfidenceTier.POSSIBLE
    else -> ConfidenceTier.UNRECOGNISED
}
