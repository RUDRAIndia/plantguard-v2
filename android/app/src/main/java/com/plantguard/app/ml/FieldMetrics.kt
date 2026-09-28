package com.plantguard.app.ml

import android.content.Context
import org.json.JSONObject

/** Measured reliability of one confidence band on real field photographs. */
data class BandMetrics(
    /** Share of field photos that land in this band, 0..1. */
    val shareOfPhotos: Float,
    /** How often the single top guess is correct here, 0..1. */
    val top1Accuracy: Float,
    /**
     * How often the correct answer is somewhere in the top three. Only measured
     * for the "possible" band, where the app actually shows three candidates —
     * null elsewhere, because nothing would display it.
     */
    val top3Accuracy: Float?,
)

/**
 * The honest numbers the UI quotes about itself, read from
 * assets/field_metrics.json rather than written into Kotlin strings.
 *
 * CLAUDE.md rule 5 says every number shown to a user must come from a metrics
 * file produced by a real run, never from memory. Keeping them in one asset with
 * a `source` field means there is exactly one place to audit and re-sync when
 * the model is re-exported, and no figure is buried in a layout somewhere.
 *
 * Read the `source` field in that asset before trusting these: at the time of
 * writing it is transcribed from a Kaggle fine-tune run by hand, because
 * artifacts/results.json in this repo predates that run.
 */
data class FieldMetrics(
    val evaluatedOn: String,
    val numImages: Int,
    val fieldAccuracy: Float,
    val fieldMacroF1: Float,
    val confident: BandMetrics,
    val possible: BandMetrics,
    val unrecognised: BandMetrics,
) {
    /** Returns the measured figures for whichever band [tier] names. */
    fun forTier(tier: ConfidenceTier): BandMetrics = when (tier) {
        ConfidenceTier.CONFIDENT -> confident
        ConfidenceTier.POSSIBLE -> possible
        ConfidenceTier.UNRECOGNISED -> unrecognised
    }

    companion object {
        private const val ASSET_FILE_NAME = "field_metrics.json"

        @Volatile
        private var instance: FieldMetrics? = null

        /**
         * Parsed once per process. Several screens quote these numbers, and
         * re-reading the asset on every recomposition would be wasteful.
         */
        fun getInstance(context: Context): FieldMetrics {
            return instance ?: synchronized(this) {
                instance ?: loadFromAssets(context).also { instance = it }
            }
        }

        private fun loadFromAssets(context: Context): FieldMetrics {
            val json = context.assets.open(ASSET_FILE_NAME).bufferedReader().use { it.readText() }
            val root = JSONObject(json)
            val bands = root.getJSONObject("bands")
            return FieldMetrics(
                evaluatedOn = root.getString("evaluated_on"),
                numImages = root.getInt("num_images"),
                fieldAccuracy = root.getDouble("field_accuracy").toFloat(),
                fieldMacroF1 = root.getDouble("field_macro_f1").toFloat(),
                confident = bands.readBand("confident"),
                possible = bands.readBand("possible"),
                unrecognised = bands.readBand("unrecognised"),
            )
        }

        /**
         * getJSONObject/getDouble rather than the opt* variants, so a malformed
         * or half-updated metrics file fails loudly here instead of quietly
         * showing a farmer a 0% reliability figure (CLAUDE.md rule 1).
         * top3_accuracy is the one genuinely optional field — only the
         * "possible" band has one.
         */
        private fun JSONObject.readBand(name: String): BandMetrics {
            val band = getJSONObject(name)
            return BandMetrics(
                shareOfPhotos = band.getDouble("share_of_photos").toFloat(),
                top1Accuracy = if (band.has("top1_accuracy")) {
                    band.getDouble("top1_accuracy").toFloat()
                } else {
                    // The unrecognised band names no class at all, so "how often
                    // is the top guess right" is not a meaningful figure there.
                    0f
                },
                top3Accuracy = if (band.has("top3_accuracy")) {
                    band.getDouble("top3_accuracy").toFloat()
                } else {
                    null
                },
            )
        }
    }
}
