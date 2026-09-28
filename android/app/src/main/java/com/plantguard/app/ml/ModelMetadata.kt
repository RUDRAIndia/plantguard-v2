package com.plantguard.app.ml

import android.content.Context
import org.json.JSONObject

/**
 * The two confidence boundaries that split every prediction into one of three
 * tiers (see [ConfidenceTier]). Both are read from model_metadata.json's
 * `confidence_bands` object and are never hardcoded in Kotlin: they are tuning
 * outputs of the Python side, and a re-tuned model must be able to change them
 * by shipping a new metadata file alone.
 *
 * Validated in [init] rather than by the caller, so a `ConfidenceBands` value can
 * never exist half-formed anywhere in the app — the moment a malformed
 * model_metadata.json is parsed, construction fails loudly (CLAUDE.md rule 1)
 * instead of quietly mistiering every result that follows.
 *
 * @param confidentMin lowest probability still treated as a confident diagnosis.
 * @param possibleMin lowest probability still worth showing as a candidate at all.
 */
data class ConfidenceBands(
    val confidentMin: Float,
    val possibleMin: Float,
) {
    init {
        // Probabilities are softmax outputs, so both boundaries must sit
        // strictly inside 0..1, and the confident band must start above the
        // possible band — otherwise one of the three tiers would be empty and
        // unreachable.
        require(possibleMin > 0f && confidentMin < 1f) {
            "confidence_bands must lie strictly inside 0..1, but got " +
                "possible_min=$possibleMin, confident_min=$confidentMin."
        }
        require(possibleMin < confidentMin) {
            "confidence_bands are inverted: possible_min=$possibleMin must be strictly " +
                "less than confident_min=$confidentMin, otherwise the 'possible matches' " +
                "tier can never be reached and hedged results would be shown as confident ones."
        }
    }
}

/**
 * Everything the app needs to know about the bundled model.tflite, read from
 * the model_metadata.json shipped beside it. Nothing here is hardcoded in
 * Kotlin — swapping in a new export means replacing two asset files, no code
 * change (see android/README.md, "The model contract").
 *
 * [confidenceThreshold] is the OOD-rejection cutoff the Python side tuned by
 * Youden's J. It is kept for provenance and for the About screen, but it is
 * *not* what drives the UI tiers — [confidenceBands] is. Today the two agree
 * (`confident_min` == `confidence_threshold` == 0.95), which is expected, since
 * the confident band starts exactly where the OOD gate opens. They are
 * deliberately not asserted equal: that is not an invariant this project
 * states, and a future run is allowed to set them apart.
 */
data class ModelMetadata(
    val imageSize: Int,
    val confidenceThreshold: Float,
    val confidenceBands: ConfidenceBands,
    val classNames: List<String>,
    val placeholder: Boolean,
) {
    companion object {
        private const val ASSET_FILE_NAME = "model_metadata.json"

        @Volatile
        private var instance: ModelMetadata? = null

        /**
         * The parsed metadata, cached for the life of the process.
         *
         * Screens that only need class names or band boundaries use this rather
         * than PlantClassifier, so that opening History or a saved result never
         * builds a LiteRT Interpreter over the 6 MB model just to read two
         * numbers out of a small JSON file.
         */
        fun getInstance(context: Context): ModelMetadata {
            return instance ?: synchronized(this) {
                instance ?: loadFromAssets(context).also { instance = it }
            }
        }

        fun loadFromAssets(context: Context): ModelMetadata {
            val json = context.assets.open(ASSET_FILE_NAME).bufferedReader().use { it.readText() }
            val root = JSONObject(json)
            val namesJson = root.getJSONArray("class_names")
            val names = (0 until namesJson.length()).map { i -> namesJson.getString(i) }

            // getJSONObject (not optJSONObject) on purpose: CLAUDE.md rule 1
            // forbids silently substituting a default. A metadata file without
            // confidence_bands cannot be tiered, and guessing boundaries would
            // mislabel real diagnoses, so this throws instead.
            val bandsJson = root.getJSONObject("confidence_bands")
            val bands = ConfidenceBands(
                confidentMin = bandsJson.getDouble("confident_min").toFloat(),
                possibleMin = bandsJson.getDouble("possible_min").toFloat(),
            )

            return ModelMetadata(
                imageSize = root.getInt("image_size"),
                confidenceThreshold = root.getDouble("confidence_threshold").toFloat(),
                confidenceBands = bands,
                classNames = names,
                placeholder = root.optBoolean("placeholder", false),
            )
        }
    }
}
