package com.plantguard.app.ml

import android.content.Context
import org.json.JSONObject

/**
 * Mirrors android/app/src/main/assets/model_metadata.json. This is the
 * *only* place class order and the OOD-rejection threshold come from —
 * never hardcode either in Kotlin. On Day 9 the real export replaces both
 * this JSON and model.tflite together; nothing else in the app should need
 * to change.
 */
data class ModelMetadata(
    val imageSize: Int,
    val confidenceThreshold: Float,
    val classNames: List<String>,
    val placeholder: Boolean,
) {
    companion object {
        private const val ASSET_FILE_NAME = "model_metadata.json"

        fun loadFromAssets(context: Context): ModelMetadata {
            val json = context.assets.open(ASSET_FILE_NAME).bufferedReader().use { it.readText() }
            val root = JSONObject(json)
            val namesJson = root.getJSONArray("class_names")
            val names = (0 until namesJson.length()).map { i -> namesJson.getString(i) }
            return ModelMetadata(
                imageSize = root.getInt("image_size"),
                confidenceThreshold = root.getDouble("confidence_threshold").toFloat(),
                classNames = names,
                placeholder = root.optBoolean("placeholder", false),
            )
        }
    }
}
