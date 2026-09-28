package com.plantguard.app.data.disease

import android.content.Context
import org.json.JSONObject

/**
 * Reads android/app/src/main/assets/disease_info.json, which now holds a
 * verified entry for every one of the 38 classes — name, symptoms, generic
 * management practice and a real citation, with no chemical, dose or schedule
 * anywhere (CLAUDE.md rule 8; enforced by tests/test_disease_info.py).
 *
 * [lookup] is still nullable-returning: the keys must match
 * model_metadata.json's class_names exactly, and a null means those two assets
 * have drifted apart. Callers fall back to formatting the raw class name rather
 * than crashing, so a single missing entry degrades one screen instead of
 * breaking the app.
 *
 * A process-lifetime singleton, because the asset is ~36 KB of JSON and a
 * fresh instance per screen re-parsed all of it on every result view.
 */
class DiseaseInfoRepository private constructor(context: Context) {

    private val entries: Map<String, DiseaseInfo> by lazy { loadEntries(context) }

    fun lookup(className: String): DiseaseInfo? = entries[className]

    private fun loadEntries(context: Context): Map<String, DiseaseInfo> {
        val json = context.assets.open(ASSET_FILE_NAME).bufferedReader().use { it.readText() }
        val root = JSONObject(json)
        val entriesJson = root.getJSONObject("entries")
        val result = mutableMapOf<String, DiseaseInfo>()
        for (className in entriesJson.keys()) {
            val entry = entriesJson.getJSONObject(className)
            result[className] = DiseaseInfo(
                displayName = entry.getString("display_name"),
                symptoms = entry.getString("symptoms"),
                management = entry.getString("management"),
                citation = entry.getString("citation"),
                status = entry.getString("status"),
            )
        }
        return result
    }

    companion object {
        private const val ASSET_FILE_NAME = "disease_info.json"

        @Volatile
        private var instance: DiseaseInfoRepository? = null

        fun getInstance(context: Context): DiseaseInfoRepository {
            return instance ?: synchronized(this) {
                // applicationContext, never an Activity's: this instance outlives
                // any single screen, and holding an Activity would leak it.
                instance ?: DiseaseInfoRepository(context.applicationContext).also { instance = it }
            }
        }
    }
}
