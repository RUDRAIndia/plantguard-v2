package com.plantguard.app.data.disease

import android.content.Context
import org.json.JSONObject

/**
 * Reads android/app/src/main/assets/disease_info.json. Only a handful of
 * the 38 classes have a real entry right now (the rest are simply absent —
 * see that file's _readme field) — [lookup] returning null for a missing
 * class name is the expected, normal case, not an error.
 */
class DiseaseInfoRepository(context: Context) {

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

    private companion object {
        const val ASSET_FILE_NAME = "disease_info.json"
    }
}
