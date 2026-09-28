package com.plantguard.app.util

/**
 * PlantVillage class names follow a fixed "Species___Condition" convention
 * (e.g. "Apple___Apple_scab"). This turns that raw string into something
 * readable ("Apple - Apple scab") for display, without needing a
 * disease_info.json entry to exist. Every class does have one now, so this is
 * the fallback for the case where that asset and model_metadata.json have
 * drifted out of sync (see DiseaseInfoRepository).
 */
object ClassNameFormatter {

    fun humanize(className: String): String {
        val parts = className.split("___", limit = 2)
        return parts.joinToString(" - ") { part -> part.replace("_", " ").trim() }
    }
}
