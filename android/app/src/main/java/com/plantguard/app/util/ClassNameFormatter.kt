package com.plantguard.app.util

/**
 * PlantVillage class names follow a fixed "Species___Condition" convention
 * (e.g. "Apple___Apple_scab"). This turns that raw string into something
 * readable ("Apple - Apple scab") for display, without needing a
 * disease_info.json entry to exist — most of the 38 classes don't have one
 * yet (see DiseaseInfoRepository).
 */
object ClassNameFormatter {

    fun humanize(className: String): String {
        val parts = className.split("___", limit = 2)
        return parts.joinToString(" - ") { part -> part.replace("_", " ").trim() }
    }
}
