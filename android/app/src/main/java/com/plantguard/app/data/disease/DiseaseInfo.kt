package com.plantguard.app.data.disease

/** One entry from the bundled disease_info.json asset. */
data class DiseaseInfo(
    val displayName: String,
    val symptoms: String,
    val management: String,
    val citation: String,
    val status: String,
)
