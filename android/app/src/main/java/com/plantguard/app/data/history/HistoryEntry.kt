package com.plantguard.app.data.history

import androidx.room.Entity
import androidx.room.PrimaryKey

/**
 * One past prediction, persisted locally (Room). [classNameOrNull] is the
 * raw class name from model_metadata.json's class_names (e.g.
 * "Apple___Apple_scab"), null when [isUnclear] is true — the OOD-rejection
 * path never assigns a disease at all, so there is nothing to store in that
 * field for those rows.
 */
@Entity(tableName = "history_entries")
data class HistoryEntry(
    @PrimaryKey(autoGenerate = true)
    val id: Long = 0,
    val imagePath: String,
    val isUnclear: Boolean,
    val classNameOrNull: String?,
    val confidence: Float,
    val inferenceLatencyMs: Long,
    val timestampMillis: Long,
)
