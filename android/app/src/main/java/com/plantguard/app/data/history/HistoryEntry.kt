package com.plantguard.app.data.history

import androidx.room.ColumnInfo
import androidx.room.Entity
import androidx.room.PrimaryKey

/**
 * One past prediction, persisted locally (Room).
 *
 * Note what is *not* stored here: the confidence tier. A tier is a pure function
 * of [confidence] and the boundaries in model_metadata.json, so it is derived
 * when a row is displayed (see [com.plantguard.app.ml.tierFor]). If the model is
 * re-tuned and those boundaries move, past results re-tier consistently instead
 * of being frozen with labels that no longer match the shipped model.
 *
 * [topClassName] is the raw class name from model_metadata.json's class_names
 * (e.g. "Apple___Apple_scab") for the highest-scoring class, whatever the tier —
 * including rows the UI presents as "not recognised", where it is recorded but
 * never shown. It is nullable only because rows written by v1 of this schema
 * could legitimately have no class at all.
 */
@Entity(tableName = "history_entries")
data class HistoryEntry(
    @PrimaryKey(autoGenerate = true)
    val id: Long = 0,
    val imagePath: String,
    // Kept under its v1 column name so renaming the Kotlin property costs no
    // SQL: in v1 this column was null whenever the old binary "unclear" gate
    // fired, and it now always holds the top class.
    @ColumnInfo(name = "classNameOrNull")
    val topClassName: String?,
    val confidence: Float,
    val inferenceLatencyMs: Long,
    val timestampMillis: Long,
    /**
     * The top three candidates, encoded by [HistoryCandidates]. Rows written by
     * v1 of this schema have `"[]"` here — the old code kept only the winner —
     * so anything reading this must cope with an empty list and fall back to
     * [topClassName].
     */
    val topCandidatesJson: String,
)
