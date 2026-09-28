package com.plantguard.app.ui.history

import android.app.Application
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.plantguard.app.data.history.AppDatabase
import com.plantguard.app.data.history.HistoryCandidates
import com.plantguard.app.data.history.HistoryEntry
import com.plantguard.app.ml.ConfidenceTier
import com.plantguard.app.ml.ModelMetadata
import com.plantguard.app.ml.tierFor
import com.plantguard.app.util.ClassNameFormatter
import kotlinx.coroutines.flow.SharingStarted
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.map
import kotlinx.coroutines.flow.stateIn

/**
 * One row of the history list, already resolved for display.
 *
 * [tier] is computed here rather than read from the database: a stored tier would
 * become a lie the moment the model's confidence boundaries were re-tuned. See
 * [com.plantguard.app.ml.tierFor].
 */
data class HistoryRow(
    val id: Long,
    val imagePath: String,
    val tier: ConfidenceTier,
    val confidence: Float,
    val timestampMillis: Long,
    /** Display name of the top class, or null for a row shown as unrecognised. */
    val displayName: String?,
    /** How many candidates were stored, for the "3 possible matches" summary. */
    val candidateCount: Int,
)

class HistoryViewModel(application: Application) : AndroidViewModel(application) {

    private val historyDao = AppDatabase.getInstance(application).historyDao()

    val rows: StateFlow<List<HistoryRow>> = historyDao.observeAll()
        .map { entries -> entries.map { toRow(it) } }
        .stateIn(
            scope = viewModelScope,
            // WhileSubscribed with a grace period: the flow stays alive across a
            // brief navigation away and back, without holding a database observer
            // open once the screen is really gone.
            started = SharingStarted.WhileSubscribed(5_000),
            initialValue = emptyList(),
        )

    private fun toRow(entry: HistoryEntry): HistoryRow {
        // ModelMetadata.getInstance, not PlantClassifier.getInstance: this list
        // only ever needs the class names and confidence bands, and building a
        // classifier would map the 6 MB model file into memory for nothing.
        val bands = ModelMetadata.getInstance(getApplication()).confidenceBands
        val tier = tierFor(entry.confidence, bands)
        val candidates = HistoryCandidates.decode(entry.topCandidatesJson)
        return HistoryRow(
            id = entry.id,
            imagePath = entry.imagePath,
            tier = tier,
            confidence = entry.confidence,
            timestampMillis = entry.timestampMillis,
            // The unrecognised tier names nothing in the list either, for the same
            // reason the result screen shows no candidates there.
            displayName = if (tier == ConfidenceTier.UNRECOGNISED) {
                null
            } else {
                entry.topClassName?.let { ClassNameFormatter.humanize(it) }
            },
            candidateCount = candidates.size,
        )
    }
}
