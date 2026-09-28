package com.plantguard.app.ui.result

import android.app.Application
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.SavedStateHandle
import androidx.lifecycle.viewModelScope
import com.plantguard.app.R
import com.plantguard.app.data.disease.DiseaseInfo
import com.plantguard.app.data.disease.DiseaseInfoRepository
import com.plantguard.app.data.history.AppDatabase
import com.plantguard.app.data.history.HistoryCandidates
import com.plantguard.app.data.history.HistoryEntry
import com.plantguard.app.ml.ClassCandidate
import com.plantguard.app.ml.ConfidenceTier
import com.plantguard.app.ml.PlantClassifier
import com.plantguard.app.ml.tierFor
import com.plantguard.app.navigation.NavRoutes
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

/** One candidate plus whatever disease information exists for it. */
data class CandidateWithInfo(
    val candidate: ClassCandidate,
    val info: DiseaseInfo?,
) {
    /** Display name from disease_info.json, or a readable form of the raw class name. */
    val displayName: String
        get() = info?.displayName
            ?: com.plantguard.app.util.ClassNameFormatter.humanize(candidate.className)
}

sealed interface ResultUiState {
    data object Loading : ResultUiState

    /**
     * Something is wrong with this row rather than with the photo — a missing
     * entry, or an id that no longer exists. Previously these threw and crashed
     * the app; a user who opens a stale history link deserves a message.
     */
    data class Error(val message: String) : ResultUiState

    /**
     * A loaded result. [tier] is derived from the stored confidence against the
     * boundaries currently in model_metadata.json, so it always reflects the
     * shipped model rather than whatever was true when the row was written.
     */
    data class Loaded(
        val entry: HistoryEntry,
        val tier: ConfidenceTier,
        val candidates: List<CandidateWithInfo>,
    ) : ResultUiState {
        /** The single best candidate — what a confident diagnosis names. */
        val top: CandidateWithInfo get() = candidates.first()
    }
}

class ResultViewModel(
    application: Application,
    savedStateHandle: SavedStateHandle,
) : AndroidViewModel(application) {

    private val _uiState = MutableStateFlow<ResultUiState>(ResultUiState.Loading)
    val uiState: StateFlow<ResultUiState> = _uiState.asStateFlow()

    init {
        val entryId: Long? = savedStateHandle[NavRoutes.RESULT_ARG_ID]
        if (entryId == null) {
            // A missing route argument is a programming error, not a data problem,
            // so it fails loudly rather than rendering a friendly empty screen.
            error("Result route requires a ${NavRoutes.RESULT_ARG_ID} argument")
        }
        load(entryId)
    }

    private fun load(entryId: Long) {
        viewModelScope.launch {
            // withContext(IO): the disease asset is parsed on first use, and that
            // must not happen on the main thread.
            val state = withContext(Dispatchers.IO) { buildState(entryId) }
            _uiState.value = state
        }
    }

    private suspend fun buildState(entryId: Long): ResultUiState {
        val context = getApplication<Application>()
        val entry = AppDatabase.getInstance(context).historyDao().getById(entryId)
            ?: return ResultUiState.Error(context.getString(R.string.result_error_missing))

        val metadata = PlantClassifier.getInstance(context).metadata
        val diseaseInfo = DiseaseInfoRepository.getInstance(context)

        val stored = HistoryCandidates.decode(entry.topCandidatesJson)
        val candidates = if (stored.isNotEmpty()) {
            stored
        } else {
            // A row written by v1 of the schema kept only the winner. Rebuild a
            // one-item list from it rather than inventing runners-up.
            val className = entry.topClassName
                ?: return ResultUiState.Error(
                    context.getString(R.string.result_error_legacy_no_class),
                )
            listOf(
                ClassCandidate(
                    classIndex = metadata.classNames.indexOf(className),
                    className = className,
                    confidence = entry.confidence,
                ),
            )
        }

        return ResultUiState.Loaded(
            entry = entry,
            tier = tierFor(entry.confidence, metadata.confidenceBands),
            candidates = candidates.map { candidate ->
                CandidateWithInfo(candidate, diseaseInfo.lookup(candidate.className))
            },
        )
    }
}
