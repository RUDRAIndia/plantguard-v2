package com.plantguard.app.ui.result

import android.app.Application
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.SavedStateHandle
import androidx.lifecycle.viewModelScope
import com.plantguard.app.data.disease.DiseaseInfo
import com.plantguard.app.data.disease.DiseaseInfoRepository
import com.plantguard.app.data.history.AppDatabase
import com.plantguard.app.data.history.HistoryEntry
import com.plantguard.app.navigation.NavRoutes
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.launch

sealed interface ResultUiState {
    data object Loading : ResultUiState

    /** The OOD-rejection path fired for this entry — no disease is shown, on purpose. */
    data class Unclear(val entry: HistoryEntry) : ResultUiState

    data class Disease(
        val entry: HistoryEntry,
        val className: String,
        val diseaseInfo: DiseaseInfo?,
    ) : ResultUiState
}

class ResultViewModel(
    application: Application,
    savedStateHandle: SavedStateHandle,
) : AndroidViewModel(application) {

    private val historyDao = AppDatabase.getInstance(application).historyDao()
    private val diseaseInfoRepository = DiseaseInfoRepository(application)

    private val _uiState = MutableStateFlow<ResultUiState>(ResultUiState.Loading)
    val uiState: StateFlow<ResultUiState> = _uiState

    init {
        val entryId: Long = savedStateHandle.get<Long>(NavRoutes.RESULT_ARG_ID)
            ?: error("Result route requires a ${NavRoutes.RESULT_ARG_ID} argument")

        viewModelScope.launch {
            val entry = historyDao.getById(entryId)
                ?: error("History entry $entryId not found")

            _uiState.value = if (entry.isUnclear) {
                ResultUiState.Unclear(entry)
            } else {
                val className = entry.classNameOrNull
                    ?: error("History entry $entryId is not marked unclear but has no classNameOrNull")
                ResultUiState.Disease(
                    entry = entry,
                    className = className,
                    diseaseInfo = diseaseInfoRepository.lookup(className),
                )
            }
        }
    }
}
