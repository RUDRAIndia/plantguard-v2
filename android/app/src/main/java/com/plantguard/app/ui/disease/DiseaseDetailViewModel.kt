package com.plantguard.app.ui.disease

import android.app.Application
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.SavedStateHandle
import androidx.lifecycle.viewModelScope
import com.plantguard.app.R
import com.plantguard.app.data.disease.DiseaseInfo
import com.plantguard.app.data.disease.DiseaseInfoRepository
import com.plantguard.app.ml.PlantClassifier
import com.plantguard.app.navigation.NavRoutes
import com.plantguard.app.util.ClassNameFormatter
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

sealed interface DiseaseDetailUiState {
    data object Loading : DiseaseDetailUiState

    data class Error(val message: String) : DiseaseDetailUiState

    data class Loaded(
        val displayName: String,
        val info: DiseaseInfo?,
    ) : DiseaseDetailUiState
}

/**
 * Loads one condition's details by its model class index.
 *
 * The index is resolved against model_metadata.json's class_names, the same list
 * the model's output is mapped through — so this screen cannot drift out of step
 * with what the classifier actually predicted.
 */
class DiseaseDetailViewModel(
    application: Application,
    savedStateHandle: SavedStateHandle,
) : AndroidViewModel(application) {

    private val _uiState = MutableStateFlow<DiseaseDetailUiState>(DiseaseDetailUiState.Loading)
    val uiState: StateFlow<DiseaseDetailUiState> = _uiState.asStateFlow()

    init {
        val classIndex: Int = savedStateHandle[NavRoutes.DISEASE_ARG_INDEX]
            ?: error("Disease route requires a ${NavRoutes.DISEASE_ARG_INDEX} argument")
        load(classIndex)
    }

    private fun load(classIndex: Int) {
        viewModelScope.launch {
            _uiState.value = withContext(Dispatchers.IO) { buildState(classIndex) }
        }
    }

    private fun buildState(classIndex: Int): DiseaseDetailUiState {
        val context = getApplication<Application>()
        val classNames = PlantClassifier.getInstance(context).metadata.classNames

        // An out-of-range index means the route was built from a class name that
        // is not in the model's list — a drift between the two assets. Reported,
        // not crashed, because it only breaks this one screen.
        val className = classNames.getOrNull(classIndex)
            ?: return DiseaseDetailUiState.Error(
                context.getString(R.string.disease_error_unknown_index),
            )

        val info = DiseaseInfoRepository.getInstance(context).lookup(className)
        return DiseaseDetailUiState.Loaded(
            displayName = info?.displayName ?: ClassNameFormatter.humanize(className),
            info = info,
        )
    }
}
