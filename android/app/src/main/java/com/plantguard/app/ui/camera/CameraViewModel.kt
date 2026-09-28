package com.plantguard.app.ui.camera

import android.app.Application
import android.graphics.Bitmap
import android.net.Uri
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.plantguard.app.R
import com.plantguard.app.data.history.AppDatabase
import com.plantguard.app.data.history.HistoryCandidates
import com.plantguard.app.data.history.HistoryEntry
import com.plantguard.app.ml.ImagePreprocessing
import com.plantguard.app.ml.PlantClassifier
import com.plantguard.app.util.ImageStorage
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.receiveAsFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File

data class CameraUiState(
    val isProcessing: Boolean = false,
    val errorMessage: String? = null,
)

class CameraViewModel(application: Application) : AndroidViewModel(application) {

    private val _uiState = MutableStateFlow(CameraUiState())
    val uiState: StateFlow<CameraUiState> = _uiState.asStateFlow()

    /**
     * Navigation is a one-shot event, not state: re-emitting it on every
     * recomposition would push the user to the result screen again after they
     * came back. A Channel delivers it exactly once.
     */
    private val navigationChannel = Channel<Long>(Channel.BUFFERED)
    val navigationEvents: Flow<Long> = navigationChannel.receiveAsFlow()

    private val classifier by lazy { PlantClassifier.getInstance(getApplication()) }
    private val historyDao by lazy { AppDatabase.getInstance(getApplication()).historyDao() }

    fun onImageCaptured(file: File) {
        processImage { ImagePreprocessing.decodeBitmapFromFile(file) }
    }

    fun onImagePicked(uri: Uri) {
        processImage { ImagePreprocessing.decodeBitmapFromUri(getApplication(), uri) }
    }

    /** The camera itself failed to save a frame — nothing to classify. */
    fun onCaptureFailed() {
        _uiState.value = CameraUiState(
            errorMessage = getApplication<Application>().getString(R.string.camera_capture_failed),
        )
    }

    fun onErrorDismissed() {
        _uiState.value = _uiState.value.copy(errorMessage = null)
    }

    /**
     * Decode, classify, save and record one photo, then hand its history id to
     * the result screen.
     *
     * The whole chain runs on [Dispatchers.IO] off the main thread. Note that the
     * result screen reads the saved row rather than receiving the
     * ClassificationResult directly, which is why the top three candidates are
     * persisted here: they are the transport as well as the record.
     */
    private fun processImage(decode: () -> Bitmap) {
        _uiState.value = CameraUiState(isProcessing = true)
        viewModelScope.launch {
            try {
                val entryId = withContext(Dispatchers.IO) {
                    val bitmap = decode()
                    val result = classifier.classify(bitmap)
                    val savedFile = ImageStorage.save(getApplication(), bitmap)
                    val entry = HistoryEntry(
                        imagePath = savedFile.absolutePath,
                        topClassName = result.candidates.first().className,
                        confidence = result.topConfidence,
                        inferenceLatencyMs = result.inferenceLatencyMs,
                        timestampMillis = System.currentTimeMillis(),
                        topCandidatesJson = HistoryCandidates.encode(result.candidates),
                    )
                    historyDao.insert(entry)
                }
                navigationChannel.send(entryId)
                _uiState.value = CameraUiState(isProcessing = false)
            } catch (e: Exception) {
                // A corrupt or unreadable photo shouldn't crash the app — show it
                // to the user instead, so they can just retake it.
                _uiState.value = CameraUiState(
                    isProcessing = false,
                    errorMessage = getApplication<Application>()
                        .getString(R.string.camera_process_failed, e.message ?: ""),
                )
            }
        }
    }
}
