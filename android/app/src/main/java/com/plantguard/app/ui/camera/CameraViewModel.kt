package com.plantguard.app.ui.camera

import android.app.Application
import android.graphics.Bitmap
import android.net.Uri
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.plantguard.app.data.history.AppDatabase
import com.plantguard.app.data.history.HistoryEntry
import com.plantguard.app.ml.ImagePreprocessing
import com.plantguard.app.ml.PlantClassifier
import com.plantguard.app.util.ImageStorage
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.receiveAsFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File

data class CameraUiState(
    val isProcessing: Boolean = false,
    val errorMessage: String? = null,
)

/**
 * Owns the capture/pick -> preprocess -> classify -> save -> record pipeline.
 * Runs entirely off the main thread (Dispatchers.IO — the dominant costs
 * here are file decode and the tiny CPU inference, neither of which may
 * touch the UI thread or the app will jank/ANR during a photo).
 */
class CameraViewModel(application: Application) : AndroidViewModel(application) {

    private val _uiState = MutableStateFlow(CameraUiState())
    val uiState: StateFlow<CameraUiState> = _uiState

    // One-shot "go to this history entry's Result screen" event. A Channel
    // (not a second StateFlow) because navigation should fire exactly once
    // per capture, not re-fire on every recomposition/config change the way
    // a re-collected StateFlow value would.
    private val navigationChannel = Channel<Long>(Channel.BUFFERED)
    val navigationEvents = navigationChannel.receiveAsFlow()

    private val classifier by lazy { PlantClassifier.getInstance(getApplication()) }
    private val historyDao by lazy { AppDatabase.getInstance(getApplication()).historyDao() }

    fun onImageCaptured(file: File) {
        processImage { ImagePreprocessing.decodeBitmapFromFile(file) }
    }

    fun onImagePicked(uri: Uri) {
        processImage { ImagePreprocessing.decodeBitmapFromUri(getApplication(), uri) }
    }

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
                        isUnclear = result.isUnclear,
                        classNameOrNull = result.className,
                        confidence = result.confidence,
                        inferenceLatencyMs = result.inferenceLatencyMs,
                        timestampMillis = System.currentTimeMillis(),
                    )
                    historyDao.insert(entry)
                }
                navigationChannel.send(entryId)
                _uiState.value = CameraUiState(isProcessing = false)
            } catch (e: Exception) {
                // A corrupt/unreadable photo shouldn't crash the app — show
                // it to the user instead, so the user can just retake it.
                _uiState.value = CameraUiState(
                    isProcessing = false,
                    errorMessage = "Could not process that photo: ${e.message}",
                )
            }
        }
    }
}
