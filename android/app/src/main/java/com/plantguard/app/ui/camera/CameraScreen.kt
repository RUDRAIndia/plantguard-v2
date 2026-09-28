package com.plantguard.app.ui.camera

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.net.Uri
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.ImageCapture
import androidx.camera.core.ImageCaptureException
import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.core.content.ContextCompat
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.lifecycle.viewmodel.compose.viewModel
import java.io.File

/**
 * The capture screen: live preview, framing guide, shutter and gallery picker.
 *
 * Three states, decided by camera permission. The gallery path works in all of
 * them — a user who refuses the camera can still analyse a photo they already
 * have, and that path needs no permission at all on any supported API level
 * because it goes through the system photo picker.
 */
@Composable
fun CameraScreen(
    onNavigateToResult: (Long) -> Unit,
    onNavigateUp: () -> Unit,
    onOpenPhotoTips: () -> Unit,
    viewModel: CameraViewModel = viewModel(),
) {
    val context = LocalContext.current
    val uiState by viewModel.uiState.collectAsStateWithLifecycle()

    var hasCameraPermission by remember {
        mutableStateOf(
            ContextCompat.checkSelfPermission(context, Manifest.permission.CAMERA) ==
                PackageManager.PERMISSION_GRANTED,
        )
    }
    // rememberSaveable: this survives the configuration change that a trip to
    // Android Settings and back can cause, so the "permanently denied" screen
    // does not flicker back to the rationale.
    var permissionPermanentlyDenied by rememberSaveable { mutableStateOf(false) }

    val permissionLauncher = rememberLauncherForActivityResult(
        ActivityResultContracts.RequestPermission(),
    ) { granted ->
        hasCameraPermission = granted
        if (!granted) {
            // Android gives no direct "don't ask again" signal. The convention is:
            // if the system will no longer show a rationale after a denial, the
            // user has permanently denied it.
            permissionPermanentlyDenied = !context.shouldShowCameraRationale()
        }
    }

    val galleryLauncher = rememberLauncherForActivityResult(
        ActivityResultContracts.PickVisualMedia(),
    ) { uri: Uri? ->
        if (uri != null) viewModel.onImagePicked(uri)
    }
    val launchGalleryPicker = {
        galleryLauncher.launch(
            PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly),
        )
    }

    LaunchedEffect(Unit) {
        if (!hasCameraPermission) permissionLauncher.launch(Manifest.permission.CAMERA)
    }

    LaunchedEffect(Unit) {
        viewModel.navigationEvents.collect { entryId -> onNavigateToResult(entryId) }
    }

    // One ImageCapture instance for the life of the screen, shared by the preview
    // binding and the shutter.
    val imageCapture = remember { ImageCapture.Builder().build() }

    Box(modifier = Modifier.fillMaxSize()) {
        when {
            hasCameraPermission -> CameraCaptureContent(
                imageCapture = imageCapture,
                isProcessing = uiState.isProcessing,
                onShutter = {
                    takePhoto(
                        context = context,
                        imageCapture = imageCapture,
                        onImageCaptured = viewModel::onImageCaptured,
                        onError = viewModel::onCaptureFailed,
                    )
                },
                onPickGallery = launchGalleryPicker,
                onNavigateUp = onNavigateUp,
                onOpenPhotoTips = onOpenPhotoTips,
            )

            permissionPermanentlyDenied -> CameraPermissionDenied(
                onPickGallery = launchGalleryPicker,
                onNavigateUp = onNavigateUp,
            )

            else -> CameraPermissionRationale(
                onRequestPermission = { permissionLauncher.launch(Manifest.permission.CAMERA) },
                onPickGallery = launchGalleryPicker,
                onNavigateUp = onNavigateUp,
            )
        }

        AnimatedVisibility(
            visible = uiState.isProcessing,
            enter = fadeIn(),
            exit = fadeOut(),
        ) {
            ScanningOverlay()
        }

        // A dismissible banner, not a silent log. Both a failed capture and an
        // unreadable photo land here.
        CameraErrorBanner(
            message = uiState.errorMessage,
            onDismiss = viewModel::onErrorDismissed,
            modifier = Modifier.fillMaxSize(),
        )
    }
}

/**
 * Whether Android would still show a permission rationale — false after the user
 * has permanently denied the permission.
 *
 * Needs an Activity, and `LocalContext` inside a Compose hierarchy is one; the
 * cast is guarded so a preview context can never crash the screen.
 */
private fun Context.shouldShowCameraRationale(): Boolean {
    val activity = this as? android.app.Activity ?: return true
    return androidx.core.app.ActivityCompat.shouldShowRequestPermissionRationale(
        activity,
        Manifest.permission.CAMERA,
    )
}

/**
 * Takes one photo to a file in the cache directory.
 *
 * Cache, not permanent storage: the bitmap is re-saved to the app's files
 * directory by ImageStorage once it has been decoded and classified, so this
 * intermediate JPEG is disposable and Android may reclaim it.
 */
private fun takePhoto(
    context: Context,
    imageCapture: ImageCapture,
    onImageCaptured: (File) -> Unit,
    onError: () -> Unit,
) {
    val photoFile = File(context.cacheDir, "capture_${System.currentTimeMillis()}.jpg")
    val outputOptions = ImageCapture.OutputFileOptions.Builder(photoFile).build()

    imageCapture.takePicture(
        outputOptions,
        ContextCompat.getMainExecutor(context),
        object : ImageCapture.OnImageSavedCallback {
            override fun onImageSaved(output: ImageCapture.OutputFileResults) {
                onImageCaptured(photoFile)
            }

            override fun onError(exception: ImageCaptureException) {
                // Previously this only printed a stack trace, so a failed shutter
                // looked to the user like the button simply did nothing.
                exception.printStackTrace()
                onError()
            }
        },
    )
}
