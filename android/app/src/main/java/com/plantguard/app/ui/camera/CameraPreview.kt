package com.plantguard.app.ui.camera

import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageCapture
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import androidx.lifecycle.compose.LocalLifecycleOwner

/**
 * The live camera feed, bound to the composition's lifecycle.
 *
 * CameraX has no Compose-native preview, so this wraps the View-based
 * [PreviewView] in an [AndroidView] — the pattern CameraX's own documentation
 * uses. It was pulled out of CameraScreen so that screen stays about permissions
 * and state rather than about camera plumbing.
 *
 * [imageCapture] is created and remembered by the caller, because the shutter
 * button needs the same instance this preview is bound to.
 */
@Composable
fun CameraPreview(
    imageCapture: ImageCapture,
    modifier: Modifier = Modifier,
) {
    val lifecycleOwner = LocalLifecycleOwner.current

    AndroidView(
        modifier = modifier,
        factory = { ctx ->
            val previewView = PreviewView(ctx).apply {
                // FILL_CENTER matches what the classifier does with the photo: it
                // centre-crops to a square, so the preview should also show a
                // filled, centred frame rather than letterboxing. What the user
                // frames is then what the model sees.
                scaleType = PreviewView.ScaleType.FILL_CENTER
            }
            val cameraProviderFuture = ProcessCameraProvider.getInstance(ctx)
            cameraProviderFuture.addListener(
                {
                    val cameraProvider = cameraProviderFuture.get()
                    val preview = Preview.Builder().build().also {
                        it.surfaceProvider = previewView.surfaceProvider
                    }
                    cameraProvider.unbindAll()
                    cameraProvider.bindToLifecycle(
                        lifecycleOwner,
                        CameraSelector.DEFAULT_BACK_CAMERA,
                        preview,
                        imageCapture,
                    )
                },
                // The listener touches the view and the provider, both of which
                // must be on the main thread.
                ContextCompat.getMainExecutor(ctx),
            )
            previewView
        },
    )
}
