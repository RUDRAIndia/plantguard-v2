package com.plantguard.app.ui.camera

import androidx.camera.core.ImageCapture
import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.slideInVertically
import androidx.compose.animation.slideOutVertically
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.systemBarsPadding
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material.icons.filled.Close
import androidx.compose.material.icons.automirrored.filled.HelpOutline
import androidx.compose.material.icons.filled.PhotoLibrary
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.unit.dp
import com.plantguard.app.R

/** The live-preview state: camera feed, framing guide and controls. */
@Composable
fun CameraCaptureContent(
    imageCapture: ImageCapture,
    isProcessing: Boolean,
    onShutter: () -> Unit,
    onPickGallery: () -> Unit,
    onNavigateUp: () -> Unit,
    onOpenPhotoTips: () -> Unit,
) {
    Box(modifier = Modifier.fillMaxSize().background(Color.Black)) {
        CameraPreview(imageCapture = imageCapture, modifier = Modifier.fillMaxSize())
        FramingGuide(modifier = Modifier.fillMaxSize())

        // Top row: back out of the camera, and a shortcut to the photo guidance —
        // the moment a user is most likely to want it is while framing a shot.
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .systemBarsPadding()
                .padding(horizontal = 8.dp, vertical = 4.dp),
            horizontalArrangement = Arrangement.SpaceBetween,
        ) {
            ScrimIconButton(
                onClick = onNavigateUp,
                icon = Icons.AutoMirrored.Filled.ArrowBack,
                contentDescription = stringResource(R.string.action_back),
            )
            ScrimIconButton(
                onClick = onOpenPhotoTips,
                icon = Icons.AutoMirrored.Filled.HelpOutline,
                contentDescription = stringResource(R.string.action_photo_tips),
            )
        }

        Column(
            modifier = Modifier
                .align(Alignment.BottomCenter)
                .fillMaxWidth()
                .systemBarsPadding()
                .padding(bottom = 24.dp),
            horizontalAlignment = Alignment.CenterHorizontally,
        ) {
            // The framing instruction sits with the controls rather than over the
            // leaf, so it never covers what the user is trying to photograph.
            AnimatedVisibility(
                visible = !isProcessing,
                enter = slideInVertically { it },
                exit = slideOutVertically { it },
            ) {
                Surface(
                    shape = MaterialTheme.shapes.small,
                    color = Color.Black.copy(alpha = 0.45f),
                ) {
                    Text(
                        text = stringResource(R.string.camera_framing_hint),
                        style = MaterialTheme.typography.labelLarge,
                        color = Color.White,
                        modifier = Modifier.padding(horizontal = 14.dp, vertical = 8.dp),
                    )
                }
            }
            Spacer(Modifier.height(20.dp))
            Row(
                modifier = Modifier.fillMaxWidth().padding(horizontal = 32.dp),
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.SpaceBetween,
            ) {
                ScrimIconButton(
                    onClick = onPickGallery,
                    icon = Icons.Filled.PhotoLibrary,
                    contentDescription = stringResource(R.string.action_pick_from_gallery),
                )
                ShutterButton(onClick = onShutter, enabled = !isProcessing)
                // An empty box the same size as the gallery button, so the shutter
                // stays optically centred without a second control.
                Spacer(Modifier.size(48.dp))
            }
        }
    }
}

/**
 * The shutter: a large white ring with a filled centre, the shape every phone
 * camera uses, so it needs no label to be understood.
 */
@Composable
private fun ShutterButton(onClick: () -> Unit, enabled: Boolean) {
    val label = stringResource(R.string.action_shutter)
    Surface(
        onClick = onClick,
        enabled = enabled,
        shape = CircleShape,
        color = Color.White.copy(alpha = if (enabled) 0.28f else 0.12f),
        // The button is a plain shape with no icon or text, so its accessibility
        // label has to be attached explicitly.
        modifier = Modifier
            .size(78.dp)
            .semantics { contentDescription = label },
    ) {
        Box(contentAlignment = Alignment.Center) {
            Box(
                modifier = Modifier
                    .size(58.dp)
                    .clip(CircleShape)
                    .background(if (enabled) Color.White else Color.White.copy(alpha = 0.5f)),
            )
        }
    }
}

/** A round translucent button, legible over any camera feed. */
@Composable
private fun ScrimIconButton(
    onClick: () -> Unit,
    icon: androidx.compose.ui.graphics.vector.ImageVector,
    contentDescription: String,
) {
    Surface(
        shape = CircleShape,
        color = Color.Black.copy(alpha = 0.35f),
        modifier = Modifier.size(48.dp),
    ) {
        IconButton(onClick = onClick) {
            Icon(imageVector = icon, contentDescription = contentDescription, tint = Color.White)
        }
    }
}

/** A dismissible error banner pinned to the bottom of the screen. */
@Composable
fun CameraErrorBanner(
    message: String?,
    onDismiss: () -> Unit,
    modifier: Modifier = Modifier,
) {
    Box(modifier = modifier) {
        AnimatedVisibility(
            visible = message != null,
            enter = slideInVertically { it },
            exit = slideOutVertically { it },
            modifier = Modifier.align(Alignment.BottomCenter),
        ) {
            Surface(
                color = MaterialTheme.colorScheme.errorContainer,
                contentColor = MaterialTheme.colorScheme.onErrorContainer,
                shape = MaterialTheme.shapes.medium,
                modifier = Modifier
                    .systemBarsPadding()
                    .padding(16.dp),
            ) {
                Row(
                    modifier = Modifier.padding(start = 16.dp, top = 6.dp, bottom = 6.dp, end = 6.dp),
                    verticalAlignment = Alignment.CenterVertically,
                ) {
                    Text(
                        text = message.orEmpty(),
                        style = MaterialTheme.typography.bodyMedium,
                        modifier = Modifier.weight(1f),
                    )
                    IconButton(onClick = onDismiss) {
                        Icon(
                            imageVector = Icons.Filled.Close,
                            contentDescription = stringResource(R.string.action_dismiss),
                        )
                    }
                }
            }
        }
    }
}
