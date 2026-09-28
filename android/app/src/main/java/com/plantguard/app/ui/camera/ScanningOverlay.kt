package com.plantguard.app.ui.camera

import androidx.compose.animation.core.RepeatMode
import androidx.compose.animation.core.animateFloat
import androidx.compose.animation.core.infiniteRepeatable
import androidx.compose.animation.core.rememberInfiniteTransition
import androidx.compose.animation.core.tween
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import com.plantguard.app.R

/**
 * Shown over the preview while inference runs: a soft green band sweeping down
 * the frame, with a caption.
 *
 * This replaced a bare spinner. Inference takes a noticeable moment on a
 * mid-range phone, and a sweep that visibly crosses the leaf communicates
 * "something is reading *this photo*" in a way a spinner does not.
 *
 * It also swallows touches. The previous overlay was a plain scrim that did not
 * intercept anything, so the shutter stayed live underneath and an impatient
 * second tap started a second inference over the first.
 */
@Composable
fun ScanningOverlay(modifier: Modifier = Modifier) {
    val transition = rememberInfiniteTransition(label = "scan")
    val sweep by transition.animateFloat(
        initialValue = 0f,
        targetValue = 1f,
        animationSpec = infiniteRepeatable(
            animation = tween(durationMillis = 1400),
            // Reverse rather than Restart: the band travels back up instead of
            // snapping to the top, which avoids a hard visual jump on each cycle.
            repeatMode = RepeatMode.Reverse,
        ),
        label = "sweep",
    )

    Box(
        modifier = modifier
            .fillMaxSize()
            .background(Color.Black.copy(alpha = 0.55f))
            // Consumes every touch, so controls underneath cannot be pressed
            // while a photo is being analysed.
            .pointerInput(Unit) { },
        contentAlignment = Alignment.Center,
    ) {
        Canvas(modifier = Modifier.fillMaxSize()) {
            val bandHeight = size.height * 0.16f
            // Travels across the full height plus one band, so it enters and
            // leaves the frame cleanly rather than appearing mid-screen.
            val centreY = (size.height + bandHeight) * sweep - bandHeight / 2f
            drawRect(
                brush = Brush.verticalGradient(
                    colors = listOf(
                        Color.Transparent,
                        Color(0xFF7CC47F).copy(alpha = 0.55f),
                        Color.Transparent,
                    ),
                    startY = centreY - bandHeight / 2f,
                    endY = centreY + bandHeight / 2f,
                ),
                topLeft = Offset(0f, centreY - bandHeight / 2f),
                size = androidx.compose.ui.geometry.Size(size.width, bandHeight),
            )
        }

        Column(
            modifier = Modifier
                .fillMaxWidth()
                .padding(32.dp),
            horizontalAlignment = Alignment.CenterHorizontally,
        ) {
            Spacer(Modifier.height(8.dp))
            Text(
                text = stringResource(R.string.camera_analyzing),
                style = MaterialTheme.typography.titleMedium,
                color = Color.White,
                textAlign = TextAlign.Center,
            )
        }
    }
}
