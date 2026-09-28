package com.plantguard.app.ui.camera

import androidx.compose.foundation.Canvas
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.unit.dp
import kotlin.math.min

/**
 * Four corner brackets marking roughly where the leaf should sit.
 *
 * This is not decoration. The classifier centre-crops every photo to a square
 * before resizing, so anything outside the centre square is discarded — a leaf
 * framed at the edge is literally cut out of what the model sees. The guide shows
 * that square.
 *
 * Corner brackets rather than a full rectangle: a closed box over a live preview
 * reads as a viewfinder the user must fill exactly, whereas brackets suggest the
 * region without hiding the leaf behind lines.
 */
@Composable
fun FramingGuide(
    modifier: Modifier = Modifier,
    color: Color = Color.White.copy(alpha = 0.85f),
) {
    Box(modifier = modifier) {
        Canvas(modifier = Modifier.fillMaxSize()) {
            // The guide is the largest centred square that leaves a comfortable
            // margin, matching the centre-crop the classifier performs.
            val side = min(size.width, size.height) * 0.78f
            val left = (size.width - side) / 2f
            val top = (size.height - side) / 2f
            val cornerLength = side * 0.16f
            val strokeWidth = 3.dp.toPx()
            val stroke = Stroke(width = strokeWidth, cap = StrokeCap.Round)

            fun line(from: Offset, to: Offset) {
                drawLine(
                    color = color,
                    start = from,
                    end = to,
                    strokeWidth = stroke.width,
                    cap = stroke.cap,
                )
            }

            val right = left + side
            val bottom = top + side

            // Top-left
            line(Offset(left, top), Offset(left + cornerLength, top))
            line(Offset(left, top), Offset(left, top + cornerLength))
            // Top-right
            line(Offset(right, top), Offset(right - cornerLength, top))
            line(Offset(right, top), Offset(right, top + cornerLength))
            // Bottom-left
            line(Offset(left, bottom), Offset(left + cornerLength, bottom))
            line(Offset(left, bottom), Offset(left, bottom - cornerLength))
            // Bottom-right
            line(Offset(right, bottom), Offset(right - cornerLength, bottom))
            line(Offset(right, bottom), Offset(right, bottom - cornerLength))

            // A faint centre tick, so "put the lesion in the middle" has a target.
            val centre = Offset(size.width / 2f, size.height / 2f)
            val tick = 10.dp.toPx()
            val faint = color.copy(alpha = 0.35f)
            drawLine(faint, Offset(centre.x - tick, centre.y), Offset(centre.x + tick, centre.y), strokeWidth / 1.5f, StrokeCap.Round)
            drawLine(faint, Offset(centre.x, centre.y - tick), Offset(centre.x, centre.y + tick), strokeWidth / 1.5f, StrokeCap.Round)
        }
    }
}
