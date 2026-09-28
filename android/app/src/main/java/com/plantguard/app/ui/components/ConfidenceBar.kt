package com.plantguard.app.ui.components

import androidx.compose.animation.core.animateFloatAsState
import androidx.compose.animation.core.tween
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.remember
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.layout.layout
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.unit.dp
import kotlin.math.roundToInt

/**
 * A horizontal meter for one confidence value, with the percentage beside it.
 *
 * The fill animates from empty on first composition, which reads as the result
 * settling into place rather than snapping in.
 *
 * The bar itself is marked as one semantic node carrying the percentage as text,
 * so a screen reader announces "61 percent" instead of describing a decorative
 * rectangle.
 */
@Composable
fun ConfidenceBar(
    confidence: Float,
    accent: Color,
    modifier: Modifier = Modifier,
    trackColor: Color = MaterialTheme.colorScheme.surfaceContainerHigh,
    barHeight: Int = 10,
) {
    val percent = remember(confidence) { (confidence * 100).roundToInt() }
    val animatedFraction by animateFloatAsState(
        targetValue = confidence.coerceIn(0f, 1f),
        animationSpec = tween(durationMillis = 650),
        label = "confidenceFill",
    )

    Row(
        modifier = modifier.semantics { contentDescription = "$percent percent confidence" },
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Box(
            modifier = Modifier
                .weight(1f)
                .height(barHeight.dp)
                .background(color = trackColor, shape = RoundedCornerShape(50)),
        ) {
            Box(
                modifier = Modifier
                    // A plain fillMaxWidth(fraction) would round a tiny value to
                    // nothing; measuring explicitly keeps a sliver visible so a
                    // very low confidence still reads as "almost none", not "none".
                    .layout { measurable, constraints ->
                        val target = (constraints.maxWidth * animatedFraction).roundToInt()
                        val width = if (animatedFraction > 0f) target.coerceAtLeast(barHeight) else 0
                        val placeable = measurable.measure(constraints.copy(minWidth = width, maxWidth = width))
                        layout(placeable.width, placeable.height) { placeable.placeRelative(0, 0) }
                    }
                    .fillMaxHeight()
                    .background(color = accent, shape = RoundedCornerShape(50)),
            )
        }
        Spacer(Modifier.width(12.dp))
        Text(
            text = "$percent%",
            style = MaterialTheme.typography.labelLarge,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
        )
    }
}

/** A full-width meter with a caption above it. */
@Composable
fun LabelledConfidenceBar(
    label: String,
    confidence: Float,
    accent: Color,
    modifier: Modifier = Modifier,
) {
    androidx.compose.foundation.layout.Column(modifier = modifier.fillMaxWidth()) {
        Text(
            text = label,
            style = MaterialTheme.typography.labelMedium,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
        )
        Spacer(Modifier.height(8.dp))
        ConfidenceBar(confidence = confidence, accent = accent)
    }
}
