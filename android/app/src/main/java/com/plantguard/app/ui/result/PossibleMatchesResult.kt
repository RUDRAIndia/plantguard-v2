package com.plantguard.app.ui.result

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.aspectRatio
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.KeyboardArrowRight
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.unit.dp
import com.plantguard.app.R
import com.plantguard.app.ml.ConfidenceTier
import com.plantguard.app.ml.FieldMetrics
import com.plantguard.app.ui.components.ConfidenceBar
import com.plantguard.app.ui.components.InlineNote
import com.plantguard.app.ui.components.StoredImage
import com.plantguard.app.ui.components.TierBadge
import com.plantguard.app.ui.theme.PlantGuardTheme
import kotlin.math.roundToInt

/**
 * Tier 2 — confidence between `possible_min` and `confident_min`.
 *
 * This is the tier the redesign exists for. It is 41% of real field photos, and
 * the old app showed every one of them as "unclear photo, please retake" — which
 * is why clear leaf photos appeared to be rejected.
 *
 * Two wording rules are deliberate here, and both matter more than they look:
 *  - The heading is "Possible matches — not confident", never "most likely". In
 *    this band the top candidate alone is right only about 41% of the time, so
 *    presenting it as the answer would be actively misleading.
 *  - The list is worth showing anyway, because the correct answer is among the
 *    three about 75% of the time. That figure is stated on screen, so the user
 *    understands why they are being shown three things instead of one.
 *
 * The candidates are numbered rather than ranked with any visual hierarchy, and
 * all three rows look alike — the first is not styled as a winner.
 */
@Composable
fun PossibleMatchesResult(
    state: ResultUiState.Loaded,
    onOpenDisease: (Int) -> Unit,
) {
    val tierColors = PlantGuardTheme.colors.possible
    val metrics = FieldMetrics.getInstance(LocalContext.current)
    val band = metrics.forTier(ConfidenceTier.POSSIBLE)

    Column(verticalArrangement = Arrangement.spacedBy(16.dp)) {
        StoredImage(
            path = state.entry.imagePath,
            contentDescription = stringResource(R.string.result_photo_description),
            modifier = Modifier
                .fillMaxWidth()
                .aspectRatio(1f)
                .clip(MaterialTheme.shapes.large),
        )

        Card(
            modifier = Modifier.fillMaxWidth(),
            shape = MaterialTheme.shapes.large,
            colors = CardDefaults.cardColors(
                containerColor = tierColors.container,
                contentColor = tierColors.onContainer,
            ),
            elevation = CardDefaults.cardElevation(defaultElevation = 0.dp),
        ) {
            Column(Modifier.padding(20.dp)) {
                TierBadge(tier = ConfidenceTier.POSSIBLE)
                Spacer(Modifier.height(14.dp))
                Text(
                    text = stringResource(R.string.result_possible_heading),
                    style = MaterialTheme.typography.headlineSmall,
                )
                Spacer(Modifier.height(8.dp))
                Text(
                    text = stringResource(R.string.result_possible_explainer),
                    style = MaterialTheme.typography.bodyMedium,
                )
            }
        }

        InlineNote(
            text = stringResource(
                R.string.result_possible_reliability,
                ((band.top3Accuracy ?: 0f) * 100).roundToInt(),
                (band.top1Accuracy * 100).roundToInt(),
                (band.shareOfPhotos * 100).roundToInt(),
            ),
        )

        Column(verticalArrangement = Arrangement.spacedBy(10.dp)) {
            state.candidates.forEachIndexed { index, candidate ->
                CandidateRow(
                    position = index + 1,
                    name = candidate.displayName,
                    confidence = candidate.candidate.confidence,
                    accent = tierColors.accent,
                    onClick = { onOpenDisease(candidate.candidate.classIndex) },
                )
            }
        }

        Text(
            text = stringResource(R.string.result_possible_tap_hint),
            style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
        )
    }
}

/**
 * One candidate: its position, name, confidence meter, and a chevron into the
 * full detail screen.
 *
 * Every row is styled identically on purpose. Giving the first one a heavier
 * treatment would undo the whole point of this tier.
 */
@Composable
private fun CandidateRow(
    position: Int,
    name: String,
    confidence: Float,
    accent: androidx.compose.ui.graphics.Color,
    onClick: () -> Unit,
) {
    Card(
        onClick = onClick,
        modifier = Modifier.fillMaxWidth(),
        shape = MaterialTheme.shapes.medium,
        colors = CardDefaults.cardColors(
            containerColor = MaterialTheme.colorScheme.surfaceContainer,
            contentColor = MaterialTheme.colorScheme.onSurface,
        ),
        elevation = CardDefaults.cardElevation(defaultElevation = 1.dp),
    ) {
        Row(
            modifier = Modifier.padding(16.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Surface(
                shape = CircleShape,
                color = accent.copy(alpha = 0.16f),
                contentColor = MaterialTheme.colorScheme.onSurface,
                modifier = Modifier.size(30.dp),
            ) {
                Row(
                    horizontalArrangement = Arrangement.Center,
                    verticalAlignment = Alignment.CenterVertically,
                ) {
                    Text(
                        text = "$position",
                        style = MaterialTheme.typography.labelMedium,
                    )
                }
            }
            Spacer(Modifier.width(14.dp))
            Column(Modifier.weight(1f)) {
                Text(text = name, style = MaterialTheme.typography.titleMedium)
                Spacer(Modifier.height(10.dp))
                ConfidenceBar(
                    confidence = confidence,
                    accent = accent,
                    barHeight = 8,
                )
            }
            Spacer(Modifier.width(8.dp))
            Icon(
                imageVector = Icons.AutoMirrored.Filled.KeyboardArrowRight,
                contentDescription = null,
                tint = MaterialTheme.colorScheme.outline,
            )
        }
    }
}
