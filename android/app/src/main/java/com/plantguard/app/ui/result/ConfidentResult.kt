package com.plantguard.app.ui.result

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.aspectRatio
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.unit.dp
import com.plantguard.app.R
import com.plantguard.app.ml.ConfidenceTier
import com.plantguard.app.ml.FieldMetrics
import com.plantguard.app.ui.components.ConfidenceBar
import com.plantguard.app.ui.components.FootnoteText
import com.plantguard.app.ui.components.InlineNote
import com.plantguard.app.ui.components.SectionCard
import com.plantguard.app.ui.components.StoredImage
import com.plantguard.app.ui.components.TierBadge
import com.plantguard.app.ui.theme.PlantGuardTheme
import kotlin.math.roundToInt

/**
 * Tier 1 — confidence at or above `confident_min`.
 *
 * The only tier that names a single condition as the answer. It is drawn in the
 * confident green, with the diagnosis as the largest thing on the screen, because
 * here the app has earned the right to be read that way: about 82% of field photos
 * scoring this high are correct.
 */
@Composable
fun ConfidentResult(state: ResultUiState.Loaded) {
    val tierColors = PlantGuardTheme.colors.confident
    val metrics = FieldMetrics.getInstance(LocalContext.current)
    val band = metrics.forTier(ConfidenceTier.CONFIDENT)
    val top = state.top

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
                TierBadge(tier = ConfidenceTier.CONFIDENT)
                Spacer(Modifier.height(14.dp))
                Text(
                    text = stringResource(R.string.result_confident_heading),
                    style = MaterialTheme.typography.labelMedium,
                )
                Spacer(Modifier.height(4.dp))
                Text(
                    text = top.displayName,
                    style = MaterialTheme.typography.headlineMedium,
                )
                Spacer(Modifier.height(16.dp))
                ConfidenceBar(
                    confidence = top.candidate.confidence,
                    accent = tierColors.accent,
                    trackColor = tierColors.accent.copy(alpha = 0.18f),
                )
            }
        }

        InlineNote(
            text = stringResource(
                R.string.result_confident_reliability,
                (band.top1Accuracy * 100).roundToInt(),
                (band.shareOfPhotos * 100).roundToInt(),
            ),
        )

        top.info?.let { info ->
            SectionCard(
                title = stringResource(R.string.result_section_symptoms),
                body = info.symptoms,
            )
            SectionCard(
                title = stringResource(R.string.result_section_management),
                body = info.management,
            )
            FootnoteText(text = stringResource(R.string.result_source, info.citation))
        }
    }
}
