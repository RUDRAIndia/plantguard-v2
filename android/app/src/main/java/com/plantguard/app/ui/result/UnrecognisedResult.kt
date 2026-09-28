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
import androidx.compose.material3.OutlinedButton
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
import com.plantguard.app.ui.components.InlineNote
import com.plantguard.app.ui.components.StoredImage
import com.plantguard.app.ui.components.TierBadge
import com.plantguard.app.ui.theme.PlantGuardTheme
import kotlin.math.roundToInt

/**
 * Tier 3 — confidence below `possible_min`.
 *
 * The wording order here is the whole point. The old app said "unclear photo,
 * please retake", which was measurably the wrong explanation: of the 31 test
 * images below this confidence, 28 were sharp and only 3 were blurry. So this
 * leads with the explanation that is usually true — the plant probably is not one
 * of the 38 covered — and mentions retaking only as a secondary possibility.
 *
 * No candidates are shown. Below this confidence the ranking carries no useful
 * information, and listing names here would invite exactly the misreading the
 * tiering exists to prevent. It also carries no KVK line, because it names
 * nothing to confirm.
 */
@Composable
fun UnrecognisedResult(
    state: ResultUiState.Loaded,
    onOpenPhotoTips: () -> Unit,
) {
    val tierColors = PlantGuardTheme.colors.unrecognised
    val metrics = FieldMetrics.getInstance(LocalContext.current)
    val band = metrics.forTier(ConfidenceTier.UNRECOGNISED)

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
                TierBadge(tier = ConfidenceTier.UNRECOGNISED)
                Spacer(Modifier.height(14.dp))
                Text(
                    text = stringResource(R.string.result_unrecognised_heading),
                    style = MaterialTheme.typography.headlineSmall,
                )
                Spacer(Modifier.height(8.dp))
                Text(
                    text = stringResource(R.string.result_unrecognised_body),
                    style = MaterialTheme.typography.bodyLarge,
                )
            }
        }

        InlineNote(
            text = stringResource(
                R.string.result_unrecognised_reliability,
                (band.shareOfPhotos * 100).roundToInt(),
            ),
        )

        OutlinedButton(
            onClick = onOpenPhotoTips,
            modifier = Modifier.fillMaxWidth(),
        ) {
            Text(stringResource(R.string.action_photo_tips))
        }
    }
}
