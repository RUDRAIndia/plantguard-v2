package com.plantguard.app.ui.components

import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.CheckCircle
import androidx.compose.material.icons.automirrored.filled.HelpOutline
import androidx.compose.material.icons.filled.Lightbulb
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.unit.dp
import com.plantguard.app.R
import com.plantguard.app.ml.ConfidenceTier
import com.plantguard.app.ui.theme.PlantGuardTheme

/**
 * A small pill naming how much the app trusts a result.
 *
 * Icon *and* colour *and* words, all three. Colour alone would be useless to a
 * colour-blind user and to anyone glancing at a phone in bright sun, and this
 * label is the one piece of information on the screen that must not be missed.
 */
@Composable
fun TierBadge(
    tier: ConfidenceTier,
    modifier: Modifier = Modifier,
) {
    val colors = PlantGuardTheme.colors.forTier(tier)
    val icon = when (tier) {
        ConfidenceTier.CONFIDENT -> Icons.Filled.CheckCircle
        ConfidenceTier.POSSIBLE -> Icons.Filled.Lightbulb
        ConfidenceTier.UNRECOGNISED -> Icons.AutoMirrored.Filled.HelpOutline
    }
    val label = stringResource(
        when (tier) {
            ConfidenceTier.CONFIDENT -> R.string.tier_badge_confident
            ConfidenceTier.POSSIBLE -> R.string.tier_badge_possible
            ConfidenceTier.UNRECOGNISED -> R.string.tier_badge_unrecognised
        },
    )

    Surface(
        shape = RoundedCornerShape(50),
        color = colors.container,
        contentColor = colors.onContainer,
        modifier = modifier,
    ) {
        Row(
            modifier = Modifier.padding(horizontal = 12.dp, vertical = 7.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Icon(
                imageVector = icon,
                contentDescription = null,
                tint = colors.accent,
                modifier = Modifier.size(16.dp),
            )
            Spacer(Modifier.width(6.dp))
            Text(text = label, style = MaterialTheme.typography.labelMedium)
        }
    }
}
