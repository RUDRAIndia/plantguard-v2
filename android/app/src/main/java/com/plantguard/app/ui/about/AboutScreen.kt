package com.plantguard.app.ui.about

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.unit.dp
import com.plantguard.app.R
import com.plantguard.app.ml.ConfidenceTier
import com.plantguard.app.ml.FieldMetrics
import com.plantguard.app.ui.components.PlantGuardTopBar
import com.plantguard.app.ui.components.SectionCard
import kotlin.math.roundToInt

/**
 * States plainly what the app does, what it covers, and where it is weak.
 *
 * The accuracy figures here come from [FieldMetrics], the same asset the
 * always-visible result-screen notice reads — one source, quoted in two places,
 * rather than a second hand-written copy that could drift from the first.
 */
@Composable
fun AboutScreen(onNavigateUp: () -> Unit) {
    val metrics = FieldMetrics.getInstance(LocalContext.current)
    val confident = metrics.forTier(ConfidenceTier.CONFIDENT)
    val possible = metrics.forTier(ConfidenceTier.POSSIBLE)

    Scaffold(
        topBar = {
            PlantGuardTopBar(
                title = stringResource(R.string.about_title),
                onNavigateUp = onNavigateUp,
            )
        },
    ) { innerPadding ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(innerPadding)
                .verticalScroll(rememberScrollState())
                .padding(horizontal = 20.dp),
            verticalArrangement = Arrangement.spacedBy(14.dp),
        ) {
            Spacer(Modifier.height(4.dp))

            SectionCard(
                title = stringResource(R.string.about_what_title),
                body = stringResource(R.string.about_what_body),
            )
            SectionCard(
                title = stringResource(R.string.about_coverage_title),
                body = stringResource(R.string.about_coverage_body),
            )
            SectionCard(
                title = stringResource(R.string.about_offline_title),
                body = stringResource(R.string.about_offline_body),
            )
            SectionCard(
                title = stringResource(R.string.about_limits_title),
                body = stringResource(
                    R.string.about_limits_body,
                    (metrics.fieldAccuracy * 100).roundToInt(),
                    (metrics.fieldMacroF1 * 100).roundToInt(),
                    metrics.numImages,
                ),
            )
            SectionCard(
                title = stringResource(R.string.about_tiers_title),
                body = stringResource(
                    R.string.about_tiers_body,
                    (confident.top1Accuracy * 100).roundToInt(),
                    (possible.top1Accuracy * 100).roundToInt(),
                    ((possible.top3Accuracy ?: 0f) * 100).roundToInt(),
                ),
            )
            SectionCard(
                title = stringResource(R.string.about_advice_title),
                body = stringResource(R.string.about_advice_body),
            )

            Text(
                text = stringResource(R.string.about_metrics_source, metrics.evaluatedOn),
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
                modifier = Modifier.fillMaxWidth(),
            )
            Spacer(Modifier.height(16.dp))
        }
    }
}
