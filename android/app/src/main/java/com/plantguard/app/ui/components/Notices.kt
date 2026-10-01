package com.plantguard.app.ui.components

import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Info
import androidx.compose.material.icons.filled.Science
import androidx.compose.material.icons.filled.SupportAgent
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import com.plantguard.app.R
import com.plantguard.app.ml.FieldMetrics
import com.plantguard.app.ui.theme.PlantGuardTheme
import kotlin.math.roundToInt

/**
 * The always-visible honesty notice about how the model performs on real field
 * photographs versus the laboratory images it was first trained on.
 *
 * The numbers are read from assets/field_metrics.json, never written into a
 * string resource: CLAUDE.md rule 5 requires every figure shown to a user to come
 * from a metrics file produced by a real run, so the only place they appear is
 * that asset.
 */
@Composable
fun AccuracyNotice(modifier: Modifier = Modifier) {
    val context = LocalContext.current
    val metrics = FieldMetrics.getInstance(context)

    Card(
        modifier = modifier.fillMaxWidth(),
        shape = MaterialTheme.shapes.medium,
        colors = CardDefaults.cardColors(
            containerColor = PlantGuardTheme.colors.noticeContainer,
            contentColor = PlantGuardTheme.colors.onNoticeContainer,
        ),
        elevation = CardDefaults.cardElevation(defaultElevation = 0.dp),
    ) {
        Row(Modifier.padding(16.dp)) {
            Icon(
                imageVector = Icons.Filled.Science,
                contentDescription = null,
                modifier = Modifier.size(20.dp),
            )
            Spacer(Modifier.width(12.dp))
            Column {
                Text(
                    text = stringResource(R.string.notice_accuracy_title),
                    style = MaterialTheme.typography.titleSmall,
                )
                Spacer(Modifier.height(5.dp))
                Text(
                    text = stringResource(
                        R.string.notice_accuracy_body,
                        (metrics.fieldAccuracy * 100).roundToInt(),
                        (metrics.fieldMacroF1 * 100).roundToInt(),
                        metrics.numImages,
                        ((metrics.confident.top3Accuracy ?: 0f) * 100).roundToInt(),
                    ),
                    style = MaterialTheme.typography.bodySmall,
                )
            }
        }
    }
}

/**
 * The KVK / extension-officer line. Shown on every result that names a class at
 * all — both the confident tier and the hedged one — because that is exactly when
 * a user might act on what they read.
 *
 * Drawn in the app's own neutral surface, not in error red. The old version used
 * `errorContainer`, which made a routine, sensible instruction look like a
 * failure, and buried the source citation inside the same alarm-coloured box.
 */
@Composable
fun KvkAdviceCard(modifier: Modifier = Modifier) {
    Card(
        modifier = modifier.fillMaxWidth(),
        shape = MaterialTheme.shapes.medium,
        colors = CardDefaults.cardColors(
            containerColor = MaterialTheme.colorScheme.primaryContainer,
            contentColor = MaterialTheme.colorScheme.onPrimaryContainer,
        ),
        elevation = CardDefaults.cardElevation(defaultElevation = 0.dp),
    ) {
        Row(Modifier.padding(16.dp)) {
            Icon(
                imageVector = Icons.Filled.SupportAgent,
                contentDescription = null,
                modifier = Modifier.size(20.dp),
            )
            Spacer(Modifier.width(12.dp))
            Text(
                text = stringResource(R.string.notice_kvk),
                style = MaterialTheme.typography.bodyMedium,
            )
        }
    }
}

/** A quiet, centred footnote — source citations and measured-reliability lines. */
@Composable
fun FootnoteText(
    text: String,
    modifier: Modifier = Modifier,
    centered: Boolean = false,
) {
    Text(
        text = text,
        style = MaterialTheme.typography.bodySmall,
        color = MaterialTheme.colorScheme.onSurfaceVariant,
        textAlign = if (centered) TextAlign.Center else null,
        modifier = modifier,
    )
}

/** A small inline note with an info icon, used for per-tier reliability lines. */
@Composable
fun InlineNote(
    text: String,
    modifier: Modifier = Modifier,
) {
    Row(modifier = modifier.fillMaxWidth()) {
        Icon(
            imageVector = Icons.Filled.Info,
            contentDescription = null,
            tint = MaterialTheme.colorScheme.onSurfaceVariant,
            modifier = Modifier.size(16.dp),
        )
        Spacer(Modifier.width(8.dp))
        Text(
            text = text,
            style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
        )
    }
}
