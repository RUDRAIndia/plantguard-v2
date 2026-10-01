package com.plantguard.app.ui.history

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.DeleteSweep
import androidx.compose.material.icons.filled.HistoryToggleOff
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Card
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.TextButton
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.lifecycle.viewmodel.compose.viewModel
import com.plantguard.app.R
import com.plantguard.app.ml.ConfidenceTier
import com.plantguard.app.ui.components.PlantGuardTopBar
import com.plantguard.app.ui.components.StoredImage
import com.plantguard.app.ui.components.TierBadge
import java.text.DateFormat
import java.util.Date
import kotlin.math.roundToInt

/**
 * Every past prediction, newest first, each carrying the same tier badge the
 * result screen uses — so a scan down history shows at a glance which answers
 * were confident and which were hedged or unrecognised, without opening any of
 * them.
 */
@Composable
fun HistoryScreen(
    onOpenResult: (Long) -> Unit,
    onNavigateUp: () -> Unit,
    viewModel: HistoryViewModel = viewModel(),
) {
    val rows by viewModel.rows.collectAsStateWithLifecycle()
    var showClearConfirm by remember { mutableStateOf(false) }

    Scaffold(
        topBar = {
            PlantGuardTopBar(
                title = stringResource(R.string.history_title),
                onNavigateUp = onNavigateUp,
            )
        },
    ) { innerPadding ->
        if (rows.isEmpty()) {
            EmptyHistory(modifier = Modifier.padding(innerPadding))
        } else {
            LazyColumn(
                modifier = Modifier
                    .fillMaxSize()
                    .padding(innerPadding),
                contentPadding = PaddingValues(horizontal = 20.dp, vertical = 12.dp),
                verticalArrangement = Arrangement.spacedBy(10.dp),
            ) {
                items(rows, key = { it.id }) { row ->
                    HistoryRowCard(row = row, onClick = { onOpenResult(row.id) })
                }
                item {
                    Spacer(Modifier.height(8.dp))
                    OutlinedButton(
                        onClick = { showClearConfirm = true },
                        modifier = Modifier.fillMaxWidth(),
                    ) {
                        Icon(
                            imageVector = Icons.Filled.DeleteSweep,
                            contentDescription = null,
                            modifier = Modifier.size(18.dp),
                        )
                        Spacer(Modifier.width(8.dp))
                        Text(stringResource(R.string.history_clear_all))
                    }
                }
            }
        }
    }

    // Deleting every saved scan cannot be undone, so it is always confirmed.
    if (showClearConfirm) {
        AlertDialog(
            onDismissRequest = { showClearConfirm = false },
            icon = {
                Icon(imageVector = Icons.Filled.DeleteSweep, contentDescription = null)
            },
            title = { Text(stringResource(R.string.history_clear_confirm_title)) },
            text = { Text(stringResource(R.string.history_clear_confirm_body, rows.size)) },
            confirmButton = {
                TextButton(onClick = {
                    viewModel.deleteAll()
                    showClearConfirm = false
                }) {
                    Text(stringResource(R.string.history_clear_confirm_action))
                }
            },
            dismissButton = {
                TextButton(onClick = { showClearConfirm = false }) {
                    Text(stringResource(R.string.history_clear_cancel))
                }
            },
        )
    }
}

@Composable
private fun EmptyHistory(modifier: Modifier = Modifier) {
    Column(
        modifier = modifier
            .fillMaxSize()
            .padding(32.dp),
        verticalArrangement = Arrangement.Center,
        horizontalAlignment = Alignment.CenterHorizontally,
    ) {
        Icon(
            imageVector = Icons.Filled.HistoryToggleOff,
            contentDescription = null,
            tint = MaterialTheme.colorScheme.outline,
            modifier = Modifier.size(48.dp),
        )
        Spacer(Modifier.height(16.dp))
        Text(
            text = stringResource(R.string.history_empty_title),
            style = MaterialTheme.typography.titleLarge,
            textAlign = TextAlign.Center,
        )
        Spacer(Modifier.height(6.dp))
        Text(
            text = stringResource(R.string.history_empty_body),
            style = MaterialTheme.typography.bodyMedium,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
            textAlign = TextAlign.Center,
        )
    }
}

@Composable
private fun HistoryRowCard(row: HistoryRow, onClick: () -> Unit) {
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
            modifier = Modifier.padding(12.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            StoredImage(
                path = row.imagePath,
                contentDescription = null,
                maxDimension = 160,
                modifier = Modifier
                    .size(56.dp)
                    .clip(MaterialTheme.shapes.small),
            )
            Spacer(Modifier.width(14.dp))
            Column(Modifier.weight(1f)) {
                TierBadge(tier = row.tier)
                Spacer(Modifier.height(6.dp))
                Text(
                    text = titleFor(row),
                    style = MaterialTheme.typography.titleMedium,
                )
                Spacer(Modifier.height(2.dp))
                Text(
                    text = stringResource(
                        R.string.history_row_meta,
                        (row.confidence * 100).roundToInt(),
                        formatTimestamp(row.timestampMillis),
                    ),
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }
        }
    }
}

@Composable
private fun titleFor(row: HistoryRow): String = when (row.tier) {
    ConfidenceTier.UNRECOGNISED -> stringResource(R.string.history_unrecognised_row)
    ConfidenceTier.POSSIBLE -> stringResource(R.string.history_possible_row, row.candidateCount)
    ConfidenceTier.CONFIDENT -> row.displayName ?: stringResource(R.string.history_unrecognised_row)
}

private fun formatTimestamp(millis: Long): String =
    DateFormat.getDateTimeInstance(DateFormat.SHORT, DateFormat.SHORT).format(Date(millis))
