package com.plantguard.app.ui.result

import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.core.tween
import androidx.compose.animation.fadeIn
import androidx.compose.animation.slideInVertically
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.lifecycle.viewmodel.compose.viewModel
import com.plantguard.app.R
import com.plantguard.app.ml.ConfidenceTier
import com.plantguard.app.ui.components.AccuracyNotice
import com.plantguard.app.ui.components.FootnoteText
import com.plantguard.app.ui.components.KvkAdviceCard
import com.plantguard.app.ui.components.PlantGuardTopBar

/**
 * Host for the three result tiers.
 *
 * The screen is split by tier into three sibling files rather than one long
 * `when`, because the three are genuinely different presentations — a named
 * diagnosis, a comparison of three candidates, and a "not in our list"
 * explanation — and the point of the redesign is that they must not look alike.
 *
 * What is shared lives here: the top bar, the photo, the always-visible accuracy
 * notice, and (for the two tiers that name a class) the KVK line.
 */
@Composable
fun ResultScreen(
    onNavigateUp: () -> Unit,
    onTakeAnotherPhoto: () -> Unit,
    onOpenDisease: (Int) -> Unit,
    onOpenPhotoTips: () -> Unit,
    viewModel: ResultViewModel = viewModel(),
) {
    val uiState by viewModel.uiState.collectAsStateWithLifecycle()

    Scaffold(
        topBar = {
            PlantGuardTopBar(
                title = stringResource(R.string.result_title),
                onNavigateUp = onNavigateUp,
            )
        },
    ) { innerPadding ->
        Box(
            modifier = Modifier
                .fillMaxSize()
                .padding(innerPadding),
        ) {
            when (val state = uiState) {
                is ResultUiState.Loading -> Box(
                    modifier = Modifier.fillMaxSize(),
                    contentAlignment = Alignment.Center,
                ) {
                    CircularProgressIndicator()
                }

                is ResultUiState.Error -> ResultError(
                    message = state.message,
                    onTakeAnotherPhoto = onTakeAnotherPhoto,
                )

                is ResultUiState.Loaded -> LoadedResult(
                    state = state,
                    onTakeAnotherPhoto = onTakeAnotherPhoto,
                    onOpenDisease = onOpenDisease,
                    onOpenPhotoTips = onOpenPhotoTips,
                )
            }
        }
    }
}

@Composable
private fun LoadedResult(
    state: ResultUiState.Loaded,
    onTakeAnotherPhoto: () -> Unit,
    onOpenDisease: (Int) -> Unit,
    onOpenPhotoTips: () -> Unit,
) {
    // Eases the result in on first composition rather than having it appear
    // fully formed, which makes the screen feel like it settled rather than
    // snapped.
    var visible by remember { mutableStateOf(false) }
    LaunchedEffect(Unit) { visible = true }

    AnimatedVisibility(
        visible = visible,
        enter = fadeIn(tween(350)) + slideInVertically(tween(350)) { it / 12 },
    ) {
        Column(
            modifier = Modifier
                .fillMaxSize()
                .verticalScroll(rememberScrollState())
                .padding(horizontal = 20.dp),
            verticalArrangement = Arrangement.spacedBy(16.dp),
        ) {
            Spacer(Modifier.height(4.dp))

            when (state.tier) {
                ConfidenceTier.CONFIDENT -> ConfidentResult(state = state)
                ConfidenceTier.POSSIBLE -> PossibleMatchesResult(
                    state = state,
                    onOpenDisease = onOpenDisease,
                )
                ConfidenceTier.UNRECOGNISED -> UnrecognisedResult(
                    state = state,
                    onOpenPhotoTips = onOpenPhotoTips,
                )
            }

            // Shown on every result that names a class at all — that is, both
            // tiers a user might act on. The unrecognised tier names nothing, so
            // there is nothing there to confirm with anyone.
            if (state.tier != ConfidenceTier.UNRECOGNISED) {
                KvkAdviceCard()
            }

            AccuracyNotice()

            FootnoteText(
                text = stringResource(R.string.result_latency, state.entry.inferenceLatencyMs),
                centered = true,
                modifier = Modifier.fillMaxWidth(),
            )

            Button(
                onClick = onTakeAnotherPhoto,
                modifier = Modifier.fillMaxWidth(),
            ) {
                Text(stringResource(R.string.action_take_another))
            }
            Spacer(Modifier.height(16.dp))
        }
    }
}

/** Shown when the row behind this screen could not be loaded. */
@Composable
private fun ResultError(
    message: String,
    onTakeAnotherPhoto: () -> Unit,
) {
    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(24.dp),
        verticalArrangement = Arrangement.Center,
        horizontalAlignment = Alignment.CenterHorizontally,
    ) {
        Text(
            text = message,
            style = MaterialTheme.typography.bodyLarge,
            textAlign = TextAlign.Center,
        )
        Spacer(Modifier.height(24.dp))
        Button(onClick = onTakeAnotherPhoto) {
            Text(stringResource(R.string.action_take_another))
        }
    }
}
