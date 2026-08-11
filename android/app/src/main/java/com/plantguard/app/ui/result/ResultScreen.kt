package com.plantguard.app.ui.result

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.lifecycle.viewmodel.compose.viewModel
import com.plantguard.app.util.ClassNameFormatter

/**
 * The out-of-distribution rejection path (Unclear) matters more than the
 * happy path (Disease) — a farmer must never be shown a confident diagnosis
 * of a photo of soil or a hand. That path never renders a disease name at
 * all, and always looks visually distinct from a real result.
 */
@Composable
fun ResultScreen(
    onRetakePhoto: () -> Unit,
    viewModel: ResultViewModel = viewModel(),
) {
    val uiState by viewModel.uiState.collectAsStateWithLifecycle()

    when (val state = uiState) {
        is ResultUiState.Loading -> Box(Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
            CircularProgressIndicator()
        }

        is ResultUiState.Unclear -> UnclearContent(
            latencyMs = state.entry.inferenceLatencyMs,
            onRetakePhoto = onRetakePhoto,
        )

        is ResultUiState.Disease -> DiseaseContent(state = state, onRetakePhoto = onRetakePhoto)
    }
}

@Composable
private fun UnclearContent(latencyMs: Long, onRetakePhoto: () -> Unit) {
    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(24.dp)
            .verticalScroll(rememberScrollState()),
        verticalArrangement = Arrangement.Center,
    ) {
        Text("Unclear photo", style = MaterialTheme.typography.headlineMedium)
        Text(
            "We can't tell what's in this photo with enough confidence to name a disease — please retake it.",
            modifier = Modifier.padding(top = 8.dp),
        )
        Card(modifier = Modifier.padding(top = 16.dp).fillMaxWidth()) {
            Column(Modifier.padding(16.dp)) {
                Text("For a better photo:", style = MaterialTheme.typography.titleSmall)
                BulletLine("Fill the frame with a single leaf")
                BulletLine("Use a plain background if possible")
                BulletLine("Make sure the leaf is in focus and well lit")
                BulletLine("Avoid strong shadows and blur")
            }
        }
        Text(
            "Inference time: ${latencyMs} ms",
            style = MaterialTheme.typography.labelSmall,
            modifier = Modifier.padding(top = 16.dp),
        )
        Button(onClick = onRetakePhoto, modifier = Modifier.padding(top = 24.dp)) {
            Text("Take another photo")
        }
    }
}

@Composable
private fun DiseaseContent(state: ResultUiState.Disease, onRetakePhoto: () -> Unit) {
    val displayName = state.diseaseInfo?.displayName ?: ClassNameFormatter.humanize(state.className)
    val confidencePercent = (state.entry.confidence * 100).toInt()

    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(24.dp)
            .verticalScroll(rememberScrollState()),
    ) {
        Text(displayName, style = MaterialTheme.typography.headlineMedium)
        Text(
            "Confidence: $confidencePercent% · Inference time: ${state.entry.inferenceLatencyMs} ms",
            style = MaterialTheme.typography.labelMedium,
            modifier = Modifier.padding(top = 4.dp),
        )

        InfoCard(title = "Symptoms", body = state.diseaseInfo?.symptoms ?: NOT_YET_ADDED_TEXT)
        InfoCard(title = "General management", body = state.diseaseInfo?.management ?: NOT_YET_ADDED_TEXT)

        Card(
            modifier = Modifier.padding(top = 16.dp).fillMaxWidth(),
            colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.errorContainer),
        ) {
            Column(Modifier.padding(16.dp)) {
                Text(
                    "Always confirm with your local KVK (Krishi Vigyan Kendra) or " +
                        "agricultural extension officer before acting on this result.",
                    style = MaterialTheme.typography.titleSmall,
                )
                state.diseaseInfo?.citation?.let { citation ->
                    Text(citation, style = MaterialTheme.typography.bodySmall, modifier = Modifier.padding(top = 8.dp))
                }
            }
        }

        Button(onClick = onRetakePhoto, modifier = Modifier.padding(top = 24.dp)) {
            Text("Take another photo")
        }
    }
}

@Composable
private fun InfoCard(title: String, body: String) {
    Card(modifier = Modifier.padding(top = 16.dp).fillMaxWidth()) {
        Column(Modifier.padding(16.dp)) {
            Text(title, style = MaterialTheme.typography.titleSmall)
            Text(body, modifier = Modifier.padding(top = 4.dp))
        }
    }
}

@Composable
private fun BulletLine(text: String) {
    Text("• $text", modifier = Modifier.padding(top = 4.dp))
}

private const val NOT_YET_ADDED_TEXT = "Management information for this disease has not been added yet."
