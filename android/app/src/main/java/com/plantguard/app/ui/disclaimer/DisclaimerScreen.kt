package com.plantguard.app.ui.disclaimer

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp

/**
 * Shown exactly once, on first launch (PlantGuardApp gates this via
 * DisclaimerPrefs). This is a research/student-project tool, not a
 * medical- or agronomy-grade diagnostic device — the text below says so
 * plainly, and the same "confirm with your local KVK" line reappears on
 * every Result screen too, not just here.
 */
@Composable
fun DisclaimerScreen(onContinue: () -> Unit) {
    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(24.dp)
            .verticalScroll(rememberScrollState()),
        verticalArrangement = Arrangement.Center,
    ) {
        Text(
            text = "Before you start",
            style = MaterialTheme.typography.headlineSmall,
        )
        Spacer(modifier = Modifier.height(16.dp))
        Text(
            "PlantGuard is a student research project, not a certified " +
                "diagnostic tool. Its predictions can be wrong — including " +
                "confidently wrong.",
        )
        Spacer(modifier = Modifier.height(16.dp))
        Text(
            "Never make a crop-protection decision based on this app alone. " +
                "Always confirm any diagnosis with your local KVK " +
                "(Krishi Vigyan Kendra) or agricultural extension officer " +
                "before taking action.",
        )
        Spacer(modifier = Modifier.height(16.dp))
        Text(
            "This app works fully offline. Your photos and results stay on " +
                "this device and are never uploaded anywhere.",
            style = MaterialTheme.typography.bodySmall,
        )
        Spacer(modifier = Modifier.height(24.dp))
        Button(
            onClick = onContinue,
            modifier = Modifier.align(Alignment.CenterHorizontally),
        ) {
            Text("I understand, continue")
        }
    }
}
