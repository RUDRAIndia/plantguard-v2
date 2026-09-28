package com.plantguard.app.ui.home

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.CameraAlt
import androidx.compose.material.icons.filled.History
import androidx.compose.material.icons.filled.Info
import androidx.compose.material.icons.filled.PhotoCamera
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import com.plantguard.app.R
import com.plantguard.app.ui.components.AccuracyNotice
import com.plantguard.app.ui.components.ActionCard
import com.plantguard.app.ui.theme.PlantGuardTheme

/**
 * The screen the app opens on.
 *
 * It exists because the app used to launch straight into the camera, which made
 * three of its four capabilities invisible — most importantly the photo guidance,
 * which measurably changes what tier a result lands in. The four cards are large
 * and evenly weighted rather than one primary button with small links, because
 * "read this before you shoot" is genuinely as useful here as "shoot".
 */
@Composable
fun HomeScreen(
    onIdentify: () -> Unit,
    onHistory: () -> Unit,
    onAbout: () -> Unit,
    onPhotoTips: () -> Unit,
) {
    Scaffold { innerPadding ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(innerPadding)
                .verticalScroll(rememberScrollState())
                .padding(horizontal = 20.dp),
        ) {
            Spacer(Modifier.height(32.dp))
            Brandmark()
            Spacer(Modifier.height(32.dp))

            Column(verticalArrangement = Arrangement.spacedBy(12.dp)) {
                ActionCard(
                    title = stringResource(R.string.home_card_identify_title),
                    subtitle = stringResource(R.string.home_card_identify_subtitle),
                    icon = Icons.Filled.CameraAlt,
                    iconTint = PlantGuardTheme.colors.confident.onContainer,
                    iconBackground = PlantGuardTheme.colors.confident.container,
                    onClick = onIdentify,
                )
                ActionCard(
                    title = stringResource(R.string.home_card_history_title),
                    subtitle = stringResource(R.string.home_card_history_subtitle),
                    icon = Icons.Filled.History,
                    iconTint = MaterialTheme.colorScheme.onSecondaryContainer,
                    iconBackground = MaterialTheme.colorScheme.secondaryContainer,
                    onClick = onHistory,
                )
                ActionCard(
                    title = stringResource(R.string.home_card_tips_title),
                    subtitle = stringResource(R.string.home_card_tips_subtitle),
                    icon = Icons.Filled.PhotoCamera,
                    iconTint = PlantGuardTheme.colors.possible.onContainer,
                    iconBackground = PlantGuardTheme.colors.possible.container,
                    onClick = onPhotoTips,
                )
                ActionCard(
                    title = stringResource(R.string.home_card_about_title),
                    subtitle = stringResource(R.string.home_card_about_subtitle),
                    icon = Icons.Filled.Info,
                    iconTint = PlantGuardTheme.colors.unrecognised.onContainer,
                    iconBackground = PlantGuardTheme.colors.unrecognised.container,
                    onClick = onAbout,
                )
            }

            Spacer(Modifier.height(24.dp))
            // Present on the home screen as well as on every result, so the
            // app's main limitation is visible before the first photo, not only
            // after a diagnosis has already been read.
            AccuracyNotice()
            Spacer(Modifier.height(32.dp))
        }
    }
}

/** The leaf mark, app name and one-line description. */
@Composable
private fun Brandmark() {
    Column(
        modifier = Modifier.fillMaxWidth(),
        horizontalAlignment = Alignment.CenterHorizontally,
    ) {
        Surface(
            shape = CircleShape,
            color = MaterialTheme.colorScheme.primaryContainer,
            modifier = Modifier.size(84.dp),
        ) {
            Box(contentAlignment = Alignment.Center) {
                Icon(
                    painter = painterResource(R.drawable.leaf_mark),
                    contentDescription = null,
                    tint = MaterialTheme.colorScheme.primary,
                    modifier = Modifier.size(44.dp),
                )
            }
        }
        Spacer(Modifier.height(16.dp))
        Text(
            text = stringResource(R.string.app_name),
            style = MaterialTheme.typography.displaySmall,
            color = MaterialTheme.colorScheme.onBackground,
        )
        Spacer(Modifier.height(6.dp))
        Text(
            text = stringResource(R.string.home_tagline),
            style = MaterialTheme.typography.bodyMedium,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
            textAlign = TextAlign.Center,
        )
    }
}
