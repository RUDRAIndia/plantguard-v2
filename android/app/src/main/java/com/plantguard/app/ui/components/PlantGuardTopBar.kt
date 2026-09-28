package com.plantguard.app.ui.components

import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.material3.TopAppBarDefaults
import androidx.compose.runtime.Composable
import androidx.compose.ui.res.stringResource
import com.plantguard.app.R

/**
 * The single top bar every screen but Home and Camera uses.
 *
 * Previously no screen had one, which meant History and Result had no way back
 * except the system gesture. A visible back affordance matters more than usual
 * here because the user arrives at a result from two different places (camera
 * and history) and needs to see that going back is possible.
 *
 * [onNavigateUp] null means this screen is a root and shows no back button.
 *
 * AutoMirrored ArrowBack, so the arrow flips for right-to-left locales — the
 * manifest already declares supportsRtl.
 */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun PlantGuardTopBar(
    title: String,
    onNavigateUp: (() -> Unit)? = null,
) {
    TopAppBar(
        title = { Text(text = title, style = MaterialTheme.typography.titleLarge) },
        navigationIcon = {
            if (onNavigateUp != null) {
                IconButton(onClick = onNavigateUp) {
                    Icon(
                        imageVector = Icons.AutoMirrored.Filled.ArrowBack,
                        contentDescription = stringResource(R.string.action_back),
                    )
                }
            }
        },
        colors = TopAppBarDefaults.topAppBarColors(
            containerColor = MaterialTheme.colorScheme.surface,
            titleContentColor = MaterialTheme.colorScheme.onSurface,
            navigationIconContentColor = MaterialTheme.colorScheme.onSurface,
        ),
    )
}
