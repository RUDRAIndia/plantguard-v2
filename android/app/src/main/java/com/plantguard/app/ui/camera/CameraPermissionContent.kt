package com.plantguard.app.ui.camera

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.NoPhotography
import androidx.compose.material.icons.filled.PhotoCamera
import androidx.compose.material3.Button
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import com.plantguard.app.R
import com.plantguard.app.ui.components.PlantGuardTopBar

/**
 * Shown before camera permission has been granted.
 *
 * The rationale explains the offline guarantee, because "why does a plant app
 * want my camera" is a fair question and the honest answer — the photo never
 * leaves the phone, the app cannot even open a network socket — is reassuring.
 */
@Composable
fun CameraPermissionRationale(
    onRequestPermission: () -> Unit,
    onPickGallery: () -> Unit,
    onNavigateUp: () -> Unit,
) {
    PermissionScaffold(
        icon = Icons.Filled.PhotoCamera,
        title = stringResource(R.string.camera_permission_rationale_title),
        body = stringResource(R.string.camera_permission_rationale_body),
        onNavigateUp = onNavigateUp,
    ) {
        Button(onClick = onRequestPermission, modifier = Modifier.fillMaxWidth()) {
            Text(stringResource(R.string.action_grant_camera))
        }
        Spacer(Modifier.height(10.dp))
        OutlinedButton(onClick = onPickGallery, modifier = Modifier.fillMaxWidth()) {
            Text(stringResource(R.string.action_pick_from_gallery))
        }
    }
}

/**
 * Shown once the user has permanently denied the camera. The app cannot ask
 * again from here — only Android Settings can grant it — so this says where to go
 * and offers the gallery path, which still works without any permission.
 */
@Composable
fun CameraPermissionDenied(
    onPickGallery: () -> Unit,
    onNavigateUp: () -> Unit,
) {
    PermissionScaffold(
        icon = Icons.Filled.NoPhotography,
        title = stringResource(R.string.camera_permission_denied_title),
        body = stringResource(R.string.camera_permission_denied_body),
        onNavigateUp = onNavigateUp,
    ) {
        Button(onClick = onPickGallery, modifier = Modifier.fillMaxWidth()) {
            Text(stringResource(R.string.action_pick_from_gallery))
        }
    }
}

/** Shared layout for both permission states: icon, heading, explanation, actions. */
@Composable
private fun PermissionScaffold(
    icon: ImageVector,
    title: String,
    body: String,
    onNavigateUp: () -> Unit,
    actions: @Composable () -> Unit,
) {
    Scaffold(
        topBar = {
            PlantGuardTopBar(
                title = stringResource(R.string.camera_title),
                onNavigateUp = onNavigateUp,
            )
        },
    ) { innerPadding ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(innerPadding)
                .padding(horizontal = 24.dp),
            verticalArrangement = Arrangement.Center,
            horizontalAlignment = Alignment.CenterHorizontally,
        ) {
            Icon(
                imageVector = icon,
                contentDescription = null,
                tint = MaterialTheme.colorScheme.primary,
                modifier = Modifier.size(56.dp),
            )
            Spacer(Modifier.height(20.dp))
            Text(
                text = title,
                style = MaterialTheme.typography.headlineSmall,
                textAlign = TextAlign.Center,
            )
            Spacer(Modifier.height(12.dp))
            Text(
                text = body,
                style = MaterialTheme.typography.bodyLarge,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
                textAlign = TextAlign.Center,
            )
            Spacer(Modifier.height(28.dp))
            actions()
        }
    }
}
