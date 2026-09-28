package com.plantguard.app.ui.tips

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.CenterFocusStrong
import androidx.compose.material.icons.filled.CropSquare
import androidx.compose.material.icons.filled.Landscape
import androidx.compose.material.icons.filled.WbSunny
import androidx.compose.material.icons.filled.Yard
import androidx.compose.material.icons.filled.ZoomIn
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.unit.dp
import com.plantguard.app.R
import com.plantguard.app.ui.components.PlantGuardTopBar

private data class Tip(
    val icon: ImageVector,
    val titleRes: Int,
    val bodyRes: Int,
)

/**
 * Concrete framing guidance, not generic advice.
 *
 * This screen is not filler. Of the confidence bands measured on real field
 * photographs, how a leaf is framed is one of the few things a user directly
 * controls — and better framing measurably raises the share of photos that land
 * in the confident tier.
 */
@Composable
fun PhotoTipsScreen(onNavigateUp: () -> Unit) {
    val tips = listOf(
        Tip(Icons.Filled.CropSquare, R.string.tips_single_title, R.string.tips_single_body),
        Tip(Icons.Filled.ZoomIn, R.string.tips_close_title, R.string.tips_close_body),
        Tip(Icons.Filled.CenterFocusStrong, R.string.tips_focus_title, R.string.tips_focus_body),
        Tip(Icons.Filled.WbSunny, R.string.tips_light_title, R.string.tips_light_body),
        Tip(Icons.Filled.Landscape, R.string.tips_background_title, R.string.tips_background_body),
        Tip(Icons.Filled.Yard, R.string.tips_flat_title, R.string.tips_flat_body),
    )

    Scaffold(
        topBar = {
            PlantGuardTopBar(
                title = stringResource(R.string.tips_title),
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
            Text(
                text = stringResource(R.string.tips_intro),
                style = MaterialTheme.typography.bodyLarge,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
            tips.forEach { tip ->
                TipCard(tip)
            }
            Spacer(Modifier.height(16.dp))
        }
    }
}

@Composable
private fun TipCard(tip: Tip) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        shape = MaterialTheme.shapes.medium,
        colors = CardDefaults.cardColors(
            containerColor = MaterialTheme.colorScheme.surfaceContainer,
        ),
        elevation = CardDefaults.cardElevation(defaultElevation = 1.dp),
    ) {
        Row(Modifier.padding(16.dp)) {
            Surface(
                shape = CircleShape,
                color = MaterialTheme.colorScheme.primaryContainer,
                modifier = Modifier.size(44.dp),
            ) {
                Row(
                    horizontalArrangement = Arrangement.Center,
                    verticalAlignment = Alignment.CenterVertically,
                ) {
                    Icon(
                        imageVector = tip.icon,
                        contentDescription = null,
                        tint = MaterialTheme.colorScheme.onPrimaryContainer,
                        modifier = Modifier.size(22.dp),
                    )
                }
            }
            Spacer(Modifier.width(14.dp))
            Column(Modifier.weight(1f)) {
                Text(
                    text = stringResource(tip.titleRes),
                    style = MaterialTheme.typography.titleMedium,
                )
                Spacer(Modifier.height(4.dp))
                Text(
                    text = stringResource(tip.bodyRes),
                    style = MaterialTheme.typography.bodyMedium,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }
        }
    }
}
