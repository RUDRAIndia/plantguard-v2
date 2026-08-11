package com.plantguard.app.ui.theme

import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.darkColorScheme
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable

private val LightColors = lightColorScheme(
    primary = LeafGreen,
    secondary = LeafGreenLight,
    error = ErrorRed,
    background = SurfaceLight,
)

private val DarkColors = darkColorScheme(
    primary = LeafGreenLight,
    secondary = LeafGreen,
    error = ErrorRed,
)

@Composable
fun PlantGuardTheme(content: @Composable () -> Unit) {
    val colors = if (isSystemInDarkTheme()) DarkColors else LightColors
    MaterialTheme(
        colorScheme = colors,
        typography = PlantGuardTypography,
        content = content,
    )
}
