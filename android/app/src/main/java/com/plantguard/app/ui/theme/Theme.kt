package com.plantguard.app.ui.theme

import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.darkColorScheme
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.runtime.CompositionLocalProvider

/**
 * Every slot any screen touches is assigned explicitly.
 *
 * The previous version set only four, so `secondaryContainer`, `errorContainer`
 * and `surfaceVariant` — all of which screens actually used — still came from
 * Material's violet baseline and clashed with the greens. Leaving a slot to the
 * default is how a "designed" app ends up looking half-themed.
 *
 * Dynamic colour (Material You) is deliberately not used: the whole point of
 * this palette is that the app looks like the same considered product on every
 * phone, and tier colours in particular must not be re-tinted by a wallpaper,
 * since they encode how much to trust a diagnosis.
 */
private val LightColors = lightColorScheme(
    primary = Canopy,
    onPrimary = CreamLight,
    primaryContainer = CanopyPale,
    onPrimaryContainer = CanopyDeep,
    secondary = Soil,
    onSecondary = CreamLight,
    secondaryContainer = SoilPale,
    onSecondaryContainer = SoilDeep,
    tertiary = Wheat,
    onTertiary = CreamLight,
    tertiaryContainer = WheatPale,
    onTertiaryContainer = WheatDark,
    background = CreamLight,
    onBackground = SoilDeep,
    surface = CreamLight,
    onSurface = SoilDeep,
    surfaceVariant = Cream,
    onSurfaceVariant = Soil,
    surfaceContainer = Cream,
    surfaceContainerHigh = CreamDim,
    outline = SoilLight,
    outlineVariant = CreamDim,
    error = ClayRed,
    onError = CreamLight,
    errorContainer = ClayRedPale,
    onErrorContainer = ClayRed,
    inverseSurface = CanopyDeep,
    inverseOnSurface = CreamLight,
)

private val DarkColors = darkColorScheme(
    primary = CanopyLight,
    onPrimary = CanopyDeep,
    primaryContainer = CanopyDark,
    onPrimaryContainer = CanopyPale,
    secondary = SoilLight,
    onSecondary = SoilDeep,
    secondaryContainer = SoilDeep,
    onSecondaryContainer = SoilPale,
    tertiary = WheatLight,
    onTertiary = SoilDeep,
    tertiaryContainer = WheatDark,
    onTertiaryContainer = WheatPale,
    background = NightGround,
    onBackground = NightText,
    surface = NightSurface,
    onSurface = NightText,
    surfaceVariant = NightSurfaceRaised,
    onSurfaceVariant = NightTextDim,
    surfaceContainer = NightSurface,
    surfaceContainerHigh = NightSurfaceRaised,
    outline = NightOutline,
    outlineVariant = NightSurfaceRaised,
    error = ClayRedLight,
    onError = ClayRedDeep,
    errorContainer = ClayRedDark,
    onErrorContainer = ClayRedLight,
    inverseSurface = CreamLight,
    inverseOnSurface = SoilDeep,
)

@Composable
fun PlantGuardTheme(
    darkTheme: Boolean = isSystemInDarkTheme(),
    content: @Composable () -> Unit,
) {
    // The tier colours are provided alongside the Material scheme so that both
    // switch together — a light card left behind in dark mode is exactly the
    // kind of seam this avoids.
    CompositionLocalProvider(
        LocalPlantGuardColors provides if (darkTheme) DarkPlantGuardColors else LightPlantGuardColors,
    ) {
        MaterialTheme(
            colorScheme = if (darkTheme) DarkColors else LightColors,
            typography = PlantGuardTypography,
            shapes = PlantGuardShapes,
            content = content,
        )
    }
}
