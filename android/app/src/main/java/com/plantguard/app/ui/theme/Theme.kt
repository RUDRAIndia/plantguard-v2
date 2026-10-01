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
    primary = OceanBlue,
    onPrimary = PureWhite,
    primaryContainer = BluePale,
    onPrimaryContainer = BlueDeep,
    secondary = SlateBlue,
    onSecondary = PureWhite,
    secondaryContainer = SlatePale,
    onSecondaryContainer = SlateDeep,
    tertiary = Wheat,
    onTertiary = PureWhite,
    tertiaryContainer = WheatPale,
    onTertiaryContainer = WheatDark,
    background = PureWhite,
    onBackground = InkDark,
    surface = PureWhite,
    onSurface = InkDark,
    surfaceVariant = MistBlue,
    onSurfaceVariant = InkDim,
    surfaceContainer = MistBlue,
    surfaceContainerHigh = MistBlueHigh,
    outline = MistOutline,
    outlineVariant = MistOutlineSoft,
    error = ClayRed,
    onError = PureWhite,
    errorContainer = ClayRedPale,
    onErrorContainer = ClayRed,
    inverseSurface = BlueDeep,
    inverseOnSurface = PureWhite,
)

private val DarkColors = darkColorScheme(
    primary = RosePink,
    onPrimary = RoseDeep,
    primaryContainer = RoseContainer,
    onPrimaryContainer = RosePale,
    secondary = TealBright,
    onSecondary = TealDeep,
    secondaryContainer = TealContainer,
    onSecondaryContainer = TealPale,
    tertiary = WheatLight,
    onTertiary = RoseDeep,
    tertiaryContainer = WheatDark,
    onTertiaryContainer = WheatPale,
    background = InkBlack,
    onBackground = InkText,
    surface = InkSurface,
    onSurface = InkText,
    surfaceVariant = InkRaised,
    onSurfaceVariant = InkTextDim,
    surfaceContainer = InkSurface,
    surfaceContainerHigh = InkRaised,
    outline = InkOutline,
    outlineVariant = InkOutlineSoft,
    error = ClayRedLight,
    onError = ClayRedDeep,
    errorContainer = ClayRedDark,
    onErrorContainer = ClayRedLight,
    inverseSurface = PureWhite,
    inverseOnSurface = InkDark,
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
