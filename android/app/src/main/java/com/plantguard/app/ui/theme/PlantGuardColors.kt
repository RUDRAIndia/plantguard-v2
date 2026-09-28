package com.plantguard.app.ui.theme

import androidx.compose.runtime.Composable
import androidx.compose.runtime.ReadOnlyComposable
import androidx.compose.runtime.staticCompositionLocalOf
import androidx.compose.ui.graphics.Color
import com.plantguard.app.ml.ConfidenceTier

/**
 * The colours one result tier is drawn in.
 *
 * @param container background of the tier's own cards and banners.
 * @param onContainer text and icons drawn on [container].
 * @param accent the saturated edge colour — meter fills, badge strokes, rules.
 */
data class TierColors(
    val container: Color,
    val onContainer: Color,
    val accent: Color,
)

/**
 * Colours this app needs that Material's `ColorScheme` has no slot for.
 *
 * The three result tiers must be tellable apart at a glance — a confident
 * diagnosis and a hedged "possible matches" list must never look alike, because
 * that visual difference is how a farmer knows how much to trust the screen.
 * Material offers primary/secondary/tertiary/error, none of which means
 * "hedged", so mapping tiers onto them would be arbitrary and would fight the
 * scheme elsewhere. Instead they live here and travel by CompositionLocal — the
 * documented way to extend a Material 3 theme.
 */
data class PlantGuardColors(
    val confident: TierColors,
    val possible: TierColors,
    val unrecognised: TierColors,
    /** Background for the always-visible accuracy notice. */
    val noticeContainer: Color,
    val onNoticeContainer: Color,
) {
    fun forTier(tier: ConfidenceTier): TierColors = when (tier) {
        ConfidenceTier.CONFIDENT -> confident
        ConfidenceTier.POSSIBLE -> possible
        ConfidenceTier.UNRECOGNISED -> unrecognised
    }
}

internal val LightPlantGuardColors = PlantGuardColors(
    confident = TierColors(container = CanopyPale, onContainer = CanopyDeep, accent = Canopy),
    possible = TierColors(container = WheatPale, onContainer = WheatDark, accent = Wheat),
    unrecognised = TierColors(container = SoilPale, onContainer = SoilDeep, accent = Soil),
    noticeContainer = Cream,
    onNoticeContainer = SoilDeep,
)

internal val DarkPlantGuardColors = PlantGuardColors(
    // Containers stay dark and only slightly tinted: a pale card on a near-black
    // screen glares at night. The accent carries the tier's identity instead.
    confident = TierColors(container = Color(0xFF1C2E1D), onContainer = CanopyLight, accent = CanopyMid),
    possible = TierColors(container = Color(0xFF2E2415), onContainer = WheatLight, accent = WheatLight),
    unrecognised = TierColors(container = Color(0xFF272119), onContainer = SoilLight, accent = SoilLight),
    noticeContainer = NightSurfaceRaised,
    onNoticeContainer = NightTextDim,
)

/**
 * staticCompositionLocalOf rather than compositionLocalOf: these values change
 * only when the whole theme changes, so there is no reason to pay for
 * fine-grained recomposition tracking on every read.
 */
internal val LocalPlantGuardColors = staticCompositionLocalOf { LightPlantGuardColors }

/**
 * Tier colours for the current theme. Read as `PlantGuardTheme.colors` from any
 * composable inside [PlantGuardTheme], mirroring how `MaterialTheme.colorScheme`
 * is normally reached.
 */
object PlantGuardTheme {
    val colors: PlantGuardColors
        @Composable
        @ReadOnlyComposable
        get() = LocalPlantGuardColors.current
}
