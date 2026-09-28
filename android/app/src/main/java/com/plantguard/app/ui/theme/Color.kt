package com.plantguard.app.ui.theme

import androidx.compose.ui.graphics.Color

/**
 * An agricultural palette: canopy greens, warm cream paper, and earth tones,
 * rather than the stock Material baseline (which is violet, and showed through
 * anywhere the old theme left a slot unassigned).
 *
 * Names describe the colour, not the role — roles are assigned in Theme.kt, so
 * the same swatch can serve more than one slot without a misleading name.
 */

// --- Greens: the crop canopy. Primary brand colour and its light-mode ramp. ---
val CanopyDeep = Color(0xFF14361D)
val CanopyDark = Color(0xFF1B5E20)
val Canopy = Color(0xFF2E7D32)
val CanopyMid = Color(0xFF4C9A51)
val CanopyLight = Color(0xFFA5D6A7)
val CanopyPale = Color(0xFFDCEEDC)

// --- Cream / paper: the warm ground the whole app sits on. ---
val CreamLight = Color(0xFFFAF6EC)
val Cream = Color(0xFFF3EDDD)
val CreamDim = Color(0xFFE6DEC9)

// --- Earth: soil and terracotta, for secondary surfaces and the third tier. ---
val SoilDeep = Color(0xFF3E2F23)
val Soil = Color(0xFF6D5643)
val SoilLight = Color(0xFFBFA98F)
val SoilPale = Color(0xFFEDE3D5)

// --- Wheat amber: the hedged, "not confident" signal. ---
val WheatDark = Color(0xFF8A5A00)
val Wheat = Color(0xFFC77800)
val WheatLight = Color(0xFFF7C566)
val WheatPale = Color(0xFFFBEED1)

// --- Alarm red: genuine errors only, never a hedged result. ---
val ClayRedDeep = Color(0xFF601410)
val ClayRedDark = Color(0xFF8C1D18)
val ClayRed = Color(0xFFB3261E)
val ClayRedLight = Color(0xFFF2B8B5)
val ClayRedPale = Color(0xFFFBE9E8)

// --- Night mode: near-black with a green cast, so it reads as the same product. ---
val NightGround = Color(0xFF11150F)
val NightSurface = Color(0xFF1A1F17)
val NightSurfaceRaised = Color(0xFF242A20)
val NightOutline = Color(0xFF47503F)
val NightText = Color(0xFFE6E8E0)
val NightTextDim = Color(0xFFBFC4B6)
