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

// --- Light mode: white ground, blue brand. ---
val PureWhite = Color(0xFFFFFFFF)
val MistBlue = Color(0xFFF1F5FB)
val MistBlueHigh = Color(0xFFE7EEF7)
val OceanBlue = Color(0xFF1565C0)
val BlueDeep = Color(0xFF0B3C78)
val BluePale = Color(0xFFE3F0FC)
val SlateBlue = Color(0xFF45607A)
val SlatePale = Color(0xFFDCE6F0)
val SlateDeep = Color(0xFF1C2C3B)
val InkDark = Color(0xFF101418)
val InkDim = Color(0xFF4A5560)
val MistOutline = Color(0xFFC4D0DC)
val MistOutlineSoft = Color(0xFFE1E8F0)

// --- Dark mode: near-black ground, rose brand, teal counterpoint. ---
val InkBlack = Color(0xFF0B0B0D)
val InkSurface = Color(0xFF15151A)
val InkRaised = Color(0xFF1F1F26)
val InkOutline = Color(0xFF3A3A45)
val InkOutlineSoft = Color(0xFF26262E)
val InkText = Color(0xFFF2F2F5)
val InkTextDim = Color(0xFFB8B8C2)
val RosePink = Color(0xFFFF4D8D)
val RoseDeep = Color(0xFF2B0518)
val RoseContainer = Color(0xFF5C0A2C)
val RosePale = Color(0xFFFFD3E2)
val TealBright = Color(0xFF6FD8D1)
val TealDeep = Color(0xFF00332F)
val TealContainer = Color(0xFF0E3B38)
val TealPale = Color(0xFFA8EFE9)
