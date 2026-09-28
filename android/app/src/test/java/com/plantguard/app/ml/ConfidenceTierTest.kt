package com.plantguard.app.ml

import org.junit.Assert.assertEquals
import org.junit.Assert.assertThrows
import org.junit.Test

/**
 * Pins the exact tiering behaviour the redesign depends on: which side of each
 * boundary counts as which tier, and that a malformed metadata file is rejected
 * rather than silently mistiering every result (CLAUDE.md rule 1).
 *
 * A plain JVM unit test, not an instrumented one — [tierFor] and
 * [ConfidenceBands] touch no Android API, so this runs with `./gradlew test`
 * on this machine without a device or emulator.
 */
class ConfidenceTierTest {

    private val bands = ConfidenceBands(confidentMin = 0.95f, possibleMin = 0.50f)

    @Test
    fun `above confident boundary is CONFIDENT`() {
        assertEquals(ConfidenceTier.CONFIDENT, tierFor(0.96f, bands))
    }

    @Test
    fun `exactly at confident boundary is CONFIDENT`() {
        // Matches the Python side's >= comparison against the tuned threshold.
        assertEquals(ConfidenceTier.CONFIDENT, tierFor(0.95f, bands))
    }

    @Test
    fun `just under confident boundary is POSSIBLE`() {
        assertEquals(ConfidenceTier.POSSIBLE, tierFor(0.94f, bands))
    }

    @Test
    fun `exactly at possible boundary is POSSIBLE`() {
        assertEquals(ConfidenceTier.POSSIBLE, tierFor(0.50f, bands))
    }

    @Test
    fun `just under possible boundary is UNRECOGNISED`() {
        assertEquals(ConfidenceTier.UNRECOGNISED, tierFor(0.49f, bands))
    }

    @Test
    fun `zero confidence is UNRECOGNISED`() {
        assertEquals(ConfidenceTier.UNRECOGNISED, tierFor(0f, bands))
    }

    @Test
    fun `inverted bands throw rather than silently mistiering`() {
        assertThrows(IllegalArgumentException::class.java) {
            ConfidenceBands(confidentMin = 0.5f, possibleMin = 0.5f)
        }
    }

    @Test
    fun `bands outside 0 to 1 throw`() {
        assertThrows(IllegalArgumentException::class.java) {
            ConfidenceBands(confidentMin = 1.0f, possibleMin = 0.5f)
        }
        assertThrows(IllegalArgumentException::class.java) {
            ConfidenceBands(confidentMin = 0.95f, possibleMin = 0f)
        }
    }
}
