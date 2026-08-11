package com.plantguard.app.data.prefs

import android.content.Context

/**
 * Whether the user has already dismissed the first-launch disclaimer.
 *
 * Plain SharedPreferences rather than DataStore: this is a single boolean
 * read synchronously once at app startup, before the navigation graph even
 * exists. DataStore's Flow/coroutine API is the modern general recommendation,
 * but it adds a real concept (suspending before the first screen can even be
 * chosen) with no payoff for one flag. If this ever grows into real
 * structured settings, migrating is a one-file change.
 */
class DisclaimerPrefs(context: Context) {

    private val prefs = context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)

    fun hasSeenDisclaimer(): Boolean = prefs.getBoolean(KEY_SEEN, false)

    fun setSeen() {
        prefs.edit().putBoolean(KEY_SEEN, true).apply()
    }

    private companion object {
        const val PREFS_NAME = "plantguard_prefs"
        const val KEY_SEEN = "has_seen_disclaimer"
    }
}
