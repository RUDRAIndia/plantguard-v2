package com.plantguard.app.navigation

/** Plain string routes — simplest option for a 4-screen app, no extra kotlinx.serialization plugin needed. */
object NavRoutes {
    const val DISCLAIMER = "disclaimer"
    const val CAMERA = "camera"
    const val HISTORY = "history"

    private const val RESULT_TEMPLATE = "result/{historyEntryId}"
    const val RESULT_ARG_ID = "historyEntryId"
    const val RESULT = RESULT_TEMPLATE

    fun resultRoute(historyEntryId: Long): String = "result/$historyEntryId"
}
