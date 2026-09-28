package com.plantguard.app.navigation

/**
 * Every destination in the app, as plain string routes.
 *
 * Plain strings rather than navigation-compose's newer type-safe routes, which
 * would pull in the kotlinx-serialization plugin for two arguments — both of them
 * a single number.
 *
 * Note that the disease-detail route carries a class *index*, not a class name.
 * The 38 names contain commas, spaces and parentheses
 * ("Pepper,_bell___Bacterial_spot"), which would have to be URL-encoded on the
 * way in and decoded on the way out — an easy thing to get subtly wrong. The
 * index is already the model's own stable ordering from model_metadata.json.
 */
object NavRoutes {
    const val DISCLAIMER = "disclaimer"
    const val HOME = "home"
    const val CAMERA = "camera"
    const val HISTORY = "history"
    const val ABOUT = "about"
    const val PHOTO_TIPS = "photo_tips"

    const val RESULT_ARG_ID = "historyEntryId"
    const val RESULT = "result/{$RESULT_ARG_ID}"

    const val DISEASE_ARG_INDEX = "classIndex"
    const val DISEASE = "disease/{$DISEASE_ARG_INDEX}"

    fun resultRoute(historyEntryId: Long): String = "result/$historyEntryId"

    fun diseaseRoute(classIndex: Int): String = "disease/$classIndex"
}
