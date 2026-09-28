package com.plantguard.app.data.history

import com.plantguard.app.ml.ClassCandidate
import org.json.JSONArray
import org.json.JSONObject

/**
 * Encodes the top-three candidate list into the single text column
 * [HistoryEntry.topCandidatesJson], and back out again.
 *
 * A small JSON string in one column is used rather than a second Room table
 * with a foreign key: the list is always exactly the top three, it is only ever
 * read as a whole alongside its row, and it is never queried or sorted by. A
 * child table would add a join and a migration for no benefit here.
 *
 * `org.json` is used because it is part of Android itself — the rest of the app
 * already parses its two JSON assets with it, so this adds no dependency.
 */
object HistoryCandidates {

    private const val KEY_CLASS_INDEX = "classIndex"
    private const val KEY_CLASS_NAME = "className"
    private const val KEY_CONFIDENCE = "confidence"

    /** The value stored for a row whose candidates are unknown (all v1 rows). */
    const val EMPTY = "[]"

    fun encode(candidates: List<ClassCandidate>): String {
        val array = JSONArray()
        candidates.forEach { candidate ->
            array.put(
                JSONObject()
                    .put(KEY_CLASS_INDEX, candidate.classIndex)
                    .put(KEY_CLASS_NAME, candidate.className)
                    .put(KEY_CONFIDENCE, candidate.confidence.toDouble()),
            )
        }
        return array.toString()
    }

    /**
     * Returns the stored candidates, or an empty list for a row written before
     * this column existed. Callers must handle empty — a v1 row genuinely has
     * no candidate list, and inventing one would show a farmer alternatives the
     * model never actually produced.
     */
    fun decode(json: String): List<ClassCandidate> {
        val array = JSONArray(json)
        return (0 until array.length()).map { i ->
            val item = array.getJSONObject(i)
            ClassCandidate(
                classIndex = item.getInt(KEY_CLASS_INDEX),
                className = item.getString(KEY_CLASS_NAME),
                confidence = item.getDouble(KEY_CONFIDENCE).toFloat(),
            )
        }
    }
}
