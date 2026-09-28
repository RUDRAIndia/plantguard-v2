package com.plantguard.app.data.history

import androidx.room.migration.Migration
import androidx.sqlite.db.SupportSQLiteDatabase

/**
 * v1 -> v2: drop the old binary `isUnclear` flag and add `topCandidatesJson`.
 *
 * `isUnclear` was the old "confident diagnosis or unclear photo" gate. That gate
 * is gone — a prediction's tier is now derived from its confidence and the
 * boundaries in model_metadata.json — so keeping the column would leave a stored
 * value that contradicts what the app shows.
 *
 * This is written as a real migration rather than
 * `fallbackToDestructiveMigration`, which the previous version used: that would
 * delete the user's saved predictions, and their photos would be left orphaned on
 * disk. CLAUDE.md rule 13 is about not destroying existing data when a working
 * replacement can be built instead, and the same principle applies to a phone.
 *
 * It rebuilds the table rather than issuing `ALTER TABLE ... DROP COLUMN`,
 * because that statement needs SQLite 3.35+ and this app supports minSdk 24,
 * whose bundled SQLite is far older. Create-copy-drop-rename is the portable
 * form, and it is what Room's own migration documentation shows.
 *
 * Existing rows keep their id, photo, confidence, latency and timestamp, and get
 * `"[]"` for candidates — honestly recording that the old code never stored the
 * runners-up, rather than fabricating them.
 */
val MIGRATION_1_2 = object : Migration(1, 2) {
    override fun migrate(db: SupportSQLiteDatabase) {
        // Column types and nullability must match exactly what Room generates
        // for the entity, or Room's schema validation fails on the next open.
        // Float -> REAL, Long/Boolean -> INTEGER, String -> TEXT.
        db.execSQL(
            """
            CREATE TABLE history_entries_new (
                id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
                imagePath TEXT NOT NULL,
                classNameOrNull TEXT,
                confidence REAL NOT NULL,
                inferenceLatencyMs INTEGER NOT NULL,
                timestampMillis INTEGER NOT NULL,
                topCandidatesJson TEXT NOT NULL
            )
            """.trimIndent(),
        )
        db.execSQL(
            """
            INSERT INTO history_entries_new (
                id, imagePath, classNameOrNull, confidence,
                inferenceLatencyMs, timestampMillis, topCandidatesJson
            )
            SELECT id, imagePath, classNameOrNull, confidence,
                   inferenceLatencyMs, timestampMillis, '[]'
            FROM history_entries
            """.trimIndent(),
        )
        db.execSQL("DROP TABLE history_entries")
        db.execSQL("ALTER TABLE history_entries_new RENAME TO history_entries")
    }
}
