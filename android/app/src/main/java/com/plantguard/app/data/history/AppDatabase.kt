package com.plantguard.app.data.history

import android.content.Context
import androidx.room.Database
import androidx.room.Room
import androidx.room.RoomDatabase

@Database(entities = [HistoryEntry::class], version = 2, exportSchema = false)
abstract class AppDatabase : RoomDatabase() {

    abstract fun historyDao(): HistoryDao

    companion object {
        @Volatile
        private var instance: AppDatabase? = null

        fun getInstance(context: Context): AppDatabase {
            return instance ?: synchronized(this) {
                instance ?: Room.databaseBuilder(
                    context.applicationContext,
                    AppDatabase::class.java,
                    "plantguard.db",
                )
                    // A real migration, and deliberately no
                    // fallbackToDestructiveMigration: an upgrade must not delete
                    // the user's saved predictions (see MIGRATION_1_2). Without a
                    // fallback, a schema change with no matching migration now
                    // crashes loudly on open instead of silently wiping the table
                    // — which is the behaviour CLAUDE.md rule 1 asks for.
                    .addMigrations(MIGRATION_1_2)
                    .build()
                    .also { instance = it }
            }
        }
    }
}
