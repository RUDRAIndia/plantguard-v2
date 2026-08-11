package com.plantguard.app.data.history

import androidx.room.Dao
import androidx.room.Insert
import androidx.room.Query
import kotlinx.coroutines.flow.Flow

@Dao
interface HistoryDao {

    @Insert
    suspend fun insert(entry: HistoryEntry): Long

    // Newest first, so History always opens on the most recent prediction.
    @Query("SELECT * FROM history_entries ORDER BY timestampMillis DESC")
    fun observeAll(): Flow<List<HistoryEntry>>

    @Query("SELECT * FROM history_entries WHERE id = :id")
    suspend fun getById(id: Long): HistoryEntry?
}
