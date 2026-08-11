package com.plantguard.app.util

import android.content.Context
import android.graphics.Bitmap
import java.io.File
import java.io.FileOutputStream
import java.util.UUID

/**
 * Saves a captured/picked photo into app-private storage so History has a
 * durable file to show a thumbnail from later. Needed because the original
 * camera/gallery source (a CameraX in-memory capture, or a picked content
 * Uri) isn't guaranteed to still be readable after this session ends.
 */
object ImageStorage {

    fun save(context: Context, bitmap: Bitmap): File {
        val imagesDir = File(context.filesDir, "images").apply { mkdirs() }
        val file = File(imagesDir, "${UUID.randomUUID()}.jpg")
        FileOutputStream(file).use { out ->
            bitmap.compress(Bitmap.CompressFormat.JPEG, 90, out)
        }
        return file
    }
}
