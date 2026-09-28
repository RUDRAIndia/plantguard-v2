package com.plantguard.app.ui.components

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.produceState
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File
import kotlin.math.max

/**
 * Draws a JPEG the app previously saved, decoded off the main thread.
 *
 * Written by hand rather than with an image-loading library: the app has no
 * network permission and loads only its own local files, so Coil's caching and
 * HTTP machinery would be dead weight — and adding a dependency to a project
 * that builds offline is a real risk.
 *
 * `produceState` is the important part. The previous version decoded inside
 * `remember`, which runs during composition on the main thread — a visible stall
 * once the history list had more than a few rows. Here the decode suspends on
 * [Dispatchers.IO] and the composable simply draws nothing until it finishes.
 *
 * [maxDimension] caps the decoded size: a 4000px photo scaled into a 56dp
 * thumbnail would otherwise waste tens of megabytes of heap.
 */
@Composable
fun StoredImage(
    path: String,
    contentDescription: String?,
    modifier: Modifier = Modifier,
    maxDimension: Int = 1024,
    contentScale: ContentScale = ContentScale.Crop,
) {
    val bitmap by produceState<Bitmap?>(initialValue = null, path, maxDimension) {
        value = withContext(Dispatchers.IO) { decodeSampled(path, maxDimension) }
    }

    Box(modifier = modifier) {
        val loaded = bitmap
        if (loaded != null) {
            Image(
                bitmap = loaded.asImageBitmap(),
                contentDescription = contentDescription,
                contentScale = contentScale,
                modifier = Modifier.fillMaxSize(),
            )
        } else {
            // A quiet placeholder of the same size, so the layout does not jump
            // when the bitmap arrives.
            Box(
                modifier = Modifier
                    .fillMaxSize()
                    .background(androidx.compose.material3.MaterialTheme.colorScheme.surfaceContainerHigh),
            )
        }
    }
}

/**
 * Decodes at roughly [maxDimension], using the two-pass bounds-then-decode
 * pattern: read the header to learn the real size, pick a power-of-two
 * subsampling factor, then decode. Returns null for a missing or unreadable
 * file — a photo the user deleted from storage should leave a blank thumbnail,
 * not crash the list.
 */
private fun decodeSampled(path: String, maxDimension: Int): Bitmap? {
    val file = File(path)
    if (!file.exists()) return null

    val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
    BitmapFactory.decodeFile(path, bounds)
    if (bounds.outWidth <= 0 || bounds.outHeight <= 0) return null

    var sampleSize = 1
    val longestEdge = max(bounds.outWidth, bounds.outHeight)
    while (longestEdge / (sampleSize * 2) >= maxDimension) {
        sampleSize *= 2
    }

    val options = BitmapFactory.Options().apply { inSampleSize = sampleSize }
    return BitmapFactory.decodeFile(path, options)
}
