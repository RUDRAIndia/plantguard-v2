package com.plantguard.app.ml

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Matrix
import android.net.Uri
import androidx.exifinterface.media.ExifInterface
import java.io.File
import java.io.InputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * Turns a raw camera/gallery photo into the exact uint8 RGB buffer
 * PlantClassifier feeds the model. Three real decisions live here, not
 * afterthoughts:
 *  1. Decode at a bounded size first (inSampleSize) — a full-resolution
 *     camera photo can be tens of megapixels; decoding that directly into
 *     a Bitmap can OOM on a minSdk-24-era device.
 *  2. Correct EXIF rotation before anything else — camera/gallery JPEGs are
 *     very often stored "sideways" with an EXIF tag saying how to rotate
 *     them for display; skipping this feeds a sideways leaf to the model.
 *  3. Center-crop to square before resizing to the model's fixed
 *     [ModelMetadata.imageSize] — a raw 4:3/16:9 photo squashed directly to
 *     square would distort the leaf's shape.
 */
object ImagePreprocessing {

    private const val MAX_DECODE_DIMENSION = 1024

    fun decodeBitmapFromFile(file: File): Bitmap {
        val bitmap = decodeSampled({ file.inputStream() }, MAX_DECODE_DIMENSION)
        val orientation = ExifInterface(file.absolutePath).rotationDegrees
        return rotateBitmap(bitmap, orientation)
    }

    fun decodeBitmapFromUri(context: Context, uri: Uri): Bitmap {
        val bitmap = decodeSampled({ context.openStreamOrThrow(uri) }, MAX_DECODE_DIMENSION)
        val orientation = context.openStreamOrThrow(uri).use { ExifInterface(it).rotationDegrees }
        return rotateBitmap(bitmap, orientation)
    }

    fun centerCropToSquare(bitmap: Bitmap): Bitmap {
        val size = minOf(bitmap.width, bitmap.height)
        val x = (bitmap.width - size) / 2
        val y = (bitmap.height - size) / 2
        return Bitmap.createBitmap(bitmap, x, y, size, size)
    }

    /** Raw uint8 RGB pixels, HWC order — exactly what the uint8-input model expects, no float math. */
    fun toUint8Buffer(bitmap: Bitmap, size: Int): ByteBuffer {
        val resized =
            if (bitmap.width == size && bitmap.height == size) bitmap
            else Bitmap.createScaledBitmap(bitmap, size, size, true)

        val buffer = ByteBuffer.allocateDirect(size * size * 3).order(ByteOrder.nativeOrder())
        val pixels = IntArray(size * size)
        resized.getPixels(pixels, 0, size, 0, 0, size, size)
        for (pixel in pixels) {
            buffer.put(((pixel shr 16) and 0xFF).toByte()) // R
            buffer.put(((pixel shr 8) and 0xFF).toByte())  // G
            buffer.put((pixel and 0xFF).toByte())          // B
        }
        buffer.rewind()
        return buffer
    }

    private fun decodeSampled(openStream: () -> InputStream, maxDimension: Int): Bitmap {
        val boundsOptions = BitmapFactory.Options().apply { inJustDecodeBounds = true }
        openStream().use { BitmapFactory.decodeStream(it, null, boundsOptions) }

        var sampleSize = 1
        while (boundsOptions.outWidth / sampleSize > maxDimension ||
            boundsOptions.outHeight / sampleSize > maxDimension
        ) {
            sampleSize *= 2
        }

        val decodeOptions = BitmapFactory.Options().apply { inSampleSize = sampleSize }
        return openStream().use { stream ->
            BitmapFactory.decodeStream(stream, null, decodeOptions)
                ?: error("Could not decode image (not a supported format?)")
        }
    }

    private fun rotateBitmap(bitmap: Bitmap, degrees: Int): Bitmap {
        if (degrees == 0) return bitmap
        val matrix = Matrix().apply { postRotate(degrees.toFloat()) }
        return Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
    }

    private fun Context.openStreamOrThrow(uri: Uri): InputStream =
        contentResolver.openInputStream(uri) ?: error("Could not open $uri")

    private val ExifInterface.rotationDegrees: Int
        get() = when (getAttributeInt(ExifInterface.TAG_ORIENTATION, ExifInterface.ORIENTATION_NORMAL)) {
            ExifInterface.ORIENTATION_ROTATE_90 -> 90
            ExifInterface.ORIENTATION_ROTATE_180 -> 180
            ExifInterface.ORIENTATION_ROTATE_270 -> 270
            else -> 0
        }
}
