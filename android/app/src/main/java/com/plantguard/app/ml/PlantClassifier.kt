package com.plantguard.app.ml

import android.content.Context
import android.graphics.Bitmap
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

/**
 * [isUnclear] true means the OOD-rejection path fired (top probability below
 * [ModelMetadata.confidenceThreshold]) — [className] is null in that case;
 * there is no disease to show, on purpose.
 */
data class ClassificationResult(
    val isUnclear: Boolean,
    val className: String?,
    val confidence: Float,
    val inferenceLatencyMs: Long,
)

/**
 * Wraps a LiteRT (com.google.ai.edge.litert) `Interpreter`. LiteRT fully
 * supports the classic `org.tensorflow.lite.Interpreter` API — same import,
 * same well-documented pattern every "TensorFlow Lite" tutorial uses — so
 * that's what this class is built on, rather than LiteRT's newer
 * `CompiledModel` API, which exists mainly for GPU acceleration this tiny
 * CPU-only model doesn't need.
 *
 * model.tflite is a full-integer-quantized graph: uint8 in, uint8 out (see
 * android/README.md and src/config.py's TFLITE_CONFIG). That means:
 *  - Input: ImagePreprocessing.toUint8Buffer() hands over raw 0-255 pixel
 *    bytes directly — no float normalization on the app side at all. The
 *    quantization baked into the .tflite already accounts for whatever
 *    float preprocessing the model was trained with.
 *  - Output: dequantized here using THAT OUTPUT TENSOR'S OWN scale/zero-point,
 *    read from the .tflite file itself at runtime via
 *    `Tensor.quantizationParams()` — never a hardcoded formula, since those
 *    two numbers are specific to how this particular model was quantized.
 *
 * A process-lifetime singleton (like AppDatabase): building an Interpreter
 * has real cost, and nothing about it needs to be recreated per-screen.
 */
class PlantClassifier private constructor(
    private val interpreter: Interpreter,
    val metadata: ModelMetadata,
) {

    fun classify(bitmap: Bitmap): ClassificationResult {
        val squared = ImagePreprocessing.centerCropToSquare(bitmap)
        val inputBuffer = ImagePreprocessing.toUint8Buffer(squared, metadata.imageSize)

        val outputTensor = interpreter.getOutputTensor(0)
        val numClasses = outputTensor.shape()[1]
        val outputBuffer = Array(1) { ByteArray(numClasses) }

        val startNanos = System.nanoTime()
        interpreter.run(inputBuffer, outputBuffer)
        val latencyMs = (System.nanoTime() - startNanos) / 1_000_000

        val quant = outputTensor.quantizationParams()
        val probabilities = outputBuffer[0].map { rawByte ->
            val rawUnsigned = rawByte.toInt() and 0xFF
            (rawUnsigned - quant.zeroPoint) * quant.scale
        }

        var topIndex = 0
        var topProbability = 0f
        probabilities.forEachIndexed { index, probability ->
            if (probability > topProbability) {
                topProbability = probability
                topIndex = index
            }
        }

        val isUnclear = topProbability < metadata.confidenceThreshold
        return ClassificationResult(
            isUnclear = isUnclear,
            className = if (isUnclear) null else metadata.classNames[topIndex],
            confidence = topProbability,
            inferenceLatencyMs = latencyMs,
        )
    }

    /** Not called during normal app life (this is a process-lifetime singleton) — kept for completeness/tests. */
    fun close() {
        interpreter.close()
    }

    companion object {
        private const val MODEL_FILE_NAME = "model.tflite"

        @Volatile
        private var instance: PlantClassifier? = null

        fun getInstance(context: Context): PlantClassifier {
            return instance ?: synchronized(this) {
                instance ?: build(context).also { instance = it }
            }
        }

        private fun build(context: Context): PlantClassifier {
            val metadata = ModelMetadata.loadFromAssets(context)
            val interpreter = Interpreter(loadModelFile(context))

            val numClasses = interpreter.getOutputTensor(0).shape()[1]
            require(numClasses == metadata.classNames.size) {
                "model.tflite outputs $numClasses classes but model_metadata.json lists " +
                    "${metadata.classNames.size} class_names — these two files must always be " +
                    "replaced together (see android/README.md)."
            }

            return PlantClassifier(interpreter, metadata)
        }

        private fun loadModelFile(context: Context): MappedByteBuffer {
            val assetFd = context.assets.openFd(MODEL_FILE_NAME)
            FileInputStream(assetFd.fileDescriptor).use { inputStream ->
                val channel = inputStream.channel
                return channel.map(FileChannel.MapMode.READ_ONLY, assetFd.startOffset, assetFd.declaredLength)
            }
        }
    }
}
