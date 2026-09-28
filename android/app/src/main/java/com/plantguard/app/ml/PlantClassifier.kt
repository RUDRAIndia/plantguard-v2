package com.plantguard.app.ml

import android.content.Context
import android.graphics.Bitmap
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

/**
 * One candidate class the model put in its top three.
 *
 * [classIndex] is the model's own output index, and it is what the app passes
 * around when it needs to refer to a class (for example in a navigation route).
 * Class *names* contain commas, spaces and parentheses —
 * "Pepper,_bell___Bacterial_spot" — so they are awkward and error-prone to put
 * in a URL-shaped route, while the index is stable and already defined by
 * model_metadata.json's ordering.
 */
data class ClassCandidate(
    val classIndex: Int,
    val className: String,
    val confidence: Float,
)

/**
 * The outcome of one inference.
 *
 * [candidates] is always the top three, highest confidence first, regardless of
 * tier — even for [ConfidenceTier.UNRECOGNISED], where they are recorded but
 * never shown to the user. Keeping them means a stored history row can be
 * re-displayed later under whatever boundaries are current.
 */
data class ClassificationResult(
    val tier: ConfidenceTier,
    val candidates: List<ClassCandidate>,
    val inferenceLatencyMs: Long,
) {
    /** Confidence of the single best class — what the tier was decided from. */
    val topConfidence: Float get() = candidates.first().confidence
}

/**
 * Wraps a LiteRT (com.google.ai.edge.litert) `Interpreter`. LiteRT fully
 * supports the classic `org.tensorflow.lite.Interpreter` API — same import,
 * same well-documented pattern every "TensorFlow Lite" tutorial uses — so
 * that's what this class is built on, rather than LiteRT's newer
 * `CompiledModel` API, which exists mainly for GPU acceleration this tiny
 * CPU-only model doesn't need.
 *
 * model.tflite is a float16-quantized graph: float32 in, float32 out (see
 * android/README.md and src/config.py's TFLITE_CONFIG). A real Kaggle export
 * run found full-integer (uint8) quantization collapses this model family's
 * accuracy (a 0.41 absolute macro-F1 drop — the known hard-swish/
 * squeeze-excite INT8 quantization pathology), so float16 is what's
 * deployed instead; src/export/to_tflite.py still builds and reports INT8
 * as a comparison artifact, but never deploys it. That means:
 *  - Input: ImagePreprocessing.toFloatBuffer() hands over raw 0-255 pixel
 *    values cast straight to float32 — no normalization/scaling on the app
 *    side at all. The exported graph's raw_pixels_0_255 input applies the
 *    backbone's own preprocessing internally.
 *  - Output: read directly as float32 probabilities — no dequantization
 *    step, unlike the full-integer path this replaced (float16 quantization
 *    only compresses weights; input/output tensors stay float32).
 *
 * build() below asserts both tensors are actually FLOAT32 before use — a
 * uint8-quantized model.tflite dropped in by mistake would otherwise
 * silently produce garbage predictions instead of a clear crash.
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
        val inputBuffer = ImagePreprocessing.toFloatBuffer(squared, metadata.imageSize)

        val outputTensor = interpreter.getOutputTensor(0)
        val numClasses = outputTensor.shape()[1]
        val outputBuffer = Array(1) { FloatArray(numClasses) }

        val startNanos = System.nanoTime()
        interpreter.run(inputBuffer, outputBuffer)
        val latencyMs = (System.nanoTime() - startNanos) / 1_000_000

        val probabilities = outputBuffer[0]
        val candidates = topCandidates(probabilities, TOP_K)

        return ClassificationResult(
            // The tier comes from the metadata's bands, never from a number
            // written in this file — re-tuning the model must not need a rebuild.
            tier = tierFor(candidates.first().confidence, metadata.confidenceBands),
            candidates = candidates,
            inferenceLatencyMs = latencyMs,
        )
    }

    /**
     * The [count] highest-scoring classes, best first.
     *
     * This used to be a single argmax pass keeping only the winner. The top
     * three are needed now because in the middle confidence band the best guess
     * is right only about 41% of the time, while the correct answer is somewhere
     * in the top three about 75% of the time — so showing three candidates is
     * the honest presentation there.
     *
     * Sorting all 38 scores is plenty fast (one inference costs far more) and is
     * much easier to read than a hand-rolled partial selection.
     */
    private fun topCandidates(probabilities: FloatArray, count: Int): List<ClassCandidate> {
        return probabilities
            .mapIndexed { index, probability ->
                ClassCandidate(
                    classIndex = index,
                    className = metadata.classNames[index],
                    confidence = probability,
                )
            }
            .sortedByDescending { it.confidence }
            .take(count)
    }

    /** Not called during normal app life (this is a process-lifetime singleton) — kept for completeness/tests. */
    fun close() {
        interpreter.close()
    }

    companion object {
        private const val MODEL_FILE_NAME = "model.tflite"

        /** How many candidates the "possible matches" tier shows. */
        const val TOP_K = 3

        @Volatile
        private var instance: PlantClassifier? = null

        fun getInstance(context: Context): PlantClassifier {
            return instance ?: synchronized(this) {
                instance ?: build(context).also { instance = it }
            }
        }

        private fun build(context: Context): PlantClassifier {
            val metadata = ModelMetadata.getInstance(context)
            val interpreter = Interpreter(loadModelFile(context))

            val inputTensor = interpreter.getInputTensor(0)
            val outputTensor = interpreter.getOutputTensor(0)
            require(inputTensor.dataType() == DataType.FLOAT32 && outputTensor.dataType() == DataType.FLOAT32) {
                "model.tflite has input dtype ${inputTensor.dataType()} and output dtype " +
                    "${outputTensor.dataType()} — PlantClassifier only supports a float16-quantized " +
                    "export (float32 input/output tensors, config.ANDROID_TFLITE_IO_DTYPE). A " +
                    "full-integer (uint8) export would silently produce garbage predictions with this " +
                    "code instead of failing here — re-export as float16 or update this class."
            }

            val numClasses = outputTensor.shape()[1]
            require(numClasses == metadata.classNames.size) {
                "model.tflite outputs $numClasses classes but model_metadata.json lists " +
                    "${metadata.classNames.size} class_names — these two files must always be " +
                    "replaced together (see android/README.md)."
            }
            require(numClasses >= TOP_K) {
                "model.tflite outputs only $numClasses classes, fewer than the $TOP_K candidates " +
                    "the 'possible matches' result tier shows."
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
