package io.github.lbenedetto.internal.model

import kotlin.math.cos
import kotlin.math.pow
import kotlin.math.sin

object RoPE {
  fun precomputeFreqsCis(contextLength: Int, headSize: Int, theta: Double): Pair<FloatArray, FloatArray> {
    assert(headSize % 2 == 0)
    val halfHead = headSize / 2
    val cr = FloatArray(contextLength * halfHead)
    val ci = FloatArray(contextLength * halfHead)
    var n = 0
    for (pos in 0..<contextLength) {
      var i = 0
      while (i < headSize) {
        val freq = (1.0 / theta.pow(i / headSize.toDouble())).toFloat()
        val `val` = pos * freq
        cr[n] = cos(`val`.toDouble()).toFloat()
        ci[n] = sin(`val`.toDouble()).toFloat()
        n++
        i += 2
      }
    }
    assert(contextLength * halfHead == n)
    return Pair(cr, ci)
  }

  fun precomputeFreqsCisFromFreqs(
    contextLength: Int,
    headSize: Int,
    ropeTheta: Double,
    ropeFreqFactors: FloatArray
  ): Pair<FloatArray, FloatArray> {
    // freq_factors are divisors on top of the standard RoPE base frequencies:
    // theta_i = pos * (1 / (ropeTheta^(2i/headSize))) / freqFactors[i]
    val halfHead = ropeFreqFactors.size
    assert(halfHead == headSize / 2)
    val cr = FloatArray(contextLength * halfHead)
    val ci = FloatArray(contextLength * halfHead)
    var n = 0
    for (pos in 0..<contextLength) {
      for (i in 0..<halfHead) {
        val baseFreq = (1.0 / ropeTheta.pow((2.0 * i) / headSize)).toFloat()
        val `val` = pos * baseFreq / ropeFreqFactors[i]
        cr[n] = cos(`val`.toDouble()).toFloat()
        ci[n] = sin(`val`.toDouble()).toFloat()
        n++
      }
    }
    assert(contextLength * halfHead == n)
    return Pair(cr, ci)
  }
}
