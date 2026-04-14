package io.github.lbenedetto.internal.model

import io.github.lbenedetto.internal.floattensor.FloatTensor
import io.github.lbenedetto.internal.floattensor.MutableFloatTensor
import io.github.lbenedetto.internal.floattensor.parallelFor
import io.github.lbenedetto.internal.sampler.Sampler
import io.github.lbenedetto.internal.tokenizer.GemmaTokenizer
import io.github.lbenedetto.internal.tokenizer.GemmaTokenizer.Companion.replaceControlCharacters
import java.nio.FloatBuffer
import java.util.function.IntConsumer
import kotlin.math.max
import kotlin.math.pow
import kotlin.math.sqrt
import kotlin.math.tanh

internal data class Llama(val configuration: LlamaConfiguration, val tokenizer: GemmaTokenizer, val weights: LlamaWeights?) {
  fun createNewState(): LlamaState {
    val state = LlamaState(this.configuration)
    state.latestToken = tokenizer.specialTokens["<bos>"]!!
    return state
  }

  companion object {
    fun gelu(x: Float): Float {
      return (0.5 * x * (1 + tanh(sqrt(2 / Math.PI) * (x + 0.044715 * x.toDouble().pow(3.0))))).toFloat()
    }

    fun rmsnorm(out: MutableFloatTensor, x: FloatTensor, weight: FloatBuffer, size: Int, rmsNormEps: Float) {
      var ss = x.reduce(0, size, 0f) { acc, xi -> acc + xi * xi }
      ss /= size.toFloat()
      ss += rmsNormEps
      ss = (1.0 / sqrt(ss.toDouble())).toFloat()
      val finalss = ss
      out.mapWithIndexInPlace(
        0,
        size
      ) { _, index -> weight.get(index) * (finalss * x[index]) }
    }

    fun rmsnorm(
      out: MutableFloatTensor,
      outOffset: Int,
      x: FloatTensor,
      xOffset: Int,
      weight: FloatBuffer,
      size: Int,
      rmsNormEps: Float
    ) {
      var ss = 0f
      for (i in 0..<size) {
        val xi = x[xOffset + i]
        ss += xi * xi
      }
      ss /= size.toFloat()
      ss += rmsNormEps
      ss = (1.0 / sqrt(ss.toDouble())).toFloat()
      for (i in 0..<size) {
        out.setFloat(outOffset + i, weight.get(i) * ss * x[xOffset + i])
      }
    }

    // Bare RMS norm without learned weights (just normalize to unit RMS)
    fun rmsnormNoWeight(out: MutableFloatTensor, outOffset: Int, x: FloatTensor, xOffset: Int, size: Int, rmsNormEps: Float) {
      var ss = 0f
      for (i in 0..<size) {
        val xi = x[xOffset + i]
        ss += xi * xi
      }
      ss /= size.toFloat()
      ss += rmsNormEps
      ss = (1.0 / sqrt(ss.toDouble())).toFloat()
      for (i in 0..<size) {
        out.setFloat(outOffset + i, ss * x[xOffset + i])
      }
    }

    fun forward(model: Llama, state: LlamaState, token: Int, position: Int): FloatTensor {
      val config = model.configuration
      val weights = model.weights
      val dim = config.embeddingLength
      val sqrtDim = sqrt(dim.toDouble()).toFloat()

      // copy the token embedding into x
      weights!!.tokenEmbeddingTable.copyTo(token * dim, state.x, 0, dim)
      state.x.mapInPlace { v: Float -> v * sqrtDim }

      // Compute per-layer inputs (if model has per-layer embeddings)
      val plDim = config.embeddingLengthPerLayer
      val plTotal = plDim * config.numberOfLayers
      if (plDim > 0 && weights.perLayerTokenEmbd != null) {
        val sqrtPlDim = sqrt(plDim.toDouble()).toFloat()
        val projScale = (1.0 / sqrt(dim.toDouble())).toFloat()
        val inputScale = (1.0 / sqrt(2.0)).toFloat()

        // Project x through perLayerModelProj, scale, and RMS norm per chunk
        weights.perLayerModelProj!!
          .matmul(state.x, state.perLayerInputs!!, plTotal, dim)
        state.perLayerInputs.mapInPlace(0, plTotal) { it * projScale }
        for (l in 0..<config.numberOfLayers) {
          rmsnorm(
            state.perLayerInputs, l * plDim, state.perLayerInputs, l * plDim,
            weights.perLayerProjNorm!!, plDim, config.rmsNormEps
          )
        }

        // Add per-layer token embedding scaled by sqrt(plDim)
        val tokEmbOffset = token.toLong() * plTotal
        for (i in 0..<plTotal) {
          val tokEmb = weights.perLayerTokenEmbd[tokEmbOffset + i] * sqrtPlDim
          state.perLayerInputs.setFloat(i, state.perLayerInputs[i] + tokEmb)
        }

        // Scale combined input by 1/sqrt(2)
        state.perLayerInputs.mapInPlace(0, plTotal) { it * inputScale }
      }

      // forward all the layers
      for (l in 0..<config.numberOfLayers) {
        val layerIsSWA = config.isSWA[l]
        val headSize = config.headSize(l)
        val halfHead = headSize / 2
        val queryDim = config.queryDim(l)
        val kvDim = config.kvDim(l)
        val hiddenDim = config.feedForwardLength[l]

        // attention rmsnorm
        rmsnorm(state.xb, state.x, weights.rmsAttWeight[l], dim, config.rmsNormEps)

        // Q projection + per-head RMS norm
        weights.wq[l].matmul(state.xb, state.q, queryDim, dim)
        for (h in 0..<config.numberOfHeads) {
          rmsnorm(state.q, h * headSize, state.q, h * headSize, weights.attnQNorm[l], headSize, config.rmsNormEps)
        }

        // RoPE (NeoX style: SWA layers use different frequencies than full attention)
        val freqsReal = if (layerIsSWA) weights.freqCisRealSwa else weights.freqCisRealFull
        val freqsImag = if (layerIsSWA) weights.freqCisImagSwa else weights.freqCisImagFull
        for (h in 0..<config.numberOfHeads) {
          val poffset = h * headSize
          var i0 = 0
          while (i0 < headSize) {
            val ic = i0 / 2
            val fcr = freqsReal.get(position * halfHead + ic)
            val fci = freqsImag.get(position * halfHead + ic)
            val v0 = state.q[poffset + ic]
            val v1 = state.q[poffset + ic + halfHead]
            state.q.setFloat(poffset + ic, v0 * fcr - v1 * fci)
            state.q.setFloat(poffset + ic + halfHead, v0 * fci + v1 * fcr)
            i0 += 2
          }
        }

        // KV projection (shared KV: later layers reuse earlier layer's cache)
        val kvLayer = config.kvSourceLayer(l)
        val nKvHeads = config.numberOfKeyValueHeads(l)
        val kvMul = config.numberOfHeads / nKvHeads
        if (config.hasKv(l)) {
          weights.wk[l].matmul(state.xb, state.k, kvDim, dim)
          // V = wv @ xb if V weight exists, otherwise V = K
          if (weights.wv[l] != null) {
            weights.wv[l]!!.matmul(state.xb, state.v, kvDim, dim)
          } else {
            state.k.copyTo(0, state.v, 0, kvDim)
          }

          // Per-head K norm (learned weights) and V norm (bare RMS)
          for (h in 0..<nKvHeads) {
            rmsnorm(state.k, h * headSize, state.k, h * headSize, weights.attnKNorm[l], headSize, config.rmsNormEps)
            rmsnormNoWeight(state.v, h * headSize, state.v, h * headSize, headSize, config.rmsNormEps)
          }

          // RoPE for K
          for (h in 0..<nKvHeads) {
            val poffset = h * headSize
            var i0 = 0
            while (i0 < headSize) {
              val ic = i0 / 2
              val fcr = freqsReal.get(position * halfHead + ic)
              val fci = freqsImag.get(position * halfHead + ic)
              val v0 = state.k[poffset + ic]
              val v1 = state.k[poffset + ic + halfHead]
              state.k.setFloat(poffset + ic, v0 * fcr - v1 * fci)
              state.k.setFloat(poffset + ic + halfHead, v0 * fci + v1 * fcr)
              i0 += 2
            }
          }

          state.k.copyTo(0, state.keyCache[kvLayer], position * kvDim, kvDim)
          state.v.copyTo(0, state.valueCache[kvLayer], position * kvDim, kvDim)
        }

        // Attention (scale=1.0, no 1/sqrt(headSize))
        val attStart = if (layerIsSWA) max(0, position - config.slidingWindow + 1) else 0

        parallelFor(0, config.numberOfHeads) { h: Int ->
          val qOffset = h * headSize
          val attOffset = h * config.contextLength
          for (t in attStart..position) {
            val keyCacheOffset = t * kvDim + (h / kvMul) * headSize
            val score = state.q.dot(qOffset, state.keyCache[kvLayer], keyCacheOffset, headSize)
            state.att.setFloat(attOffset + t, score)
          }

          state.att.softmaxInPlace(attOffset + attStart, position - attStart + 1)
          val xbOffset = h * headSize
          state.xbK.fillInPlace(xbOffset, headSize, 0f)
          for (t in attStart..position) {
            val vOffset = t * kvDim + (h / kvMul) * headSize
            val a = state.att[attOffset + t]
            state.xbK.saxpyInPlace(xbOffset, state.valueCache[kvLayer], vOffset, headSize, a)
          }
        }

        // Output projection + post-attention norm + residual
        weights.wo[l].matmul(state.xbK, state.xb2, dim, queryDim)
        rmsnorm(state.xb2, state.xb2, weights.postAttentionNorm[l], dim, config.rmsNormEps)
        state.x.addInPlace(state.xb2)

        // FFN
        if (config.isMoE) {
          // === MoE FFN: shared MLP + expert MoE ===

          // Shared MLP path: ffn_norm -> gate/up/down -> post_norm_1

          rmsnorm(state.xb, state.x, weights.rmsFfnWeight[l], dim, config.rmsNormEps)
          weights.w1[l].matmul(state.xb, state.hb, hiddenDim, dim)
          weights.w3[l].matmul(state.xb, state.hb2, hiddenDim, dim)
          state.hb.mapInPlace(0, hiddenDim) { gelu(it) }
          state.hb.multiplyInPlace(0, state.hb2, 0, hiddenDim)
          weights.w2[l].matmul(state.hb, state.xb, dim, hiddenDim)
          rmsnorm(
            state.xb,
            state.xb,
            weights.ffnPostNorm1!![l],
            dim,
            config.rmsNormEps
          )

          // state.xb now holds shared MLP output

          // Expert MoE path: pre_norm_2 -> routing -> expert FFN -> post_norm_2
          rmsnorm(
            state.moeInput!!,
            state.x,
            weights.preFfwNorm2!![l],
            dim,
            config.rmsNormEps
          )

          // Router: rms_norm(x) -> scale(1/sqrt(dim)) -> elementwise_mul(gate_inp_s) -> matmul(gate_inp) -> softmax -> top-k
          run {
            var ss = state.x.reduce(0, dim, 0f) { acc, xi -> acc + xi * xi }
            ss /= dim.toFloat()
            ss += config.rmsNormEps
            val rmsScale = (1.0 / sqrt(ss.toDouble())).toFloat() / sqrt(dim.toDouble()).toFloat()
            for (i in 0..<dim) {
              state.xb2.setFloat(
                i,
                state.x[i] * rmsScale * weights.ffnGateInpScale!![l].get(i)
              )
            }
          }
          weights.ffnGateInp!![l]
            .matmul(state.xb2, state.routerLogits!!, config.expertCount, dim)

          // Softmax over router logits
          state.routerLogits.softmaxInPlace(0, config.expertCount)

          // Find top-k experts
          val nExperts = config.expertCount
          val topK = config.expertUsedCount
          val topExperts = IntArray(topK)
          val topProbs = FloatArray(topK)
          for (ki in 0..<topK) {
            var bestIdx = 0
            var bestVal = Float.NEGATIVE_INFINITY
            for (ei in 0..<nExperts) {
              val value = state.routerLogits[ei]
              if (value > bestVal) {
                bestVal = value
                bestIdx = ei
              }
            }
            topExperts[ki] = bestIdx
            topProbs[ki] = bestVal
            state.routerLogits.setFloat(bestIdx, Float.NEGATIVE_INFINITY) // mask for next iteration
          }

          // Run selected experts and accumulate
          val expertFF = config.expertFeedForwardLength
          val gateUpDim = 2 * expertFF
          state.moeOutput!!.fillInPlace(0, dim, 0f)

          for (ki in 0..<topK) {
            val expertIdx = topExperts[ki]
            val prob = topProbs[ki]
            val downScale = weights.ffnDownExpsScale!![l].get(expertIdx)

            // gate_up = ffn_gate_up_exps[expert] @ moeInput -> (2*expertFF,)
            val gateUpOffset = expertIdx * gateUpDim * dim
            weights.ffnGateUpExps!![l].matmul(
              state.moeInput,
              state.expertGateUp!!,
              gateUpDim,
              dim,
              gateUpOffset
            )

            // gate = gelu(gate_up[0:expertFF]), up = gate_up[expertFF:2*expertFF]
            state.expertGateUp.mapInPlace(0, expertFF) { gelu(it) }
            for (i in 0..<expertFF) {
              state.expertGateUp.setFloat(
                i,
                state.expertGateUp[i] * state.expertGateUp[expertFF + i]
              )
            }

            // down = ffn_down_exps[expert] @ (gate * up) -> (dim,)
            val downOffset = expertIdx * dim * expertFF
            weights.ffnDownExps!![l].matmul(
              state.expertGateUp,
              state.expertDown!!,
              dim,
              expertFF,
              downOffset
            )

            // Accumulate: moeOutput += prob * downScale * expertDown
            val finalWeight = prob * downScale
            state.moeOutput.saxpyInPlace(0, state.expertDown, 0, dim, finalWeight)
          }

          // Post-norm for MoE output
          rmsnorm(
            state.moeOutput,
            state.moeOutput,
            weights.ffnPostNorm2!![l],
            dim,
            config.rmsNormEps
          )

          // Combine shared MLP + MoE: xb += moeOutput
          state.xb.addInPlace(0, state.moeOutput, 0, dim)

          // Overall post-FFW norm + residual
          rmsnorm(state.xb, state.xb, weights.postFfwNorm[l], dim, config.rmsNormEps)
          state.x.addInPlace(state.xb)
        } else {
          // Standard dense FFN: w2(GELU(w1(x)) * w3(x))
          rmsnorm(state.xb, state.x, weights.rmsFfnWeight[l], dim, config.rmsNormEps)
          weights.w1[l].matmul(state.xb, state.hb, hiddenDim, dim)
          weights.w3[l].matmul(state.xb, state.hb2, hiddenDim, dim)
          state.hb.mapInPlace(0, hiddenDim) { gelu(it) }
          state.hb.multiplyInPlace(0, state.hb2, 0, hiddenDim)
          weights.w2[l].matmul(state.hb, state.xb, dim, hiddenDim)
          rmsnorm(state.xb, state.xb, weights.postFfwNorm[l], dim, config.rmsNormEps)
          state.x.addInPlace(state.xb)
        }

        // Per-layer embedding: GELU-gated projection
        if (plDim > 0 && weights.perLayerInpGate != null) {
          weights.perLayerInpGate[l].matmul(state.x, state.plGate!!, plDim, dim)
          state.plGate.mapInPlace(0, plDim) { gelu(it) }
          val plOffset = l * plDim
          for (i in 0..<plDim) {
            state.plGate.setFloat(
              i,
              state.plGate[i] * state.perLayerInputs!![plOffset + i]
            )
          }
          weights.perLayerProj!![l].matmul(
            state.plGate,
            state.plProj!!,
            dim,
            plDim
          )
          rmsnorm(
            state.plProj,
            state.plProj,
            weights.perLayerPostNorm!![l],
            dim,
            config.rmsNormEps
          )
          state.x.addInPlace(state.plProj)
        }

        // Layer output scale
        val scale = weights.layerOutputScale[l]
        if (scale != 1.0f) {
          state.x.mapInPlace(0, dim) { it * scale }
        }
      }

      // Final norm + logits
      rmsnorm(state.x, state.x, weights.rmsFinalWeight, dim, config.rmsNormEps)
      weights.wcls.matmul(state.x, state.logits, config.vocabularySize, dim)
      if (config.logitSoftcapping > 0) {
        val cap = config.logitSoftcapping
        state.logits.mapInPlace { cap * tanh((it / cap).toDouble()).toFloat() }
      }
      return state.logits
    }

    private const val ANSI_CYAN = "\u001b[36m"
    private const val ANSI_RESET = "\u001b[0m"

    fun generateTokens(
      model: Llama,
      state: LlamaState,
      startPosition: Int,
      promptTokens: List<Int>,
      stopTokens: Set<Int>,
      maxTokens: Int,
      sampler: Sampler,
      echo: Boolean,
      color: Boolean,
      onTokenGenerated: IntConsumer
    ): MutableList<Int> {
      var maxTokens = maxTokens
      val startNanos = System.nanoTime()
      var startGen: Long = 0
      if (maxTokens < 0 || model.configuration.contextLength < maxTokens) {
        maxTokens = model.configuration.contextLength
      }
      val generatedTokens = ArrayList<Int>(maxTokens)
      var token = state.latestToken // current token (initialized to BOS)
      var nextToken: Int
      var promptIndex = 0
      for (position in startPosition..<maxTokens) {
        forward(model, state, token, position)
        if (promptIndex < promptTokens.size) {
          nextToken = promptTokens[promptIndex++]
          if (echo) {
            System.err.print(replaceControlCharacters(model.tokenizer.decode(listOf(nextToken))))
          }
          if (promptIndex >= promptTokens.size) {
            startGen = System.nanoTime()
          }
        } else {
          nextToken = sampler.sampleToken(state.logits)
          if (echo) {
            System.err.print(replaceControlCharacters(model.tokenizer.decode(listOf(nextToken))))
          }
          generatedTokens.add(nextToken)
          onTokenGenerated.accept(nextToken)
          if (stopTokens.contains(nextToken)) {
            break
          }
        }
        token = nextToken
        state.latestToken = token
      }

      val elapsedNanos = System.nanoTime() - startNanos
      val promptNanos = startGen - startNanos
      val genNanos = elapsedNanos - startGen + startNanos
      val timingPrefix = if (color) ANSI_CYAN else ""
      val timingSuffix = if (color) ANSI_RESET else ""
      val contextUsed = startPosition + promptIndex + generatedTokens.size
      val contextLength = model.configuration.contextLength
      val promptTps = "%.2f".format(promptTokens.size / (promptNanos / 1000000000.0))
      val genTps = "%.2f".format(generatedTokens.size / (genNanos / 1000000000.0))
      System.err.println(
        "\n${timingPrefix}" +
            "context: $contextUsed/$contextLength " +
            "prompt: $promptTps tokens/s (${promptTokens.size}) " +
            "generation: $genTps tokens/s (${generatedTokens.size})" +
            "${timingSuffix}\n"
      )
      return generatedTokens
    }
  }
}
