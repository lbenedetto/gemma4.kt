package io.github.lbenedetto.api

import io.kotest.assertions.throwables.shouldThrow
import io.kotest.core.spec.style.BehaviorSpec
import io.kotest.matchers.nulls.shouldNotBeNull
import io.kotest.matchers.shouldBe
import io.kotest.matchers.shouldNotBe
import io.kotest.matchers.string.shouldNotBeBlank
import okio.Path.Companion.toPath

class GemmaModelIntegrationSpec : BehaviorSpec({

    val model = GemmaModel.load("../models/gemma-4-E2B-it-Q8_0.gguf".toPath(), contextLength = 2048)

    Given("a loaded GemmaModel") {

        When("generating a response to a simple prompt") {
            val result = model.generate("Say hello and nothing else.") {
                maxTokens = 200
                seed = 42L
            }
            println(result.text)
            Then("result text is non-blank") { result.text.shouldNotBeBlank() }
            Then("thinking is null by default") { result.thinking shouldBe null }
        }

        When("generating with a system prompt") {
            val result = model.generate("What is your job? Answer in less than 10 words.") {
                systemPrompt = "You are a pirate. Always respond in pirate speak."
                maxTokens = 100
                seed = 42L
            }
            println(result.text)
            Then("result text is non-blank") { result.text.shouldNotBeBlank() }
        }

        When("generating with streaming enabled") {
            val streamed = StringBuilder()
            val result = model.generate("Write a haiku about the ocean. Output nothing else, keep it short.") {
                maxTokens = 100
                seed = 42L
                onToken = { piece -> streamed.append(piece) }
            }
            println(result.text)
            Then("streamed text matches result.text") { streamed.toString() shouldBe result.text }
        }

        When("generating with thinking enabled") {
            val result = model.generate("What is 2 × 3?") {
                thinking = true
                maxTokens = 512
                seed = 42L
            }
            println("text: ${result.text}")
            println("thinking: ${result.thinking}")
            Then("result text is non-blank") { result.text.shouldNotBeBlank() }
            Then("thinking content is present") { result.thinking.shouldNotBeNull() }
        }

        When("running a multi-turn chat session") {
            val chat = model.chat {
                systemPrompt = "You are a helpful assistant."
                maxTokens = 200
                seed = 42L
            }
            val r1 = chat.send("My name is Claude.")
            Then("first response is non-blank") { r1.text.shouldNotBeBlank() }
            val contextAfterFirst = chat.contextUsed

            val r2 = chat.send("What is my name?")
            Then("second response is non-blank") { r2.text.shouldNotBeBlank() }

            val contextAfterSecond = chat.contextUsed
            Then("context grows after each turn") { contextAfterSecond shouldNotBe contextAfterFirst }

            chat.reset()
            Then("reset reduces context") { chat.contextUsed shouldNotBe contextAfterSecond }
        }

        When("completing fill-in-the-middle on a model without FIM support") {
            Then("throws IllegalArgumentException") {
                shouldThrow<IllegalArgumentException> {
                    model.fillInMiddle(
                        prefix = "fun greet(name: String) { ",
                        suffix = "}",
                    ) { maxTokens = 50 }
                }
            }
        }
    }
})
