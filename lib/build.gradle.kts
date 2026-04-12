plugins {
    alias(libs.plugins.kotlin.multiplatform)
}

repositories {
    mavenCentral()
}

// Build ggml + bridge as static libraries via CMake
val ggmlBuildDir = layout.buildDirectory.dir("ggml-build")
val ggmlBridgeDir = layout.projectDirectory.dir("ggml-bridge")

// Find cmake: check cmake.dir gradle property, then search PATH + common locations
val cmakeBin: String = providers.gradleProperty("cmake.dir").map { "$it/cmake" }.orElse(
    provider {
        val searchPaths = (System.getenv("PATH") ?: "").split(":") +
            listOf("/opt/homebrew/bin", "/usr/local/bin", "/usr/bin")
        searchPaths.map { File(it, "cmake") }.firstOrNull { it.canExecute() }?.absolutePath
            ?: error("cmake not found. Install cmake or set -Pcmake.dir=/path/to/bin in gradle.properties")
    }
).get()

val buildGgml by tasks.registering(Exec::class) {
    group = "build"
    description = "Build ggml and bridge static libraries via CMake"
    inputs.dir(ggmlBridgeDir)
    inputs.dir(layout.projectDirectory.dir("ggml/src"))
    inputs.dir(layout.projectDirectory.dir("ggml/include"))
    outputs.dir(ggmlBuildDir)
    val buildDir = ggmlBuildDir.get().asFile
    val srcDir = ggmlBridgeDir.asFile
    doFirst { buildDir.mkdirs() }
    workingDir(buildDir)
    commandLine(cmakeBin, "-DCMAKE_BUILD_TYPE=Release", srcDir.absolutePath)
}

val compileGgml by tasks.registering(Exec::class) {
    group = "build"
    description = "Compile ggml and bridge static libraries"
    dependsOn(buildGgml)
    val buildDir = ggmlBuildDir.get().asFile
    workingDir(buildDir)
    commandLine(cmakeBin, "--build", ".", "--config", "Release", "-j")
}

kotlin {
    jvmToolchain(25)

    jvm()

    val ggmlIncludeDir = ggmlBridgeDir.asFile.absolutePath
    val ggmlLibDir = ggmlBuildDir.get().asFile.absolutePath
    val ggmlSrcLibDir = "${ggmlLibDir}/ggml/src"

    fun org.jetbrains.kotlin.gradle.plugin.mpp.KotlinNativeTarget.configureGgmlCinterop() {
        compilations.getByName("main") {
            cinterops {
                val ggml by creating {
                    defFile(file("src/nativeInterop/cinterop/ggml.def"))
                    packageName("ggml.bridge")
                    includeDirs(ggmlIncludeDir)
                }
            }
        }
        binaries.all {
            linkerOpts("-L$ggmlLibDir", "-L$ggmlSrcLibDir")
            linkerOpts("-lggml-bridge", "-lggml-cpu", "-lggml-base")
            linkerOpts("-lstdc++")
        }
    }

    macosArm64 {
        configureGgmlCinterop()
        binaries.all {
            linkerOpts("-framework", "Accelerate")
        }
    }
    linuxX64 {
        configureGgmlCinterop()
    }

    compilerOptions {
        freeCompilerArgs.add("-Xexpect-actual-classes")
    }

    sourceSets {
        commonMain {
            kotlin.srcDir("src/commonMain/kotlin")
            dependencies {
                implementation(libs.okio)
            }
        }
        nativeMain {
            kotlin.srcDir("src/nativeMain/kotlin")
            dependencies {
                implementation(libs.coroutines.core)
            }
        }
        jvmMain {
            kotlin.srcDir("src/jvmMain/kotlin")
            kotlin.srcDir("src/jvmMain/java")
        }
        jvmTest {
            kotlin.srcDir("src/jvmTest/kotlin")
            dependencies {
                implementation(libs.kotest.runner.junit5)
                implementation(libs.kotest.assertions.core)
            }
        }
    }
}

// Compiled ggml libraries are only needed at link time, not for cinterop stub generation.
// This avoids requiring cmake during IDE sync.
tasks.matching { it.name.startsWith("linkDebug") || it.name.startsWith("linkRelease") }.configureEach {
    dependsOn(compileGgml)
}

tasks.withType<JavaCompile>().configureEach {
    options.compilerArgs.add("--add-modules=jdk.incubator.vector")
}

tasks.named<JavaCompile>("compileJvmMainJava") {
    val kotlinCompile = tasks.named<org.jetbrains.kotlin.gradle.tasks.KotlinJvmCompile>("compileKotlinJvm")
    dependsOn(kotlinCompile)
    val kotlinOutputDir = kotlinCompile.flatMap { it.destinationDirectory }
    options.compilerArgumentProviders.add(CommandLineArgumentProvider {
        listOf("--patch-module", "io.github.lbenedetto.gemma4kt=${kotlinOutputDir.get().asFile}")
    })
}

tasks.named<Test>("jvmTest") {
    useJUnitPlatform()
    jvmArgs("--add-modules=jdk.incubator.vector", "-Xmx16g")
}
