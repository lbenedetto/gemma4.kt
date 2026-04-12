plugins {
    alias(libs.plugins.kotlin.multiplatform)
}

repositories {
    mavenCentral()
}

val ggmlLibDir = project(":lib").layout.buildDirectory.dir("ggml-build").get().asFile.absolutePath
val ggmlSrcLibDir = "$ggmlLibDir/ggml/src"

kotlin {
    jvmToolchain(25)

    jvm {
        mainRun {
            mainClass = "io.github.lbenedetto.cli.Gemma4Kt"
        }
    }

    if (org.gradle.internal.os.OperatingSystem.current().isMacOsX) {
        macosArm64 {
            binaries {
                executable {
                    entryPoint = "io.github.lbenedetto.cli.main"
                    baseName = "gemma4"
                    linkerOpts("-L$ggmlLibDir", "-L$ggmlSrcLibDir")
                    linkerOpts("-lggml-bridge", "-lggml-cpu", "-lggml-base")
                    linkerOpts("-framework", "Accelerate")
                }
            }
        }
    }
    if (org.gradle.internal.os.OperatingSystem.current().isLinux) {
        linuxX64 {
            binaries {
                executable {
                    entryPoint = "io.github.lbenedetto.cli.main"
                    baseName = "gemma4"
                    linkerOpts("-L$ggmlLibDir", "-L$ggmlSrcLibDir")
                    linkerOpts("-lggml-bridge", "-lggml-cpu", "-lggml-base")
                }
            }
        }
    }

    sourceSets {
        commonMain {
            dependencies {
                implementation(project(":lib"))
                dependencies {
                    implementation(libs.okio)
                }
            }
        }
    }
}

class RunArgumentsProvider(
    @get:Input @get:Optional val runArguments: Property<String>
) : CommandLineArgumentProvider {
    override fun asArguments(): Iterable<String> =
        runArguments.orNull?.split(" ") ?: emptyList()
}

fun Task.forwardRunArguments() {
    (this as Exec).argumentProviders.add(
        RunArgumentsProvider(project.objects.property(String::class.java).apply {
            set(providers.gradleProperty("runArguments"))
        })
    )
}

tasks.matching { it.name == "runDebugExecutableMacosArm64" || it.name == "runDebugExecutableLinuxX64" }.configureEach {
    forwardRunArguments()
    (this as Exec).standardInput = System.`in`
}

tasks.matching { it.name.startsWith("linkDebug") || it.name.startsWith("linkRelease") }.configureEach {
    dependsOn(":lib:compileGgml")
}

tasks.withType<JavaExec> {
    jvmArgs("--add-modules=jdk.incubator.vector", "-Djdk.incubator.vector.VECTOR_ACCESS_OOB_CHECK=0")
    standardInput = System.`in`
}

tasks.register<Jar>("fatJar") {
    group = "build"
    description = "Assembles a fat JAR containing cli and all runtime dependencies"
    archiveFileName.set("gemma4.jar")

    manifest {
        attributes["Main-Class"] = "io.github.lbenedetto.cli.Gemma4Kt"
    }

    val jvmMainCompilation = kotlin.jvm().compilations.getByName("main")
    from(jvmMainCompilation.output.allOutputs)

    dependsOn(jvmMainCompilation.compileTaskProvider)
    from({
        jvmMainCompilation.runtimeDependencyFiles
            .filter { it.name.endsWith("jar") }
            .map { zipTree(it) }
    })

    from(rootProject.file("LICENSE"))

    duplicatesStrategy = DuplicatesStrategy.EXCLUDE
}

tasks.named("assemble") {
    dependsOn("fatJar")
}
