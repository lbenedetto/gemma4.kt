plugins {
    alias(libs.plugins.kotlin.multiplatform)
}

repositories {
    mavenCentral()
}

kotlin {
    jvmToolchain(25)

    jvm()
    macosArm64()
    linuxX64()

    compilerOptions {
        freeCompilerArgs.add("-Xexpect-actual-classes")
    }

    sourceSets {
        commonMain {
            kotlin.srcDir("src/commonMain/kotlin")
        }
        nativeMain {
            kotlin.srcDir("src/nativeMain/kotlin")
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
