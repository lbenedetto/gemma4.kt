plugins {
    alias(libs.plugins.kotlin.jvm)
}

repositories {
    mavenCentral()
}

dependencies {
    testImplementation(libs.kotest.runner.junit5)
    testImplementation(libs.kotest.assertions.core)
}

tasks.test {
    useJUnitPlatform()
    jvmArgs("--add-modules=jdk.incubator.vector", "-Xmx16g")
}

java {
    toolchain {
        languageVersion = JavaLanguageVersion.of(25)
    }
}

tasks.withType<JavaCompile>().configureEach {
    options.compilerArgs.add("--add-modules=jdk.incubator.vector")
}

tasks.named<JavaCompile>("compileJava") {
    val kotlinCompile = tasks.named<org.jetbrains.kotlin.gradle.tasks.KotlinJvmCompile>("compileKotlin")
    dependsOn(kotlinCompile)
    val kotlinOutputDir = kotlinCompile.flatMap { it.destinationDirectory }
    options.compilerArgumentProviders.add(org.gradle.process.CommandLineArgumentProvider {
        listOf("--patch-module", "io.github.lbenedetto.gemma4kt=${kotlinOutputDir.get().asFile}")
    })
}
