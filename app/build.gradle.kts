plugins {
    alias(libs.plugins.kotlin.jvm)
    application
}

repositories {
    mavenCentral()
}

dependencies {
    implementation(project(":lib"))
}

application {
    mainClass = "com.llama4j.Gemma4"
}

java {
    toolchain {
        languageVersion = JavaLanguageVersion.of(25)
    }
}

tasks.withType<JavaExec> {
    jvmArgs("--add-modules=jdk.incubator.vector", "-Djdk.incubator.vector.VECTOR_ACCESS_OOB_CHECK=0")
}
