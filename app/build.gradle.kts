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

tasks.register<Jar>("fatJar") {
    group = "build"
    description = "Assembles a fat JAR containing app and all runtime dependencies"
    archiveFileName.set("gemma4.jar")

    manifest {
        attributes["Main-Class"] = "com.llama4j.Gemma4"
    }

    from(sourceSets.main.get().output)

    dependsOn(configurations.runtimeClasspath)
    from({
        configurations.runtimeClasspath.get()
            .filter { it.name.endsWith("jar") }
            .map { zipTree(it) }
    })

    from(rootProject.file("LICENSE"))

    duplicatesStrategy = DuplicatesStrategy.EXCLUDE
}

tasks.named("assemble") {
    dependsOn("fatJar")
}
