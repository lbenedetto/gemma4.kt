abstract class DownloadModelTask @Inject constructor(
  private val execOperations: ExecOperations
) : DefaultTask() {

  @get:Input
  abstract val url: Property<String>

  @get:OutputFile
  abstract val destination: RegularFileProperty

  @TaskAction
  fun download() {
    val destUrl = url.get()
    require(destUrl.isNotEmpty()) { "Pass -Pmodel.url=<url>" }
    destination.get().asFile.parentFile.mkdirs()
    execOperations.exec {
      commandLine("curl", "-L", "-o",
        destination.get().asFile.absolutePath, destUrl)
    }
  }
}

val hfBase = "https://huggingface.co/unsloth"
val quant = providers.gradleProperty("model.quant").getOrElse("Q8_0")
val modelsDir = rootProject.layout.projectDirectory.dir("models")

fun registerDownloadTask(name: String, repo: String, fileName: String) {
  tasks.register<DownloadModelTask>(name) {
    group = "models"
    description = "Download $fileName into models/"
    url.set("$hfBase/$repo/resolve/main/$fileName")
    destination.set(modelsDir.file(fileName))
  }
}

registerDownloadTask("downloadE2B",  "gemma-4-E2B-it-GGUF",    "gemma-4-E2B-it-$quant.gguf")
registerDownloadTask("downloadE4B",  "gemma-4-E4B-it-GGUF",    "gemma-4-E4B-it-$quant.gguf")
registerDownloadTask("downloadE31B", "gemma-4-31B-it-GGUF",    "gemma-4-31B-it-$quant.gguf")
registerDownloadTask("downloadE26B", "gemma-4-26B-A4B-it-GGUF","gemma-4-26B-A4B-it-$quant.gguf")

tasks.register<DownloadModelTask>("downloadModel") {
  group = "models"
  description = "Download any model: ./gradlew downloadModel -Pmodel.url=<url>"
  url.set(providers.gradleProperty("model.url").orElse(""))
  destination.set(
    url.map { u -> modelsDir.file(u.substringAfterLast("/").ifEmpty { ".unused" }) }
  )
}
