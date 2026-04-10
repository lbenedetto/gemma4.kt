package com.llama4j.internal.model

@JvmRecord
data class Role(val name: String) {
  override fun toString(): String {
    return name
  }

  companion object {
    var SYSTEM: Role = Role("system")
    var USER: Role = Role("user")
    var MODEL: Role = Role("model")
  }
}
