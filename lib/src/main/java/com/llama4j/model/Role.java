package com.llama4j.model;

public record Role(String name) {
  public static Role SYSTEM = new Role("system");
  public static Role USER = new Role("user");
  public static Role MODEL = new Role("model");

  @Override
  public String toString() {
    return name;
  }
}
