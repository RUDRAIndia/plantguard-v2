// Top-level build file: declares plugin versions once (via the version
// catalog in gradle/libs.versions.toml) without applying them to the root
// project itself — each module (just :app here) applies what it needs.
//
// No org.jetbrains.kotlin.android plugin here: AGP 9+ has built-in Kotlin
// support enabled by default, so applying the classic Kotlin Android
// plugin on top now fails the build with "Cannot add extension with name
// 'kotlin', as there is an extension already registered with that name."
// The Compose compiler plugin (kotlin.compose) is still applied explicitly
// below/in app/build.gradle.kts, pinned to an exact Kotlin version, which
// overrides whichever Compose-compiler version AGP's built-in support
// would otherwise pick automatically.
plugins {
    alias(libs.plugins.android.application) apply false
    alias(libs.plugins.kotlin.compose) apply false
    alias(libs.plugins.ksp) apply false
}
