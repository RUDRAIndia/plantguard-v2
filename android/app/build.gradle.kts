// No org.jetbrains.kotlin.android plugin: AGP 9+ compiles Kotlin sources
// itself via built-in Kotlin support (see root build.gradle.kts comment).
plugins {
    alias(libs.plugins.android.application)
    alias(libs.plugins.kotlin.compose)
    alias(libs.plugins.ksp)
}

android {
    namespace = "com.plantguard.app"
    compileSdk = 37

    defaultConfig {
        applicationId = "com.plantguard.app"
        minSdk = 24
        targetSdk = 37
        versionCode = 1
        versionName = "0.1.0-placeholder"

        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            proguardFiles(getDefaultProguardFile("proguard-android-optimize.txt"), "proguard-rules.pro")
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    // Kotlin's jvmTarget follows compileOptions.targetCompatibility above
    // automatically under AGP's built-in Kotlin support — no separate
    // kotlinOptions {} block needed (that DSL belonged to the classic
    // org.jetbrains.kotlin.android plugin, which this project no longer applies).

    buildFeatures {
        compose = true
    }

    // The placeholder model.tflite and its metadata are real, committed
    // assets (see android/README.md) — nothing here should compress or
    // exclude them.
    androidResources {
        noCompress += "tflite"
    }

    packaging {
        resources {
            excludes += "/META-INF/{AL2.0,LGPL2.1}"
        }
    }
}

dependencies {
    implementation(libs.androidx.core.ktx)
    implementation(libs.androidx.activity.compose)
    implementation(libs.androidx.lifecycle.runtime.ktx)
    implementation(libs.androidx.lifecycle.runtime.compose)
    implementation(libs.androidx.lifecycle.viewmodel.compose)

    implementation(platform(libs.androidx.compose.bom))
    implementation(libs.androidx.compose.ui)
    implementation(libs.androidx.compose.ui.graphics)
    implementation(libs.androidx.compose.ui.tooling.preview)
    implementation(libs.androidx.compose.material3)
    implementation(libs.androidx.compose.material.icons.extended)
    debugImplementation(libs.androidx.compose.ui.tooling)

    implementation(libs.androidx.navigation.compose)

    implementation(libs.androidx.room.runtime)
    implementation(libs.androidx.room.ktx)
    ksp(libs.androidx.room.compiler)

    implementation(libs.androidx.camera.core)
    implementation(libs.androidx.camera.camera2)
    implementation(libs.androidx.camera.lifecycle)
    implementation(libs.androidx.camera.view)

    implementation(libs.androidx.exifinterface)

    implementation(libs.litert)

    testImplementation(libs.junit)
    androidTestImplementation(libs.androidx.junit)
}
