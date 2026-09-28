package com.plantguard.app

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import com.plantguard.app.ui.theme.PlantGuardTheme

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        // Draws the app behind the status and navigation bars so the cream (or
        // near-black) background runs to the edges of the screen, and keeps the
        // system icons legible against it. Every screen that needs to stay clear
        // of the bars asks for that with its own insets padding.
        enableEdgeToEdge()
        super.onCreate(savedInstanceState)
        setContent {
            PlantGuardTheme {
                PlantGuardApp()
            }
        }
    }
}
