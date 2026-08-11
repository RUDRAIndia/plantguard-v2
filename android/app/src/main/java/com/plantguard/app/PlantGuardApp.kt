package com.plantguard.app

import androidx.compose.runtime.Composable
import androidx.compose.runtime.remember
import androidx.compose.ui.platform.LocalContext
import androidx.navigation.NavType
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.rememberNavController
import androidx.navigation.navArgument
import com.plantguard.app.data.prefs.DisclaimerPrefs
import com.plantguard.app.navigation.NavRoutes
import com.plantguard.app.ui.camera.CameraScreen
import com.plantguard.app.ui.disclaimer.DisclaimerScreen
import com.plantguard.app.ui.history.HistoryScreen
import com.plantguard.app.ui.result.ResultScreen

@Composable
fun PlantGuardApp() {
    val context = LocalContext.current
    val disclaimerPrefs = remember { DisclaimerPrefs(context) }
    val navController = rememberNavController()

    val startDestination =
        if (disclaimerPrefs.hasSeenDisclaimer()) NavRoutes.CAMERA else NavRoutes.DISCLAIMER

    NavHost(navController = navController, startDestination = startDestination) {
        composable(NavRoutes.DISCLAIMER) {
            DisclaimerScreen(
                onContinue = {
                    disclaimerPrefs.setSeen()
                    navController.navigate(NavRoutes.CAMERA) {
                        popUpTo(NavRoutes.DISCLAIMER) { inclusive = true }
                    }
                },
            )
        }

        composable(NavRoutes.CAMERA) {
            CameraScreen(
                onNavigateToResult = { entryId -> navController.navigate(NavRoutes.resultRoute(entryId)) },
                onNavigateToHistory = { navController.navigate(NavRoutes.HISTORY) },
            )
        }

        composable(
            route = NavRoutes.RESULT,
            arguments = listOf(navArgument(NavRoutes.RESULT_ARG_ID) { type = NavType.LongType }),
        ) {
            ResultScreen(
                onRetakePhoto = { navController.popBackStack(NavRoutes.CAMERA, inclusive = false) },
            )
        }

        composable(NavRoutes.HISTORY) {
            HistoryScreen(
                onOpenResult = { entryId -> navController.navigate(NavRoutes.resultRoute(entryId)) },
            )
        }
    }
}
