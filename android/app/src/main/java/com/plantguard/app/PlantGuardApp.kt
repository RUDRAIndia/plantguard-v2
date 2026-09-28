package com.plantguard.app

import androidx.compose.animation.AnimatedContentTransitionScope
import androidx.compose.animation.core.tween
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.animation.slideInHorizontally
import androidx.compose.animation.slideOutHorizontally
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
import com.plantguard.app.ui.about.AboutScreen
import com.plantguard.app.ui.camera.CameraScreen
import com.plantguard.app.ui.disclaimer.DisclaimerScreen
import com.plantguard.app.ui.disease.DiseaseDetailScreen
import com.plantguard.app.ui.history.HistoryScreen
import com.plantguard.app.ui.home.HomeScreen
import com.plantguard.app.ui.result.ResultScreen
import com.plantguard.app.ui.tips.PhotoTipsScreen

/** Duration of the screen-to-screen slide, in milliseconds. */
private const val NAV_ANIM_MS = 300

@Composable
fun PlantGuardApp() {
    val context = LocalContext.current
    val disclaimerPrefs = remember { DisclaimerPrefs(context) }
    val navController = rememberNavController()

    // The disclaimer is a one-time gate: the accuracy limits are stated before
    // the user's first photo, not buried in About. Afterwards the app opens on
    // Home rather than jumping straight into the camera, so the other three
    // things it can do are discoverable at all.
    val startDestination =
        if (disclaimerPrefs.hasSeenDisclaimer()) NavRoutes.HOME else NavRoutes.DISCLAIMER

    NavHost(
        navController = navController,
        startDestination = startDestination,
        // Set once on the host rather than per destination: going deeper slides in
        // from the right, coming back reverses it, which is what makes the app feel
        // like a stack of screens instead of a set of unrelated views.
        enterTransition = {
            slideIntoContainer(
                AnimatedContentTransitionScope.SlideDirection.Start,
                animationSpec = tween(NAV_ANIM_MS),
            ) + fadeIn(tween(NAV_ANIM_MS))
        },
        exitTransition = {
            slideOutHorizontally(tween(NAV_ANIM_MS)) { fullWidth -> -fullWidth / 4 } +
                fadeOut(tween(NAV_ANIM_MS))
        },
        popEnterTransition = {
            slideInHorizontally(tween(NAV_ANIM_MS)) { fullWidth -> -fullWidth / 4 } +
                fadeIn(tween(NAV_ANIM_MS))
        },
        popExitTransition = {
            slideOutOfContainer(
                AnimatedContentTransitionScope.SlideDirection.End,
                animationSpec = tween(NAV_ANIM_MS),
            ) + fadeOut(tween(NAV_ANIM_MS))
        },
    ) {
        composable(NavRoutes.DISCLAIMER) {
            DisclaimerScreen(
                onContinue = {
                    disclaimerPrefs.setSeen()
                    navController.navigate(NavRoutes.HOME) {
                        // inclusive: the gate must not be reachable with the back
                        // button once accepted.
                        popUpTo(NavRoutes.DISCLAIMER) { inclusive = true }
                    }
                },
            )
        }

        composable(NavRoutes.HOME) {
            HomeScreen(
                onIdentify = { navController.navigate(NavRoutes.CAMERA) },
                onHistory = { navController.navigate(NavRoutes.HISTORY) },
                onAbout = { navController.navigate(NavRoutes.ABOUT) },
                onPhotoTips = { navController.navigate(NavRoutes.PHOTO_TIPS) },
            )
        }

        composable(NavRoutes.CAMERA) {
            CameraScreen(
                onNavigateToResult = { entryId ->
                    navController.navigate(NavRoutes.resultRoute(entryId)) {
                        // The camera screen is not worth returning to from a
                        // result — "take another photo" should open a fresh
                        // camera, not restore the one that already fired.
                        popUpTo(NavRoutes.CAMERA) { inclusive = true }
                    }
                },
                onNavigateUp = { navController.popBackStack() },
                onOpenPhotoTips = { navController.navigate(NavRoutes.PHOTO_TIPS) },
            )
        }

        composable(
            route = NavRoutes.RESULT,
            arguments = listOf(navArgument(NavRoutes.RESULT_ARG_ID) { type = NavType.LongType }),
        ) {
            ResultScreen(
                // popBackStack, not navigate(CAMERA): a result reached from
                // History must go back to History. The previous version always
                // popped to the camera, which silently threw the user out of the
                // history list they were browsing.
                onNavigateUp = { navController.popBackStack() },
                onTakeAnotherPhoto = {
                    navController.navigate(NavRoutes.CAMERA) {
                        launchSingleTop = true
                    }
                },
                onOpenDisease = { classIndex ->
                    navController.navigate(NavRoutes.diseaseRoute(classIndex))
                },
                onOpenPhotoTips = { navController.navigate(NavRoutes.PHOTO_TIPS) },
            )
        }

        composable(
            route = NavRoutes.DISEASE,
            arguments = listOf(navArgument(NavRoutes.DISEASE_ARG_INDEX) { type = NavType.IntType }),
        ) {
            DiseaseDetailScreen(onNavigateUp = { navController.popBackStack() })
        }

        composable(NavRoutes.HISTORY) {
            HistoryScreen(
                onOpenResult = { entryId -> navController.navigate(NavRoutes.resultRoute(entryId)) },
                onNavigateUp = { navController.popBackStack() },
            )
        }

        composable(NavRoutes.ABOUT) {
            AboutScreen(onNavigateUp = { navController.popBackStack() })
        }

        composable(NavRoutes.PHOTO_TIPS) {
            PhotoTipsScreen(onNavigateUp = { navController.popBackStack() })
        }
    }
}
