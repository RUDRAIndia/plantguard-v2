package com.plantguard.app.ui.disease

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.lifecycle.viewmodel.compose.viewModel
import com.plantguard.app.R
import com.plantguard.app.ui.components.FootnoteText
import com.plantguard.app.ui.components.KvkAdviceCard
import com.plantguard.app.ui.components.PlantGuardTopBar
import com.plantguard.app.ui.components.SectionCard

/**
 * One condition in full, reached by tapping a candidate in the "possible matches"
 * tier.
 *
 * It carries the KVK line as well, because a user can arrive here and read
 * management advice without the result screen still in front of them — the
 * instruction to confirm has to travel with the advice.
 */
@Composable
fun DiseaseDetailScreen(
    onNavigateUp: () -> Unit,
    viewModel: DiseaseDetailViewModel = viewModel(),
) {
    val uiState by viewModel.uiState.collectAsStateWithLifecycle()

    Scaffold(
        topBar = {
            PlantGuardTopBar(
                title = stringResource(R.string.disease_detail_title),
                onNavigateUp = onNavigateUp,
            )
        },
    ) { innerPadding ->
        Box(
            modifier = Modifier
                .fillMaxSize()
                .padding(innerPadding),
        ) {
            when (val state = uiState) {
                is DiseaseDetailUiState.Loading -> Box(
                    modifier = Modifier.fillMaxSize(),
                    contentAlignment = Alignment.Center,
                ) {
                    CircularProgressIndicator()
                }

                is DiseaseDetailUiState.Error -> Box(
                    modifier = Modifier
                        .fillMaxSize()
                        .padding(24.dp),
                    contentAlignment = Alignment.Center,
                ) {
                    Text(
                        text = state.message,
                        style = MaterialTheme.typography.bodyLarge,
                        textAlign = TextAlign.Center,
                    )
                }

                is DiseaseDetailUiState.Loaded -> Column(
                    modifier = Modifier
                        .fillMaxSize()
                        .verticalScroll(rememberScrollState())
                        .padding(horizontal = 20.dp),
                    verticalArrangement = Arrangement.spacedBy(16.dp),
                ) {
                    Spacer(Modifier.height(8.dp))
                    Text(
                        text = state.displayName,
                        style = MaterialTheme.typography.headlineMedium,
                        modifier = Modifier.fillMaxWidth(),
                    )
                    state.info?.let { info ->
                        SectionCard(
                            title = stringResource(R.string.result_section_symptoms),
                            body = info.symptoms,
                        )
                        SectionCard(
                            title = stringResource(R.string.result_section_management),
                            body = info.management,
                        )
                        FootnoteText(text = stringResource(R.string.result_source, info.citation))
                    }
                    KvkAdviceCard()
                    Spacer(Modifier.height(16.dp))
                }
            }
        }
    }
}
