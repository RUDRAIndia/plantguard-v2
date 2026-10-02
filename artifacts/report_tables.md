# PlantGuard Report Tables

_Figure 1 from results.json (stage 1: four models, PlantVillage only). All other figures and tables from results_finetuned.json (stage 2: selected model fine-tuned on PlantDoc train), commit `9138f32bf6ba9100f530f326c1fa5ba2e99c3334`._

## Table 1: Model comparison

| Model | Val macro-F1 | Params | Checkpoint size |
|---|---|---|---|
| MobileNetV3Large | 0.9678 | 3,032,870 | 29.0 MB |
| MobileNetV3Small | 0.9538 | 961,046 | 9.9 MB |
| EfficientNetV2B0 | 0.9064 | 5,967,990 | 65.4 MB |
| MobileNetV2 | 0.8382 | 2,306,662 | 21.3 MB |

## Table 2: PlantVillage test set vs PlantDoc field test split

_PlantDoc figures are its official 236-image test split only. The model was fine-tuned on PlantDoc's train split, so that portion is no longer external and is excluded._

| Metric | PlantVillage (test) | PlantDoc (field) |
|---|---|---|
| Accuracy | 0.9823 | 0.5720 |
| Macro-F1 | 0.9791 | 0.5444 |

## Table 3: PlantDoc mapping confidence tiers

| Tier | macro-F1 | Num images |
|---|---|---|
| Exact | 0.5993 | 119 |
| Convention | 0.6429 | 90 |
| Forced | 0.5101 | 27 |

## Table 4: OOD threshold sweep (selected rows)

| Threshold | PlantDoc confident-wrong % | Val false-rejection % |
|---|---|---|
| 0.00 | 42.8% | 0.0% |
| 0.90 | 13.1% | 3.2% |
| 1.00 | 0.0% | 65.2% |

## Table 5: Robustness under corruption (macro-F1)

| Corruption | Severity 1 | Severity 2 | Severity 3 |
|---|---|---|---|
| Blur | 0.9830 | 0.9684 | 0.9457 |
| Gaussian Noise | 0.9632 | 0.9165 | 0.3377 |
| Brightness Up | 0.9792 | 0.9790 | 0.9732 |
| Brightness Down | 0.9808 | 0.9755 | 0.9373 |
| Rotation | 0.9831 | 0.9831 | 0.9790 |
| Jpeg Compression | 0.9693 | 0.9744 | 0.8751 |

## Table 6: Top-10 confused pairs

| True class | Predicted as | Count | Rate of true class |
|---|---|---|---|
| Tomato___Spider_mites Two-spotted_spider_mite | Tomato___Target_Spot | 28 | 11.2% |
| Corn_(maize)___Northern_Leaf_Blight | Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot | 8 | 5.4% |
| Tomato___Early_blight | Tomato___Target_Spot | 8 | 5.3% |
| Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot | Corn_(maize)___Northern_Leaf_Blight | 7 | 9.1% |
| Tomato___Tomato_Yellow_Leaf_Curl_Virus | Tomato___Bacterial_spot | 7 | 0.9% |
| Tomato___Early_blight | Tomato___Late_blight | 5 | 3.3% |
| Tomato___Early_blight | Tomato___Septoria_leaf_spot | 4 | 2.7% |
| Tomato___Late_blight | Potato___Late_blight | 4 | 1.4% |
| Tomato___Septoria_leaf_spot | Tomato___Bacterial_spot | 4 | 1.5% |
| Corn_(maize)___Common_rust_ | Corn_(maize)___Northern_Leaf_Blight | 3 | 1.7% |

## OOD rejection at deployed threshold (0.95)

- Deployed threshold: 0.95
- PlantDoc rejection rate: 0.5424
- PlantDoc accepted: 108
- Accuracy on accepted: 0.8148
- Confident-wrong: 20 (8.5% of PlantDoc test)

## Confidence bands (PlantDoc test)

| Band | Images | Share | Top-1 | Top-3 |
|---|---|---|---|---|
| confident | 108 | 45.8% | 0.8148 | 0.9444 |
| possible | 97 | 41.1% | 0.4124 | 0.7526 |
| unrecognised | 31 | 13.1% | 0.2258 | 0.6129 |

## Before vs after fine-tuning

| Metric | Before | After |
|---|---|---|
| PlantVillage test accuracy | 0.9659 | 0.9823 |
| PlantVillage test macro-F1 | 0.9613 | 0.9791 |
| PlantDoc test accuracy | 0.2585 | 0.5720 |
| PlantDoc test macro-F1 | 0.2350 | 0.5444 |
| OOD threshold | 0.98 | 0.95 |

PlantDoc accuracy 95% CI: [0.5085, 0.6356] (2000 bootstrap resamples)
