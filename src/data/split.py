"""Creates train/val/test splits from deduped PlantVillage data.
Asserts TRAIN + VAL + TEST == 1.0 and that class count == 38 before writing
any split manifest. Splits are leak-free with respect to dedupe.py output.
"""
