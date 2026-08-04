"""Computes primary classification metrics: macro-F1, per-class recall,
worst-class recall, and accuracy (secondary, never reported alone).
No regression metrics (R2, RMSE, etc.) are computed here or anywhere else.
Writes results to a metrics JSON; nothing downstream may hand-write a number.
"""
