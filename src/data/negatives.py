"""Handles non-plant / out-of-distribution negative examples so the model
has a defined behavior for inputs that are not leaves at all, rather than
forcing a confident wrong prediction into one of the 38 classes.
"""
