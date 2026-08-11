"""Bounded-retry helper for the download-after-upload race described in
src/models/kaggle_persist.py's module docstring: Kaggle ingests a new
dataset version asynchronously, so a `dataset_download_files`/
`dataset_download_file` call moments after an upload can 404 even though
that upload fully succeeded. Split out from kaggle_persist.py to stay under
CLAUDE.md rule 12's ~300-line guideline — the same reason
kaggle_persist_report.py was already split out.
"""

import time

import requests

# 5 attempts at 2/4/8/16s backoff is ~30s of waiting in the worst case,
# comfortably under the 60s total cap -- long enough to ride out Kaggle's
# async version-ingestion delay without hanging a session for minutes if
# something is genuinely broken.
MAX_ATTEMPTS = 5
BASE_BACKOFF_SECONDS = 2.0
MAX_TOTAL_WAIT_SECONDS = 60.0


def is_transient_kaggle_error(exc: Exception) -> bool:
    """404 (the dataset version this call is about to download can still be
    mid-ingestion right after an upload) and 5xx/network errors are worth
    retrying. 401/403 auth failures, 400 bad requests, etc. are not --
    retrying those would just waste the backoff budget on something that
    will never succeed.
    """
    if isinstance(exc, requests.exceptions.HTTPError):
        status = exc.response.status_code if exc.response is not None else None
        return status == 404 or (status is not None and 500 <= status < 600)
    return isinstance(exc, (requests.exceptions.ConnectionError, requests.exceptions.Timeout))


def call_with_retry(func, *, description: str):
    """Calls func() with bounded exponential-backoff retry on transient
    errors (see is_transient_kaggle_error). Re-raises the last exception
    once attempts or the total-wait budget are exhausted, or immediately
    for a non-transient error -- callers are responsible for turning that
    into a clearly-worded error (see kaggle_persist.py's module docstring
    on why a download failure must never be worded like an upload failure).
    """
    total_waited = 0.0
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            return func()
        except Exception as exc:
            if not is_transient_kaggle_error(exc):
                raise
            if attempt == MAX_ATTEMPTS or total_waited >= MAX_TOTAL_WAIT_SECONDS:
                raise
            backoff = min(
                BASE_BACKOFF_SECONDS * (2 ** (attempt - 1)),
                MAX_TOTAL_WAIT_SECONDS - total_waited,
            )
            print(
                f"[kaggle_retry] {description} failed on attempt "
                f"{attempt}/{MAX_ATTEMPTS} ({exc}); retrying in {backoff:.1f}s "
                "(Kaggle dataset-version ingestion is asynchronous, so a 404 "
                "right after an upload is expected transient behavior)..."
            )
            time.sleep(backoff)
            total_waited += backoff
