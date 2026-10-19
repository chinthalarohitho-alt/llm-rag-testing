import os

# Local models are slower than cloud APIs: give each judge call more time.
os.environ.setdefault("DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS_OVERRIDE", "300")
os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "1")
