from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class VerificationOutput:
    verdict: bool
    output: Dict[str, Any]
    system_error: bool = (
        False  # If verification failed due to some error outside of verification
    )
    # ^ This is not perfectly supported right now, for local verification it is
