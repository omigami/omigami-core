from dataclasses import dataclass


@dataclass
class FeatureSelectionResult:
    name: str
    rank: float
