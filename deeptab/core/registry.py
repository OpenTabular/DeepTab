from dataclasses import dataclass
from typing import Literal

ModelStatus = Literal["stable", "experimental"]


@dataclass(frozen=True)
class ModelInfo:
    name: str
    status: ModelStatus
    import_path: str


MODEL_REGISTRY: dict[str, ModelInfo] = {
    "Mambular": ModelInfo("Mambular", "stable", "deeptab.models"),
    "TabM": ModelInfo("TabM", "stable", "deeptab.models"),
    "NODE": ModelInfo("NODE", "stable", "deeptab.models"),
    "ENODE": ModelInfo("ENODE", "stable", "deeptab.models"),
    "FTTransformer": ModelInfo("FTTransformer", "stable", "deeptab.models"),
    "MLP": ModelInfo("MLP", "stable", "deeptab.models"),
    "ResNet": ModelInfo("ResNet", "stable", "deeptab.models"),
    "TabTransformer": ModelInfo("TabTransformer", "stable", "deeptab.models"),
    "MambaTab": ModelInfo("MambaTab", "stable", "deeptab.models"),
    "TabulaRNN": ModelInfo("TabulaRNN", "stable", "deeptab.models"),
    "MambAttention": ModelInfo("MambAttention", "stable", "deeptab.models"),
    "NDTF": ModelInfo("NDTF", "stable", "deeptab.models"),
    "SAINT": ModelInfo("SAINT", "stable", "deeptab.models"),
    "AutoInt": ModelInfo("AutoInt", "stable", "deeptab.models"),
    "TabR": ModelInfo("TabR", "stable", "deeptab.models"),
    "ModernNCA": ModelInfo("ModernNCA", "experimental", "deeptab.models.experimental"),
    "Tangos": ModelInfo("Tangos", "experimental", "deeptab.models.experimental"),
    "Trompt": ModelInfo("Trompt", "experimental", "deeptab.models.experimental"),
}
