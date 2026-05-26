from .datamodule import TabularDataModule
from .dataset import TabularDataset

# For backwards compatibility
MambularDataModule = TabularDataModule
MambularDataset = TabularDataset

__all__ = [
    "MambularDataModule",  # Deprecated alias
    "MambularDataset",  # Deprecated alias
    "TabularDataModule",
    "TabularDataset",
]
