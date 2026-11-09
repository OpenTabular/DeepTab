"""The MLflow checkpoint upload follows ``ObservabilityConfig.mlflow_log_model``."""

import numpy as np
import pytest

from deeptab.core.observability import ObservabilityConfig
from deeptab.models import MLPRegressor


def _artifact_paths(client, run_id, path=None):
    paths = []
    for artifact in client.list_artifacts(run_id, path):
        paths.extend(_artifact_paths(client, run_id, artifact.path) if artifact.is_dir else [artifact.path])
    return paths


@pytest.mark.parametrize("log_model", [True, False])
def test_checkpoint_upload_follows_mlflow_log_model(log_model, tmp_path):
    pytest.importorskip("mlflow")
    from mlflow.tracking import MlflowClient

    observability = ObservabilityConfig(
        root_dir=str(tmp_path), experiment_trackers=["mlflow"], mlflow_log_model=log_model
    )
    features = np.random.default_rng(0).normal(size=(60, 3))
    model = MLPRegressor(observability_config=observability, random_state=0)
    model.fit(features, features[:, 0], max_epochs=1, accelerator="cpu", enable_progress_bar=False)

    client = MlflowClient(observability.mlflow_tracking_uri)
    experiment = client.get_experiment_by_name(observability.mlflow_experiment_name)
    assert experiment is not None
    (run,) = client.search_runs([experiment.experiment_id])
    artifacts = _artifact_paths(client, run.info.run_id)
    assert "summary.json" in artifacts

    checkpoints = [path for path in artifacts if path.endswith(".ckpt")]
    if log_model:
        assert "checkpoints/best_model.ckpt" in checkpoints
    else:
        assert checkpoints == []
