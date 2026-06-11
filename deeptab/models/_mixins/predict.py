"""Inference, encoding, and scoring logic for all DeepTab estimators."""

from __future__ import annotations

import torch
from sklearn.utils.validation import check_is_fitted
from torch.utils.data import DataLoader
from tqdm import tqdm

from deeptab.core.sklearn_compat import validate_input_features


class _PredictMixin:
    """Inference, encoding, and internal scoring.

    Responsibilities
    ----------------
    * ``predict`` — abstract; overridden by each concrete estimator to
      return predictions in the expected sklearn shape.
    * ``_validate_predict_input`` — checks the model is fitted and that
      the input columns match those seen during ``fit``.
    * ``encode`` — returns dense embedding vectors from the model backbone
      for a given input DataFrame.
    * ``_score`` — internal helper used by ``optimize_hparams`` to evaluate
      validation loss with the best checkpoint loaded.
    """

    def predict(self, X, embeddings=None, device=None):
        """Return predictions for input *X*.

        Parameters
        ----------
        X : array-like or DataFrame of shape (n_samples, n_features)
            Input features.
        embeddings : array-like or None, optional
            Pre-computed external embeddings aligned with the rows of *X*.
        device : str or torch.device or None, optional
            Device override for inference (e.g. ``"cpu"`` to force CPU).
            When ``None`` the model's current device is used.

        Returns
        -------
        numpy.ndarray
            1-D array of shape ``(n_samples,)`` for classification and
            regression tasks.

        Raises
        ------
        NotImplementedError
            Always — this method must be overridden by each concrete subclass.
        """
        raise NotImplementedError("The 'predict' method is not implemented in the Parent class.")

    def _validate_predict_input(self, X):
        """Check the model is fitted and validate the input feature columns.

        Parameters
        ----------
        X : array-like or DataFrame
            Raw input to be passed to ``predict``.

        Returns
        -------
        pandas.DataFrame
            The validated and coerced input, with columns verified against
            those seen during ``fit``.

        Raises
        ------
        sklearn.exceptions.NotFittedError
            If ``fit`` has not been called yet.
        deeptab.core.exceptions.ColumnCountError
            If the number of columns differs from ``n_features_in_``.
        """
        check_is_fitted(self)  # raises sklearn's NotFittedError before any other check
        return validate_input_features(self, X)

    def _score(self, X, y, embeddings, metric):
        """Evaluate *metric* on *X* / *y* using the best-checkpoint weights.

        Reloads the best model checkpoint before running ``predict`` so that
        the score reflects the best validation state rather than the last
        epoch's weights.

        Parameters
        ----------
        X : array-like or DataFrame
            Input features.
        y : array-like
            True target values.
        embeddings : array-like or None
            Pre-computed external embeddings aligned with *X*.
        metric : Callable[[array-like, array-like], float]
            A scoring callable that accepts ``(y_true, y_pred)`` and
            returns a scalar (lower = better for losses, higher = better
            for accuracy-style metrics).

        Returns
        -------
        float
            The metric value computed on the predictions.
        """
        # Explicitly load the best model state if needed
        if hasattr(self, "_trainer") and self._best_model_path:
            torch.serialization.add_safe_globals([type(self.config)])
            checkpoint = torch.load(self._best_model_path, weights_only=False)
            self._task_model.load_state_dict(checkpoint["state_dict"])  # type: ignore

        predictions = self.predict(X, embeddings)

        return metric(y, predictions)

    def encode(self, X, embeddings=None, batch_size=64):
        """Return dense embedding vectors from the model backbone.

        Runs the fitted model's ``encode`` method on batches of *X* and
        concatenates the results into a single tensor.

        Parameters
        ----------
        X : array-like or DataFrame of shape (n_samples, n_features)
            Input features to encode.
        embeddings : array-like or None, optional
            Pre-computed external embeddings aligned with the rows of *X*.
        batch_size : int, default=64
            Number of samples processed in each forward pass.

        Returns
        -------
        torch.Tensor of shape (n_samples, embedding_dim)
            Encoded representations of the input data.

        Raises
        ------
        ValueError
            If the model has not been fitted yet.

        Examples
        --------
        >>> clf = MLPClassifier()
        >>> clf.fit(X_train, y_train)
        >>> embeddings = clf.encode(X_test)        # (n_samples, embedding_dim)
        >>> embeddings.shape
        torch.Size([100, 64])
        """
        if self._task_model is None or self._data_module is None:
            raise ValueError("The model or data module has not been fitted yet.")

        encoded_dataset = self._data_module.preprocess_new_data(X, embeddings)
        data_loader = DataLoader(encoded_dataset, batch_size=batch_size, shuffle=False)

        encoded_outputs = []
        for batch in tqdm(data_loader):
            emb = self._task_model.estimator.encode(batch)  # type: ignore[union-attr]
            encoded_outputs.append(emb)

        return torch.cat(encoded_outputs, dim=0)
