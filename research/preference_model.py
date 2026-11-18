from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np


ROOT_DIR = Path(__file__).resolve().parent.parent

DEFAULT_CHECKPOINT = (
    ROOT_DIR
    / "artifacts"
    / "preference"
    / "preference_model.joblib"
)


FEATURES = [
    "safety",
    "efficiency",
    "comfort",
    "path_length",
]


@dataclass(frozen=True)
class PreferencePrediction:
    """
    Result of comparing two candidate trajectories.

    probability_a:
        Probability that candidate A is preferred.

    probability_b:
        Probability that candidate B is preferred.

    preferred:
        "A", "B", or "tie".

    margin:
        Absolute difference between the two probabilities.

    logit:
        Raw preference-model score for A relative to B.
    """

    probability_a: float
    probability_b: float
    preferred: str
    margin: float
    logit: float


class PreferenceModel:
    """
    Inference wrapper for the trained pairwise human-preference model.

    The underlying model was trained using:

        delta = candidate_A - candidate_B

    with the four trajectory-quality features:

        safety
        efficiency
        comfort
        path_length

    The underlying classifier predicts:

        P(A preferred | A - B)
    """

    def __init__(
        self,
        checkpoint_path: str | Path = DEFAULT_CHECKPOINT,
        tie_threshold: float = 0.05,
    ) -> None:

        self.checkpoint_path = Path(
            checkpoint_path
        )

        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                "Preference model checkpoint not found:\n"
                f"{self.checkpoint_path}"
            )

        if tie_threshold < 0.0 or tie_threshold >= 0.5:
            raise ValueError(
                "tie_threshold must satisfy "
                "0 <= tie_threshold < 0.5"
            )

        self.tie_threshold = float(
            tie_threshold
        )

        artifact = joblib.load(
            self.checkpoint_path
        )

        if not isinstance(artifact, dict):
            raise ValueError(
                "Invalid preference model artifact. "
                "Expected a dictionary."
            )

        if "model" not in artifact:
            raise ValueError(
                "Preference model artifact does not "
                "contain a 'model' entry."
            )

        self.model = artifact["model"]

        saved_features = artifact.get(
            "features",
            FEATURES,
        )

        if list(saved_features) != FEATURES:
            raise ValueError(
                "Preference model feature mismatch.\n"
                f"Expected: {FEATURES}\n"
                f"Found:    {saved_features}"
            )

        self.features = list(FEATURES)

        self.model_type = artifact.get(
            "model_type",
            "pairwise_logistic_bradley_terry",
        )

        self.training_seed = artifact.get(
            "training_seed",
            42,
        )

        self.weights = artifact.get(
            "weights",
            None,
        )

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_features(
        features: Any,
        name: str,
    ) -> np.ndarray:
        """
        Convert a trajectory feature vector into a validated
        numpy array.

        Expected order:

            [safety, efficiency, comfort, path_length]
        """

        array = np.asarray(
            features,
            dtype=np.float64,
        ).reshape(-1)

        if array.size != len(FEATURES):
            raise ValueError(
                f"{name} must contain exactly "
                f"{len(FEATURES)} values in this order:\n"
                f"{FEATURES}\n"
                f"Received {array.size} values."
            )

        if not np.all(np.isfinite(array)):
            raise ValueError(
                f"{name} contains NaN or infinite values."
            )

        return array

    @classmethod
    def _feature_vector_from_dict(
        cls,
        candidate: dict[str, Any],
        name: str,
    ) -> np.ndarray:
        """
        Extract the four required trajectory features from
        a candidate dictionary.
        """

        missing = [
            feature
            for feature in FEATURES
            if feature not in candidate
        ]

        if missing:
            raise ValueError(
                f"{name} is missing required features: "
                f"{missing}"
            )

        values = [
            candidate[feature]
            for feature in FEATURES
        ]

        return cls._validate_features(
            values,
            name,
        )

    # ------------------------------------------------------------------
    # Pairwise scoring
    # ------------------------------------------------------------------

    def score_pair(
        self,
        candidate_a: Any,
        candidate_b: Any,
    ) -> PreferencePrediction:
        """
        Compare two candidate feature vectors.

        Each candidate must contain:

            [safety, efficiency, comfort, path_length]

        Returns:

            PreferencePrediction
        """

        A = self._validate_features(
            candidate_a,
            "candidate_a",
        )

        B = self._validate_features(
            candidate_b,
            "candidate_b",
        )

        delta = (
            A - B
        ).reshape(1, -1)

        probability_a = float(
            self.model.predict_proba(delta)[0, 1]
        )

        probability_a = float(
            np.clip(
                probability_a,
                0.0,
                1.0,
            )
        )

        probability_b = 1.0 - probability_a

        margin = abs(
            probability_a
            - probability_b
        )

        if (
            abs(
                probability_a
                - 0.5
            )
            <= self.tie_threshold
        ):
            preferred = "tie"

        elif probability_a > 0.5:
            preferred = "A"

        else:
            preferred = "B"

        logit = self._logit(
            probability_a
        )

        return PreferencePrediction(
            probability_a=probability_a,
            probability_b=probability_b,
            preferred=preferred,
            margin=margin,
            logit=logit,
        )

    def score_candidates(
        self,
        candidate_a: dict[str, Any],
        candidate_b: dict[str, Any],
    ) -> PreferencePrediction:
        """
        Compare two candidate dictionaries.

        Expected keys:

            safety
            efficiency
            comfort
            path_length
        """

        A = self._feature_vector_from_dict(
            candidate_a,
            "candidate_a",
        )

        B = self._feature_vector_from_dict(
            candidate_b,
            "candidate_b",
        )

        return self.score_pair(
            A,
            B,
        )

    # ------------------------------------------------------------------
    # Individual candidate score
    # ------------------------------------------------------------------

    def utility(
        self,
        candidate: Any,
    ) -> float:
        """
        Return the learned latent preference utility of one candidate.

        This uses the underlying logistic model's intercept and
        standardized feature transformation.

        The absolute value is not a probability.

        It is useful for ranking multiple candidates because:

            utility(A) > utility(B)

        means the model considers A more preferable than B.
        """

        features = self._validate_features(
            candidate,
            "candidate",
        ).reshape(1, -1)

        return float(
            self.model.decision_function(
                features
            )[0]
        )

    def utility_from_dict(
        self,
        candidate: dict[str, Any],
    ) -> float:
        """
        Return latent preference utility for a candidate dictionary.
        """

        features = self._feature_vector_from_dict(
            candidate,
            "candidate",
        )

        return self.utility(
            features
        )

    def rank_candidates(
        self,
        candidates: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """
        Rank candidate dictionaries from highest to lowest
        learned preference utility.

        The original candidate dictionaries are copied and augmented
        with:

            preference_utility
            preference_rank
        """

        ranked = []

        for index, candidate in enumerate(
            candidates
        ):
            utility = self.utility_from_dict(
                candidate
            )

            item = dict(candidate)

            item[
                "preference_utility"
            ] = utility

            item[
                "_original_index"
            ] = index

            ranked.append(item)

        ranked.sort(
            key=lambda item: item[
                "preference_utility"
            ],
            reverse=True,
        )

        for rank, item in enumerate(
            ranked,
            start=1,
        ):
            item[
                "preference_rank"
            ] = rank

        return ranked

    # ------------------------------------------------------------------
    # Pairwise matrix
    # ------------------------------------------------------------------

    def pairwise_probability_matrix(
        self,
        candidates: list[dict[str, Any]],
    ) -> np.ndarray:
        """
        Calculate P(i preferred over j) for every candidate pair.

        Returns an N x N matrix.

        matrix[i, j] =
            P(candidate i preferred over candidate j)

        Diagonal entries are 0.5.
        """

        n = len(candidates)

        if n == 0:
            return np.empty(
                (0, 0),
                dtype=np.float64,
            )

        feature_matrix = np.vstack(
            [
                self._feature_vector_from_dict(
                    candidate,
                    f"candidate[{i}]",
                )
                for i, candidate in enumerate(
                    candidates
                )
            ]
        )

        matrix = np.full(
            (n, n),
            0.5,
            dtype=np.float64,
        )

        for i in range(n):
            for j in range(i + 1, n):

                prediction = self.score_pair(
                    feature_matrix[i],
                    feature_matrix[j],
                )

                matrix[i, j] = (
                    prediction.probability_a
                )

                matrix[j, i] = (
                    prediction.probability_b
                )

        return matrix

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _logit(
        probability: float,
    ) -> float:
        """
        Convert probability to log-odds.
        """

        epsilon = 1e-12

        probability = float(
            np.clip(
                probability,
                epsilon,
                1.0 - epsilon,
            )
        )

        return float(
            np.log(
                probability
                / (1.0 - probability)
            )
        )

    def describe(self) -> dict[str, Any]:
        """
        Return model metadata.
        """

        return {
            "checkpoint": str(
                self.checkpoint_path
            ),
            "model_type": self.model_type,
            "features": list(
                self.features
            ),
            "training_seed": self.training_seed,
            "weights": self.weights,
            "tie_threshold": self.tie_threshold,
        }


# ----------------------------------------------------------------------
# Simple command-line test
# ----------------------------------------------------------------------

def main() -> None:

    print("=" * 72)
    print("PREFERENCE MODEL TEST")
    print("=" * 72)

    model = PreferenceModel()

    print(
        f"Checkpoint: "
        f"{model.checkpoint_path}"
    )

    print(
        f"Features: "
        f"{model.features}"
    )

    print(
        f"Model type: "
        f"{model.model_type}"
    )

    print()

    # Two example feature vectors.
    #
    # These are ONLY a smoke test.
    # They are not from the annotated dataset.
    candidate_a = {
        "safety": 0.90,
        "efficiency": 0.80,
        "comfort": 0.75,
        "path_length": 100.0,
    }

    candidate_b = {
        "safety": 0.88,
        "efficiency": 0.70,
        "comfort": 0.73,
        "path_length": 105.0,
    }

    prediction = model.score_candidates(
        candidate_a,
        candidate_b,
    )

    print("=" * 72)
    print("PAIRWISE TEST")
    print("=" * 72)

    print(
        f"P(A preferred): "
        f"{prediction.probability_a:.6f}"
    )

    print(
        f"P(B preferred): "
        f"{prediction.probability_b:.6f}"
    )

    print(
        f"Preferred: "
        f"{prediction.preferred}"
    )

    print(
        f"Margin: "
        f"{prediction.margin:.6f}"
    )

    print(
        f"Logit: "
        f"{prediction.logit:.6f}"
    )

    print()

    print("=" * 72)
    print("UTILITY TEST")
    print("=" * 72)

    utility_a = model.utility_from_dict(
        candidate_a
    )

    utility_b = model.utility_from_dict(
        candidate_b
    )

    print(
        f"Candidate A utility: "
        f"{utility_a:+.6f}"
    )

    print(
        f"Candidate B utility: "
        f"{utility_b:+.6f}"
    )

    print()

    print("=" * 72)
    print("MODEL READY")
    print("=" * 72)


if __name__ == "__main__":
    main()