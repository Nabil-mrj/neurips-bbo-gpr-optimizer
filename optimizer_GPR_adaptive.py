import bayesmark.random_search as rs
from bayesmark import np_util
from bayesmark.abstract_optimizer import AbstractOptimizer
from bayesmark.experiment import experiment_main

import numpy as np
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern


class AdaptiveExplorerOptimizer(AbstractOptimizer):
    """Bayesian Optimization with an adaptive exploration factor."""

    primary_import = "bayesmark"

    def __init__(self, api_config, random=np_util.random):
        """Initialize the optimizer with a Gaussian Process model."""
        AbstractOptimizer.__init__(self, api_config)
        self.random = random
        self.model = GaussianProcessRegressor(kernel=Matern(nu=2.5))
        self.api_config = api_config
        self.samples = np.empty((0, len(api_config)))
        self.observations = np.empty((0, 1))
        self.exploration_factor = 0.1  # Initial exploration weight

    def suggest(self, n_suggestions=1):
        """Suggest promising hyperparameter values based on observations."""
        if self.samples.shape[0] > 0 and self.observations.shape[0] > 0:
            candidates = self._acquisition_sampling(n_suggestions)
        else:
            candidates = rs.suggest_dict([], [], self.api_config, n_suggestions=n_suggestions)
        return candidates

    def observe(self, X, y):
        """Update the model with new observations."""
        X_transformed = np.array([list(x.values()) for x in X])
        self.samples = np.vstack((self.samples, X_transformed))
        self.observations = np.append(self.observations, y)

        if self.samples.shape[0] > 0 and self.observations.shape[0] > 0:
            self.model.fit(self.samples, self.observations)

        # Adjust exploration factor based on recent variance
        if len(self.observations) > 5:
            recent_variance = np.std(self.observations[-5:])
            if recent_variance < 0.01:
                self.exploration_factor = min(0.5, self.exploration_factor * 1.2)  # Increase exploration
            else:
                self.exploration_factor = max(0.05, self.exploration_factor * 0.9)  # Reduce exploration

    def _acquisition_sampling(self, n_suggestions):
        """Selects promising points using Expected Improvement with adaptive exploration."""
        candidate_count = 1000
        sample_candidates = rs.suggest_dict([], [], self.api_config, n_suggestions=max(candidate_count, 10 * n_suggestions))
        X_transformed = np.array([list(c.values()) for c in sample_candidates])
        ei_scores = self._expected_improvement(X_transformed)

        # Adjust exploration weight dynamically
        weighted_scores = (1 - self.exploration_factor) * ei_scores + self.exploration_factor * np.random.rand(len(ei_scores))

        best_indices = np.argsort(weighted_scores)[-n_suggestions:]
        return [sample_candidates[i] for i in best_indices]

    def _expected_improvement(self, X):
        """Computes Expected Improvement based on the current Gaussian Process model."""
        mean, std_dev = self.model.predict(X, return_std=True)
        best_observed = np.max(self.model.predict(self.samples))

        with np.errstate(divide='warn'):
            improvement_ratio = (best_observed - mean) / std_dev
            ei_values = std_dev * (improvement_ratio * norm.cdf(improvement_ratio) + norm.pdf(improvement_ratio))
            ei_values[std_dev == 0.0] = 0.0
        return ei_values


if __name__ == "__main__":
    experiment_main(AdaptiveExplorerOptimizer)
