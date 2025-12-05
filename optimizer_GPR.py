import bayesmark.random_search as rs
from bayesmark import np_util
from bayesmark.abstract_optimizer import AbstractOptimizer
from bayesmark.experiment import experiment_main

import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern


class SmartOptimizer(AbstractOptimizer):
    """Custom Bayesian optimization strategy with Gaussian Process Regression."""

    primary_import = "bayesmark"

    def __init__(self, api_config, random=np_util.random):
        """Initialize the optimizer with Bayesian modeling capabilities."""
        AbstractOptimizer.__init__(self, api_config)
        self.random = random
        self.model = GaussianProcessRegressor(kernel=Matern(nu=2.5))
        self.api_config = api_config
        self.samples = np.empty((0, len(api_config)))
        self.observations = np.empty((0, 1))
        self.processors = self._initialize_processors(api_config)

    def _initialize_processors(self, api_config):
        """Generate appropriate transformations for different variable types."""
        processors = {}
        for key in api_config:
            if api_config[key]['type'] == 'bool':
                processors[key] = lambda x: int(x)
            elif api_config[key]['type'] == 'cat':
                processors[key] = self._categorical_encoding(api_config[key]['values'])
            elif api_config[key]['type'] == 'int':
                if 'range' in api_config[key]:
                    min_val, max_val = api_config[key]['range']
                    processors[key] = self._normalize_range(min_val, max_val)
            elif api_config[key]['type'] == 'real':
                if 'range' in api_config[key]:
                    min_val, max_val = api_config[key]['range']
                    processors[key] = self._log_scale(min_val, max_val) if api_config[key]['space'] == 'log' else self._normalize_range(min_val, max_val)
                else:
                    processors[key] = lambda x: x
        return processors

    def _categorical_encoding(self, categories):
        """Encodes categorical variables into numerical format."""
        def encode(x):
            return categories.index(x)
        return encode

    def _normalize_range(self, min_val, max_val):
        """Scales numerical values to a [0,1] range."""
        def scale(x):
            return (x - min_val) / (max_val - min_val)
        return scale

    def _log_scale(self, min_val, max_val):
        """Applies logarithmic transformation to real variables."""
        def scale(x):
            return (np.log10(x) - np.log10(min_val)) / (np.log10(max_val) - np.log10(min_val))
        return scale

    def _preprocess(self, x):
        """Converts hyperparameters into a structured format for the model."""
        return [self.processors[key](x[key]) for key in self.api_config]

    def suggest(self, n_suggestions=1):
        """Suggest promising hyperparameter values based on observations."""
        if self.samples.shape[0] > 0 and self.observations.shape[0] > 0:
            # Utilize Expected Improvement to guide the search
            candidates = self._acquisition_sampling(n_suggestions)
        else:
            # Initial exploration with random search
            candidates = rs.suggest_dict([], [], self.api_config, n_suggestions=n_suggestions)
        return candidates

    def observe(self, X, y):
        """Incorporate newly observed evaluations into the model."""
        X_transformed = np.array([self._preprocess(x) for x in X])
        self.samples = np.vstack((self.samples, X_transformed))
        self.observations = np.append(self.observations, y)

        if self.samples.shape[0] > 0 and self.observations.shape[0] > 0:
            self.model.fit(self.samples, self.observations)

    def _acquisition_sampling(self, n_suggestions):
        """Selects promising points using Expected Improvement."""
        candidate_count = 1000
        sample_candidates = rs.suggest_dict([], [], self.api_config, n_suggestions=max(candidate_count, 10 * n_suggestions))
        X_transformed = np.array([self._preprocess(c) for c in sample_candidates])
        ei_scores = self._expected_improvement(X_transformed)
        best_indices = np.argsort(ei_scores)[-n_suggestions:]
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
    experiment_main(SmartOptimizer)
