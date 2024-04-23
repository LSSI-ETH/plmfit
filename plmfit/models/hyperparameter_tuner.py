import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from scipy.optimize import minimize

class BayesOptSearch:
    def __init__(self, bounds, kappa=2.5, xi=0.0, random_search_steps=10):
        self.bounds = bounds
        self.kappa = kappa
        self.xi = xi
        self.random_search_steps = random_search_steps
        self.kernel = Matern()
        self.gp = GaussianProcessRegressor(kernel=self.kernel)
        self.samples = []
        self.results = []

    def suggest(self):
        if len(self.samples) < self.random_search_steps:
            return {k: np.random.uniform(v[0], v[1]) for k, v in self.bounds.items()}
        else:
            self.gp.fit(self.samples, self.results)
            def acquisition(x):
                mean, std = self.gp.predict([x], return_std=True)
                return - (mean - self.kappa * std)
            res = minimize(acquisition, [np.mean(v) for v in self.bounds.values()])
            return {k: res.x[i] for i, k in enumerate(self.bounds)}

    def update(self, sample, result):
        self.samples.append([sample[k] for k in self.bounds])
        self.results.append(result)

class HyperTuner:
    def __init__(self, function_to_run, initial_config, bounds, trials, logger, experiment_dir, **kwargs):
        self.function_to_run = function_to_run
        self.initial_config = initial_config
        self.bounds = bounds
        self.trials = trials
        self.logger = logger
        self.experiment_dir = experiment_dir
        self.run_args = kwargs
        self.current_loss = float('inf')  # Initialize current loss

    def fit(self):
        config = self.initial_config.copy()

        self.searcher = BayesOptSearch(self.bounds)
        best_config = None
        best_loss = float('inf')
        
        for _ in range(self.trials):
            suggested_config = self.searcher.suggest()
            temp_config = config.copy()
            temp_config['training_parameters'].update(suggested_config)
            temp_config['training_parameters']['epochs'] = 10  # Temporarily drop epochs
            temp_config['training_parameters']['early_stopping'] = 2  # Temporarily set early stopping

            self.run_args['report_loss'] = self.report_loss  # Ensure report_loss is passed correctly
            self.function_to_run(config=temp_config, **self.run_args)

            if self.current_loss < best_loss:
                best_loss = self.current_loss
                best_config = suggested_config  # Store only the suggested part
            self.searcher.update(suggested_config, self.current_loss)
            self.logger.log(f"Trial completed: Loss={self.current_loss}, Config={suggested_config}")
            self.current_loss = float('inf')  # Reset for the next trial

        return best_config, best_loss
    
    def report_loss(self, loss):
        if loss < self.current_loss:
            self.current_loss = loss