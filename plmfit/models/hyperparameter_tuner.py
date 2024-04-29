from bayes_opt import BayesianOptimization
import copy
import time

class HyperTuner:
    def __init__(self, function_to_run, initial_config, trials, logger, experiment_dir, **kwargs):
        self.function_to_run = function_to_run
        self.initial_config = initial_config
        self.bounds = self.extract_bounds(initial_config)
        self.trials = trials
        self.logger = logger
        self.experiment_dir = experiment_dir
        self.run_args = kwargs
        self.current_loss = 0

    def extract_bounds(self, config):
        bounds = {}
        # Example: Extracting bounds for training and architecture parameters
        for param_section in ['architecture_parameters', 'training_parameters']:
            for key, value in config[param_section].items():
                if isinstance(value, tuple):  # Assuming bounds are given as tuples
                    bounds[key] = value
        return bounds

    def fit(self):
        num_hyperparameters = len(self.bounds)
        init_points = 5 * num_hyperparameters  # Set initial points to five times the number of hyperparameters
        start_time = time.time()
        optimizer = BayesianOptimization(
            f=self.run_trial,
            pbounds=self.bounds,
            random_state=1,
        )
        self.best_config = copy.deepcopy(self.initial_config)  # Start with a copy of the initial full configuration
        self.best_loss = float('-inf')
        self.current_trial = 1
        optimizer.maximize(init_points=init_points, n_iter=self.trials - init_points)
        self.logger.log(f'Hyperparameter tuning completed in {time.time() - start_time}s')
        return self.best_config, -self.best_loss
    
    def run_trial(self, **args):
        start_time = time.time()        
        temp_config = copy.deepcopy(self.best_config)
        # Update the parameters with suggested values
        for key in args:
            if key in temp_config['architecture_parameters']:
                temp_config['architecture_parameters'][key] = args[key]
            elif key in temp_config['training_parameters']:
                temp_config['training_parameters'][key] = args[key]
        # Set epochs and early stopping temporarily
        old_epochs = temp_config['training_parameters']['epochs']
        old_early_stopping = temp_config['training_parameters']['early_stopping']
        old_epoch_sizing = temp_config['training_parameters']['epoch_sizing']
        temp_config['training_parameters']['epochs'] = 10
        temp_config['training_parameters']['early_stopping'] = 2
        temp_config['training_parameters']['epoch_sizing'] = 0.25

        self.current_loss = self.function_to_run(config=temp_config, logger=self.logger, **self.run_args)
        self.current_loss = 0.0 - self.current_loss

        temp_config['training_parameters']['epochs'] = old_epochs
        temp_config['training_parameters']['early_stopping'] = old_early_stopping
        temp_config['training_parameters']['epoch_sizing'] = old_epoch_sizing

        if self.current_loss > self.best_loss:
            self.best_loss = self.current_loss
            self.best_config = copy.deepcopy(temp_config)  # Update best_config with the best found configuration

        self.logger.log(f"[{self.current_trial}] Trial completed in {time.time() - start_time:.4f}s: Loss={0-self.current_loss:.4f}, Config={temp_config}")
        self.logger.log(f"Current best trial: Loss={0-self.best_loss:.4f}, Config={self.best_config}\n")
        self.current_trial = self.current_trial + 1
        return self.current_loss
    
