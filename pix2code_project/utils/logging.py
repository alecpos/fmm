import logging
import torch
from torch.utils.tensorboard import SummaryWriter
import os

class TrainingLogger:
    def __init__(self, log_dir, model_name):
        self.log_dir = log_dir
        self.model_name = model_name
        
        # Create log directory
        os.makedirs(log_dir, exist_ok=True)
        
        # Initialize tensorboard writer
        self.writer = SummaryWriter(log_dir=os.path.join(log_dir, model_name))
        
        # Initialize logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(log_dir, f'{model_name}.log')),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(model_name)
    
    def log_metrics(self, metrics, step):
        """Log training metrics to tensorboard"""
        for name, value in metrics.items():
            self.writer.add_scalar(name, value, step)
    
    def log_model_graph(self, model, input_shape):
        """Log model architecture to tensorboard"""
        dummy_input = torch.randn(input_shape)
        self.writer.add_graph(model, dummy_input)
    
    def log_hyperparameters(self, config):
        """Log hyperparameters to TensorBoard."""
        # Flatten nested dictionaries and convert values to supported types
        flat_config = {}
        for key, value in config.items():
            if isinstance(value, dict):
                for subkey, subvalue in value.items():
                    if isinstance(subvalue, dict):
                        for subsubkey, subsubvalue in subvalue.items():
                            # Convert value to supported type
                            if isinstance(subsubvalue, (int, float, str, bool)):
                                flat_config[f"{key}/{subkey}/{subsubkey}"] = subsubvalue
                            elif isinstance(subsubvalue, list):
                                # Convert lists to strings
                                flat_config[f"{key}/{subkey}/{subsubkey}"] = str(subsubvalue)
                            else:
                                # Convert other types to strings
                                flat_config[f"{key}/{subkey}/{subsubkey}"] = str(subsubvalue)
                    else:
                        # Convert value to supported type
                        if isinstance(subvalue, (int, float, str, bool)):
                            flat_config[f"{key}/{subkey}"] = subvalue
                        elif isinstance(subvalue, list):
                            # Convert lists to strings
                            flat_config[f"{key}/{subkey}"] = str(subvalue)
                        else:
                            # Convert other types to strings
                            flat_config[f"{key}/{subkey}"] = str(subvalue)
            else:
                # Convert value to supported type
                if isinstance(value, (int, float, str, bool)):
                    flat_config[key] = value
                elif isinstance(value, list):
                    # Convert lists to strings
                    flat_config[key] = str(value)
                else:
                    # Convert other types to strings
                    flat_config[key] = str(value)
        
        # Log flattened hyperparameters
        self.writer.add_hparams(flat_config, {})
    
    def close(self):
        """Close the tensorboard writer"""
        self.writer.close() 