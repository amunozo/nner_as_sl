import json
import os

class ConfigCreator:
    def __init__(self, dataset, encoder, encoding, seed, template_dir='parameter_configs'):
        """
        Initialize the ConfigCreator with dataset, encoder, encoding, seed, and optional template directory.
        """
        self.dataset = dataset
        self.encoder = encoder
        self.encoding = encoding
        self.seed = seed
        self.template_dir = template_dir
        self.encoder_name = encoder.split('/')[-1]
        self.model_dir = f'logs/machamp/{dataset}/{self.encoder_name}/{encoding}/seed_{seed}'
        self._ensure_model_directory()

    def _ensure_model_directory(self):
        """
        Ensure that the model directory exists.
        """
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

    def create_parameters_config(self):
        """
        Create the parameters config for the MaChAmp model based on the BERT model template.
        """
        template_file = f'{self.template_dir}/bert.json'  # other models may need other templates
        
        with open(template_file, 'r') as f:
            parameters_config = json.load(f)

        parameters_config["transformer_model"] = self.encoder
        parameters_config["random_seed"] = self.seed
        
        config_path = f'{self.model_dir}/params-config.json'
        with open(config_path, 'w') as f:
            json.dump(parameters_config, f)
        
        return config_path

    def create_dataset_config(self):
        """
        Create the dataset config for the MaChAmp model.
        """
        dataset_config = {
            f"{self.dataset}": {
                "train_data_path": f"clean_data/{self.dataset}/{self.encoding}/train.labels",
                "dev_data_path": f"clean_data/{self.dataset}/{self.encoding}/dev.labels",
                "word_idx": 0,
                "tasks": {
                    "ci": {
                        "task_type": "seq",
                        "column_idx": 2,
                    },
                    "ni": {
                        "task_type": "seq",
                        "column_idx": 3,
                    },
                    "ui": {
                        "task_type": "seq",
                        "column_idx": 4,
                    },
                }
            }
        }
        
        config_path = f'{self.model_dir}/dataset-configs.json'
        with open(config_path, 'w') as f:
            json.dump(dataset_config, f)
        
        return config_path


