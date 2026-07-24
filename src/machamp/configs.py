"""Create per-run MaChAmp configuration files."""

import json
from pathlib import Path


class ConfigCreator:
    def __init__(
        self,
        dataset,
        encoder,
        encoding,
        num_epochs,
        seed,
        template_dir="parameter_configs",
        data_dir="data",
        logs_dir="logs",
    ):
        self.dataset = dataset
        self.encoder = encoder
        self.encoding = encoding
        self.seed = int(seed)
        self.num_epochs = int(num_epochs)
        self.template_dir = Path(template_dir)
        self.data_dir = Path(data_dir)
        self.logs_dir = Path(logs_dir)
        self.encoder_name = Path(encoder).name
        self.model_dir = (
            self.logs_dir
            / "machamp"
            / dataset
            / self.encoder_name
            / encoding
            / f"seed_{self.seed}"
        )
        self.model_dir.mkdir(parents=True, exist_ok=True)

    def create_parameters_config(self):
        template_file = self.template_dir / "bert.json"
        if not template_file.is_file():
            raise FileNotFoundError(f"Parameter template not found: {template_file}")
        parameters = json.loads(template_file.read_text(encoding="utf-8"))
        parameters["transformer_model"] = self.encoder
        parameters["random_seed"] = self.seed
        parameters["training"]["num_epochs"] = self.num_epochs
        config_path = self.model_dir / "params-config.json"
        config_path.write_text(json.dumps(parameters, indent=2) + "\n", encoding="utf-8")
        return str(config_path)

    def create_dataset_config(self):
        encoded_dir = self.data_dir / self.dataset / self.encoding
        dataset_config = {
            self.dataset: {
                "train_data_path": str(encoded_dir / "train.labels"),
                "dev_data_path": str(encoded_dir / "dev.labels"),
                "word_idx": 0,
                "tasks": {
                    "ci": {"task_type": "seq", "column_idx": 2},
                    "ni": {"task_type": "seq", "column_idx": 3},
                    "ui": {"task_type": "seq", "column_idx": 4},
                },
            }
        }
        config_path = self.model_dir / "dataset-configs.json"
        config_path.write_text(
            json.dumps(dataset_config, indent=2) + "\n",
            encoding="utf-8",
        )
        return str(config_path)
