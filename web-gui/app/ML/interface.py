import torch
from torch import nn
import pandas as pd, numpy as np
import tsai
from tsai.inference import load_learner
from pathlib import Path

class ModelInterference(nn.Module):
    def __init__(self, loader):
        super().__init__()
        if loader["mode"] not in ["file_name", "model_variable"]:
            raise Exception("Wrong model creation type given. Use \"file_name\" or \"model_variable\" only.")
        if loader["mode"] == "file_name":
            self.model = torch.load(loader["file_name"], weights_only=True)
        if loader["mode"] == "model_variable":
            self.model = loader["model_variable"]
        
    def predict_sequence(self, inputs, seq_len):
        inputs = torch.tensor(inputs)
        x = inputs.clone()
        x_new = torch.tensor(self.predict_single(inputs))
        for _ in range(1, seq_len):
            x = torch.concat([x[..., 1:, :], x_new], dim=len(x.shape)-2)
            x_new = torch.tensor(self.predict_single(x))
        x = torch.concat([x[..., 1:, :], x_new], dim=len(x.shape)-2)
        return x[..., -seq_len:, :].tolist()
    
    def predict_single(self, inputs):
        inputs = torch.tensor(inputs)
        return self.model(inputs.clone())[..., -1:, :].tolist()


class TSAIModel:
    def __init__(self, weights_dir, is_forecast=True):
        model_path = Path(weights_dir)
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {weights_dir}")
        self.model = load_learner(weights_dir , cpu=True)
        self.is_forecast = is_forecast

    def __call__(self, x):
        
        if self.is_forecast:
            return self.model.get_X_preds(x.transpose(-1, -2))[0].transpose(-1, -2)

        return self.model.get_X_preds(x.transpose(-1, -2))[0]
