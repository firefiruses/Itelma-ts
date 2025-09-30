import torch
from torch import nn
import pandas as pd, numpy as np

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
        x = inputs.clone()
        x_new = self.predict_single(inputs)
        for _ in range(1, seq_len):
            x = torch.concat([x, x_new], dim=len(x.shape)-2)
            x_new = self.predict_single(x)
        return x[..., -seq_len:, :]
    
    def predict_single(self, inputs):
        return self.model(inputs)[..., -1:, :]