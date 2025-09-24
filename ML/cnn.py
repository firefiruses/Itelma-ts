import torch
from torch import nn

class GlobalPoolingCNN(nn.Module):
    def __init__(self, seq_name, inp, inp_upclass, layers, classes, _unsqueeze=True):
        super().__init__()
        self.seq_name = seq_name
        self.upclass = nn.Conv1d(inp, inp_upclass, 1, bias=False)
        self.upclass_bn = nn.InstanceNorm1d(inp_upclass)
        self.time_aggreg = nn.Conv1d(inp_upclass, inp_upclass, 5, dilation=7, bias=False)
        self.ta_bn = nn.InstanceNorm1d(inp_upclass)
        prev = inp_upclass
        self.layers = nn.ModuleList()
        for l in layers:
            if str(l).isnumeric():
                curr = int(l)
                self.layers.append(nn.Conv1d(prev, curr, 3, bias=False))
                self.layers.append(nn.InstanceNorm1d(curr))
                self.layers.append(nn.LeakyReLU())
                prev = curr
            else:
                assert str(l) in ["max_pool", "avg_pool"]
                if str(l) == "max_pool":
                    self.layers.append(nn.MaxPool1d(2, 2))
                if str(l) == "avg_pool":
                    self.layers.append(nn.AvgPool1d(2, 2))
        self.lin_fin = nn.Linear(prev, classes)
        self.leaky_relu = nn.LeakyReLU()
        self.global_pooling = nn.AdaptiveMaxPool1d(1)
        self._unsqueeze = _unsqueeze


    def forward(self, inputs: torch.tensor):
        x = inputs[self.seq_name]
        if self._unsqueeze:
            x = x.unsqueeze(0)
        x = self.upclass(x)
        #print(x.shape)
        x = self.leaky_relu(self.upclass_bn(x))
        x = self.leaky_relu(self.ta_bn(self.time_aggreg(x)))
        for l in self.layers:
            x = l(x)
        x = self.global_pooling(x).squeeze(-1)
        x = self.lin_fin(x)
        return x


class JoinEmbedClassifier(nn.Module):
    def __init__(self, inp_size, models, layers, classes):
        super().__init__()
        self.models = nn.ModuleList()
        for m in models:
            self.models.append(m)
        self.layers = nn.ModuleList()
        prev = inp_size * len(models)
        for l in layers:
            curr = l
            self.layers.append(nn.Linear(prev, curr, bias=False))
            self.layers.append(nn.InstanceNorm1d(curr))
            self.layers.append(nn.LeakyReLU())
            prev = curr
        self.lin_fin = nn.Linear(prev, classes)

    def forward(self, inputs):
        x = []
        if self.training:
            coeffs = torch.randn((2,))
            coeffs = (coeffs - coeffs.min()) / (coeffs.max() - coeffs.min())
            coeffs **= 10
            coeffs = (coeffs - coeffs.mean()) / (coeffs.max() - coeffs.min())
        else:
            coeffs = [0.5] * len (self.models)
        i = 0
        for m in self.models:
            x.append(m(inputs) * coeffs[i])
            i += 1
            #print(x[-1].shape)
        x = torch.concat(x, dim=1)
        #print(x.shape)
        for l in self.layers:
            x = l(x)
        return self.lin_fin(x)