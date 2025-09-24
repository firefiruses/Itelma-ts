import torch
from torch import nn


class TimeEmbed(nn.Module):
    def __init__(self, info_params, ti, dti, layers):
        super().__init__()
        self.ti = ti
        self.dti = dti
        self.layers = nn.ModuleList()
        prev = info_params + 2
        for l in layers:
            self.layers.append(nn.ModuleList())
            self.layers[-1].append(nn.Linear(prev, l, bias=False))
            self.layers[-1].append(nn.LeakyReLU())
            self.layers[-1].append(nn.InstanceNorm1d(l))
            prev = l + 2
        
    def forward(self, inputs):
        xti = inputs[..., self.ti].unsqueeze(-1)
        xdti = inputs[..., self.dti].unsqueeze(-1)
        x = inputs[..., [i not in [self.ti, self.dti] for i in range(inputs.shape[-1])]]
        x = torch.cat([x, xti, xdti], dim=len(inputs.shape)-1)
        for lm in self.layers:
            for l in lm:
                x = l(x)
            x = torch.cat([x, xti, xdti], dim=len(x.shape)-1)
            #print(x.shape)
        return x


class GRUClassifier(nn.Module):
    def __init__(self, layers, hid_dim, embed_model, embed_h, classes):
        super().__init__()
        self.embed = embed_model
        self.gru = nn.GRU(embed_h, hid_dim, layers, batch_first=True)
        self.fin_lin = nn.Linear(hid_dim, classes)

    def forward(self, inputs):
        x = self.embed(inputs).contiguous()
        if x.is_contiguous():
            _, x = self.gru(x)
            x = self.fin_lin(x[-1, :, :])
        else:
            print(1/0)
        return x