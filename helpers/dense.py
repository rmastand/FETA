import torch
import torch.nn as nn
from copy import deepcopy


def norm_exception():
    raise Exception("Can't set both layer and batch normalization")


class base_network(nn.Module):

    def forward(self, data, context=None):
        NotImplementedError('Must implement a forward method')

    # TODO: make this take data and a batch_size arg so that you can automatically batch the data
    def batch_predict(self, data, context=None):
        store = []
        for data in data:
            store += [self(data, context)]
        return torch.cat(store)


class dense_net(base_network):
    def __init__(self, input_dim, latent_dim, islast=True, output_activ=nn.Identity(), layers=[64, 64, 64], drp=0,
                 batch_norm=False, layer_norm=False, int_activ=torch.relu, context_features=2):
        super(dense_net, self).__init__()
        layers = deepcopy(layers)
        # If adding additional layers to the encoder, don't compress directly to the latent dimension
        # Useful when expanind the capacity of these base models to compare with implicit approach

        self.latent_dim = latent_dim
        self.drp_p = drp
        self.inner_activ = int_activ
        # This is necessary for scaling the outputs to softmax when using splines
        self.hidden_features = layers[-1]

        if context_features is not None:
            self.context_layer = nn.Linear(context_features, layers[0])

        self.functions = nn.ModuleList([nn.Linear(input_dim, layers[0])])

        if islast:
            layers += [latent_dim]

        self.functions.extend(nn.ModuleList([nn.Linear(layers[i], layers[i + 1]) for i in range(len(layers) - 1)]))

        # Change the initilization
        for function in self.functions:
            torch.nn.init.xavier_uniform_(function.weight)
            function.bias.data.fill_(0.0)
        self.output_activ = output_activ

        if batch_norm and layer_norm:
            norm_exception()

        self.norm = 0
        if batch_norm:
            self.norm = 1
            self.norm_func = nn.BatchNorm1d
        if layer_norm:
            self.norm = 1
            self.norm_func = nn.LayerNorm
        if self.norm:
            self.norm_funcs = nn.ModuleList([self.norm_func(layers[i]) for i in range(len(layers) - 1)])

    def forward(self, x, context=None):
        for i, function in enumerate(self.functions[:-1]):
            x = function(x)
            if (context is not None) and (i == 0):
                x += self.context_layer(context)
            if self.norm:
                x = self.norm_funcs[i](x)
            x = self.inner_activ(x)
            x = nn.Dropout(p=self.drp_p)(x)
        x = self.output_activ(self.functions[-1](x))
        return x
