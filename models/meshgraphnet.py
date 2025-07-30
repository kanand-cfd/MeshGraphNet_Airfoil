import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.nn import MessagePassing
import torch.nn.functional as F

# ==== Utility MLP block ==== #
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers=2):
        super().__init__()
        layers = []
        dims = [input_dim] + [hidden_dim] * n_layers + [output_dim]
        for i in range(len(dims) - 2):
            layers += [nn.Linear(dims[i], dims[i+1]), nn.ReLU()]
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.mlp = nn.Sequential(*layers)


    def forward(self, x):
        return self.mlp(x)


# ==== MPNN Layer ==== #
class EdgeUpdateBlock(MessagePassing):
    def __init__(self, latent_dim):
        super().__init__(aggr='mean')
        self.edge_mlp = MLP(2 * latent_dim + 2, latent_dim, latent_dim)
        self.node_mlp = MLP(2 * latent_dim, latent_dim, latent_dim)

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor) -> Tensor:
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_i: Tensor, x_j: Tensor, edge_attr: Tensor) -> Tensor:
        # x_i: target_node, x_j: source node
        edge_input = torch.cat([x_i, x_j, edge_attr], dim=-1)
        return self.edge_mlp(edge_input)

    def update(self, aggr_out: Tensor, x: Tensor) -> Tensor:
        node_input = torch.cat([x, aggr_out], dim=-1)
        return self.node_mlp(node_input)

# ==== MeshGraphNet Model ==== #
class MeshGraphNet(nn.Module):
    def __init__(self, node_input_dim, edge_input_dim, output_dim,
                 latent_dim=128, processor_steps=5):
        super().__init__()
        self.node_encoder = MLP(node_input_dim, latent_dim, latent_dim)
        self.edge_encoder = MLP(edge_input_dim, latent_dim, latent_dim)
        self.processor = nn.ModuleList([EdgeUpdateBlock(latent_dim) for _ in range(processor_steps)])
        self.decoder = MLP(latent_dim, latent_dim, output_dim)

    def forward(self, data):
        x = self.node_encoder(data.x)
        edge_attr = self.edge_encoder(data.edge_attr)
        edge_index = data.edge_index

        for layer in self.processor:
            x = x + layer(x, edge_index, edge_attr)  # Residual

        out = self.decoder(x)

        return out