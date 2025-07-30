import os
import pickle
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data

class GraphDataset(Dataset):
    def __init__(self, data_dir, transform=None, target_fields=(0,1)):
        """
        Args:
        :param data_dir: Path to directory containing .pkl graph files
        :param transform: Optional transform to apply
        :param target_fields: Indices of target features to predict
        """

        self.data_dir = data_dir
        self.transform = transform
        self.target_fields = target_fields

        # Load file list
        self.graph_files = sorted([
            os.path.join(data_dir, f)
            for f in os.listdir(data_dir)
            if f.endswith('.pkl')
        ])

    def __len__(self):
        return len(self.graph_files)


    def __getitem__(self, item):
        with open(self.graph_files[item], 'rb') as f:
            g = pickle.load(f)


        # Convert to torch tensors
        nodes = torch.tensor(g["nodes"], dtype=torch.float32)
        edges = torch.tensor(g["edges"], dtype=torch.float32)
        senders = torch.tensor(g["senders"], dtype=torch.long)
        receivers = torch.tensor(g["receivers"], dtype=torch.long)
        targets = torch.tensor(g["target"],dtype=torch.float32)

        # Slice only selected target fields (u, v)
        if self.target_fields is not None:
            targets = targets[:, self.target_fields]

        data = Data(
            x=nodes,
            edge_index=torch.stack([senders, receivers], dim=0),
            edge_attr=edges,
            y=targets,
        )

        if  self.transform:
            data = self.transform(data)


        return data
