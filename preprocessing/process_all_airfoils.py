import os
import pickle
import numpy as np
from tqdm import tqdm
from graph_from_snapshot import build_graph_from_snapshot

def process_split(split_name):
    print(f"\n Processing {split_name} set....")

    # Paths
    data_dir ="/Users/karan94/Documents/MeshGraphNets_Airfoil/data/Airfoil"
    save_dir =f"/Users/karan94/Documents/MeshGraphNets_Airfoil/data/graphsAll_{split_name}_airfoil"
    os.makedirs(save_dir, exist_ok=True)

    # Load .npy files
    grid = np.load(os.path.join(data_dir, f"{split_name}_grid.npy"))    # [N_airfoils, N_codes, N_steps, 2]
    cells = np.load(os.path.join(data_dir, f"{split_name}_cells.npy"))  # [N_airfoils, N_codes, N_steps, 3]
    data = np.load(os.path.join(data_dir, f"{split_name}_data.npy"))    # [N_airfoils, N_codes, N_steps, 4]

    n_airfoils, n_nodes, n_steps, _ = data.shape
    print(f"{n_airfoils} airfoils, {n_steps} time steps")

    for a in tqdm(range(n_airfoils), desc=f"Airfoils ({split_name})"):
        for t in range(n_steps - 1):
            node_pos = grid[a, :, t, :]       # [N,2]
            triangles = cells[a, :, t, :]     # [T,3]
            features =  data[a, :, t, :4]     # [N,3] u, v, RHO, p
            target = data[a, :, t + 1, :2]    # [N,3] next frame only u,v
            node_type = np.zeros((n_nodes,))  # dummy placeholder

            # Build Graph
            graph = build_graph_from_snapshot(node_pos, triangles, features, target, node_type)

            # Save to file
            fname = os.path.join(save_dir, f"graph_{a:03d}_{t:03d}.pkl")
            with open(fname, "wb") as f:
                pickle.dump(graph, f)


    print(f"Saved {n_airfoils * (n_steps - 1)} graphs to {save_dir}")


if __name__ == "__main__":
    process_split("train")
    process_split("test")