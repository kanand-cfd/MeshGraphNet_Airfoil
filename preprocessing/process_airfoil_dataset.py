import os
import pickle
import numpy as np
from tqdm import tqdm
from graph_from_snapshot import build_graph_from_snapshot

# Config
DATA_DIR = "/Users/karan94/Documents/MeshGraphNets_Airfoil/data/Airfoil"
SAVE_DIR = "/Users/karan94/Documents/MeshGraphNets_Airfoil/data/graphs_test_airfoil"
os.makedirs(SAVE_DIR, exist_ok=True)

airfoil_id = 0

print("Loading raw data...")
grid = np.load(f"{DATA_DIR}/test_grid.npy")
cells = np.load(f"{DATA_DIR}/test_cells.npy")
data = np.load(f"{DATA_DIR}/test_data.npy")

num_timesteps = grid.shape[2]

print(f"Building graphs for airfoil {airfoil_id}")

for frame in tqdm(range(num_timesteps - 1)):
    # Inputs
    node_pos = grid[airfoil_id, :, frame, :]        # [N, 2]
    triangles = cells[airfoil_id, :, frame, :]      # [T, 3]
    features = data[airfoil_id, :, frame, :4]       # [N, 3] u, v, rho, p
    target = data[airfoil_id, :, frame + 1, :4]     # [N,3] next timestep
    node_type = np.zeros((features.shape[0],))

    # Build graph
    graph = build_graph_from_snapshot(node_pos, triangles, features, target, node_type)

    # Save to *.pkl files
    fname = os.path.join(SAVE_DIR, f"graph_{frame:05d}.pkl")
    with open(fname, "wb") as f:
        pickle.dump(graph, f)

print(f"Saved {num_timesteps - 1} graphs to {SAVE_DIR}")

