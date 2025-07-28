import numpy as np
from scipy.spatial import Delaunay


def build_graph_from_snapshot(node_pos, triangles, features, target, node_type):
    """
    Build a graph from a CFD snapshot

    Inputs:
        node_pos: [N,2] node coordinates
        triangles: [T, 3] triangle connectivity
        features: [N, F] node features (u,v,p)
        target: [N, T] target (next u,v,p)
        node_type: [N,] one hot encoded integer mask (0=interior, 1=wall ...)

    :return:
        dict with keys: nodes, edges, senders, receivers, globals, targets
    """

    # Extract edges from triangles
    edges = set()
    for tri in triangles:
        edges.add((tri[0], tri[1]))
        edges.add((tri[1], tri[2]))
        edges.add((tri[2], tri[0]))
        edges.add((tri[1], tri[0]))   # bidirectional
        edges.add((tri[2], tri[1]))
        edges.add((tri[0], tri[2]))
    edges = list(edges)

    senders = [s for s, r in edges]
    receivers = [r for s, r in edges]

    # Node Features
    node_features = np.concatenate([features, node_pos, node_type[:, None]], axis=-1)

    # Edge Features
    rel_pos = node_pos[receivers] - node_pos[senders]
    edge_features = rel_pos

    # Global features
    global_features = np.zeros((1,))

    # Return graph dictionary
    graph = {
        "nodes": node_features.astype(np.float32),
        "edges": edge_features.astype(np.float32),
        "senders": np.array(senders, dtype=np.int32),
        "receivers": np.array(receivers, dtype=np.int32),
        "globals": global_features.astype(np.float32),
        "target": target.astype(np.float32)
    }

    return graph