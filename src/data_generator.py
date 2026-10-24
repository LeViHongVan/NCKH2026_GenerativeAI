import numpy as np
from typing import Tuple, Dict

MALWARE_FAMILIES = ["Zbot", "Winwebsec", "Zeroaccess", "Expiro.BK", "Hotbar"]

def generate_synthetic_benchmark_dataset(
    n_features: int = 104,
    random_seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, Dict[int, str]]:
    """
    Generates a realistic benchmark embedding dataset mimicking malware opcode distributions:
    - 5 Malware Families with severe class imbalance (Majority vs Minority classes)
    - Multimodal non-linear cluster distributions within [-1, 1] range
    - Ring manifold simulation for Expiro.BK (as documented in IEEE Access paper Figure 6)
    """
    np.random.seed(random_seed)

    # Class sample counts simulating real-world imbalance
    family_counts = {
        0: 600, # Zbot (Majority)
        1: 500, # Winwebsec (Majority)
        2: 300, # Zeroaccess (Medium)
        3: 80,  # Expiro.BK (Minority - Complex Ring structure)
        4: 40   # Hotbar (Severe Minority)
    }

    X_list = []
    y_list = []

    for fam_idx, count in family_counts.items():
        if fam_idx == 3:
            # Expiro.BK: Simulates a ring/circular manifold distribution
            theta = np.linspace(0, 2 * np.pi, count)
            radius = 0.6 + 0.1 * np.random.randn(count)
            # Project onto 104 dimensions with ring structure in primary components
            data = np.zeros((count, n_features))
            data[:, 0] = radius * np.cos(theta)
            data[:, 1] = radius * np.sin(theta)
            # Add secondary feature correlations
            data[:, 2:] = 0.15 * np.random.randn(count, n_features - 2)
        else:
            # Other families: Distinct Gaussian centroids with specific covariance
            centroid = (np.random.rand(n_features) - 0.5) * 1.2
            noise = np.random.randn(count, n_features) * (0.2 + 0.05 * fam_idx)
            data = centroid + noise

        # Clip and normalize into [-1, 1] as required by Tanh
        data = np.clip(data, -1.0, 1.0)

        X_list.append(data)
        y_list.append(np.full(count, fam_idx, dtype=np.int64))

    X_all = np.vstack(X_list).astype(np.float32)
    y_all = np.concatenate(y_list).astype(np.int64)

    family_map = {idx: name for idx, name in enumerate(MALWARE_FAMILIES)}
    return X_all, y_all, family_map
