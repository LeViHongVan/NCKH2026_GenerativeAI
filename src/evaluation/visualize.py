import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from typing import Dict, List, Optional

def plot_tsne_comparison(real_data: np.ndarray,
                         gan_data: Optional[np.ndarray],
                         wgan_data: Optional[np.ndarray],
                         diff_data: Optional[np.ndarray],
                         family_name: str = "Expiro.BK",
                         save_path: str = "results/tsne_comparison.png"):
    """
    Metric 2: t-SNE Dimension Reduction Visualization (Replicates Figure 6 in Paper).
    Visualizes 104D embeddings into 2D space for Real vs GAN vs WGAN-GP vs Diffusion.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(9, 8))

    X_list = [real_data]
    labels = ["Real"] * len(real_data)

    if gan_data is not None:
        X_list.append(gan_data)
        labels.extend(["GAN"] * len(gan_data))
    if wgan_data is not None:
        X_list.append(wgan_data)
        labels.extend(["WGAN-GP"] * len(wgan_data))
    if diff_data is not None:
        X_list.append(diff_data)
        labels.extend(["Diffusion"] * len(diff_data))

    X_all = np.vstack(X_list)

    print("[*] Running t-SNE (2 components, perplexity=30)...")
    tsne = TSNE(n_components=2, perplexity=min(30, max(5, len(X_all)//5)), random_state=42)
    X_2d = tsne.fit_transform(X_all)

    # Plot each category
    colors = {"Real": "#2ca02c", "Diffusion": "#d62728", "WGAN-GP": "#1f77b4", "GAN": "#ff7f0e"}
    start_idx = 0

    for name, data in [("Real", real_data), ("GAN", gan_data), ("WGAN-GP", wgan_data), ("Diffusion", diff_data)]:
        if data is not None:
            end_idx = start_idx + len(data)
            plt.scatter(
                X_2d[start_idx:end_idx, 0],
                X_2d[start_idx:end_idx, 1],
                c=colors[name],
                label=name,
                alpha=0.6,
                edgecolors='none',
                s=25
            )
            start_idx = end_idx

    plt.title(f"t-SNE Dimension Reduction on {family_name} Family (Paper Fig. 6 Replication)", fontsize=13)
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.legend(loc='best')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[+] Saved t-SNE plot to {save_path}")


def plot_augmentation_delta(delta_dict: Dict[str, float], family_names: List[str], save_path: str = "results/augmentation_delta.png"):
    """
    Plots horizontal bar chart of F1 score changes after augmentation (Figure 7 in Paper).
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(10, 6))

    classes = list(delta_dict.keys())
    deltas = [delta_dict[c] for c in classes]
    y_pos = np.arange(len(classes))

    colors = ['#2ca02c' if d >= 0 else '#d62728' for d in deltas]

    plt.barh(y_pos, deltas, color=colors, align='center', alpha=0.85)
    plt.axvline(0, color='black', linewidth=1)
    plt.yticks(y_pos, [family_names[int(c)] if int(c) < len(family_names) else f"Family {c}" for c in classes])
    plt.xlabel("Delta F1 Score (Augmented - Baseline)")
    plt.title("Differences in Malware Classification with Threshold Augmentation", fontsize=13)
    plt.grid(axis='x', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[+] Saved Augmentation Delta chart to {save_path}")
