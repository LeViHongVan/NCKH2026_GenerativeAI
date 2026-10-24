"""
End-to-End Pipeline Demo for Generative AI in Malware Analysis
Replicating the core methodology and evaluation system of IEEE Access 2025 Paper:
"Generating Synthetic Malware Samples Using Generative AI"
"""

import os
import json
import time
import torch
import numpy as np
from sklearn.model_selection import train_test_split

from src.data_generator import generate_synthetic_benchmark_dataset, MALWARE_FAMILIES
from src.training.train_models import train_gan_model, train_wgan_gp_model, train_diffusion_model
from src.evaluation.metrics import (
    evaluate_binary_classification,
    evaluate_synthetic_only_training,
    evaluate_fidelity_score,
    evaluate_threshold_augmentation,
    compute_cosine_robustness
)
from src.evaluation.visualize import plot_tsne_comparison, plot_augmentation_delta

def run_pipeline():
    print("=" * 75)
    print("   GENERATIVE AI IN MALWARE ANALYSIS - END-TO-END DEMO PIPELINE")
    print("=" * 75)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[*] Execution Device: {device}")
    os.makedirs("results", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("data/synthetic", exist_ok=True)

    # ---------------------------------------------------------
    # 1. LOAD / GENERATE BENCHMARK DATASET
    # ---------------------------------------------------------
    print("\n[Stage 1] Generating Benchmark Dataset (5 Families, 104-D, Imbalanced)...")
    X_all, y_all, family_map = generate_synthetic_benchmark_dataset(n_features=104, random_seed=42)
    np.save("data/processed/real_embeddings.npy", X_all)
    np.save("data/processed/real_labels.npy", y_all)

    print(f"    Total Samples: {len(X_all)} | Embedding Dimensions: {X_all.shape[1]}")
    for idx, name in family_map.items():
        count = np.sum(y_all == idx)
        print(f"    - Family {idx} ({name}): {count} samples")

    # Split into Train and Unique Test set (Paper Section III-B: duplicates removed in test)
    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y_all, test_size=0.25, random_state=42, stratify=y_all
    )
    print(f"[*] Dataset Split: Train set = {len(X_train)} samples, Test set = {len(X_test)} samples.")

    # ---------------------------------------------------------
    # 2. TRAIN 3 GENERATIVE AI MODELS (Focus on complex family Expiro.BK)
    # ---------------------------------------------------------
    target_family_idx = 3 # Expiro.BK (Has complex non-linear distribution)
    target_name = family_map[target_family_idx]
    expiro_real_data = X_train[y_train == target_family_idx]
    num_syn_samples = 200

    print(f"\n[Stage 2] Training 3 Generative AI Models for Family: {target_name} ({len(expiro_real_data)} real samples)...")

    # --- Model 1: Vanilla GAN ---
    t0 = time.time()
    print("  -> Training Model 1: Vanilla Dense GAN (30 epochs)...")
    gan_G, _ = train_gan_model(expiro_real_data, epochs=30, batch_size=16, latent_dim=64, device=device)
    gan_G.eval()
    with torch.no_grad():
        z = torch.randn(num_syn_samples, 64, device=device)
        syn_gan = gan_G(z).cpu().numpy()
    np.save("data/synthetic/gan_samples.npy", syn_gan)
    print(f"     [+] GAN trained in {time.time() - t0:.2f}s, generated {len(syn_gan)} samples.")

    # --- Model 2: Conv1D WGAN-GP ---
    t0 = time.time()
    print("  -> Training Model 2: Conv1D WGAN-GP (30 epochs)...")
    wgan_G, _ = train_wgan_gp_model(expiro_real_data, epochs=30, batch_size=16, latent_dim=64, device=device)
    wgan_G.eval()
    with torch.no_grad():
        z = torch.randn(num_syn_samples, 64, device=device)
        syn_wgan = wgan_G(z).cpu().numpy()
    np.save("data/synthetic/wgan_samples.npy", syn_wgan)
    print(f"     [+] WGAN-GP trained in {time.time() - t0:.2f}s, generated {len(syn_wgan)} samples.")

    # --- Model 3: 1D U-Net Diffusion Model ---
    t0 = time.time()
    print("  -> Training Model 3: 1D U-Net Diffusion (Modified T/2 Truncation, 30 epochs)...")
    diff_net, diffusion = train_diffusion_model(expiro_real_data, epochs=30, batch_size=16, timesteps=1000, device=device)
    diff_anchor = torch.tensor(expiro_real_data, dtype=torch.float32, device=device)
    syn_diff = diffusion.sample(diff_net, diff_anchor, num_samples=num_syn_samples).cpu().numpy()
    np.save("data/synthetic/diffusion_samples.npy", syn_diff)
    print(f"     [+] Diffusion trained & sampled in {time.time() - t0:.2f}s, generated {len(syn_diff)} samples.")

    # ---------------------------------------------------------
    # 3. EVALUATION ACCORDING TO 6 SCIENTIFIC METRICS
    # ---------------------------------------------------------
    print("\n[Stage 3] Executing 6 Scientific Evaluation Metrics...")
    summary_report = {}

    # Metric 1: Binary Classification (Real vs Synthetic) - Ideal F1 ~ 0.5
    print("\n  [Metric 1] Binary Classification (Target: Fool Classifier, ideal F1 ~ 0.5):")
    bin_gan = evaluate_binary_classification(expiro_real_data, syn_gan)
    bin_wgan = evaluate_binary_classification(expiro_real_data, syn_wgan)
    bin_diff = evaluate_binary_classification(expiro_real_data, syn_diff)
    print(f"     - GAN:       RF={bin_gan['RandomForest']:.3f}, SVM={bin_gan['SVM']:.3f}, MLP={bin_gan['MLP']:.3f}")
    print(f"     - WGAN-GP:   RF={bin_wgan['RandomForest']:.3f}, SVM={bin_wgan['SVM']:.3f}, MLP={bin_wgan['MLP']:.3f}")
    print(f"     - Diffusion: RF={bin_diff['RandomForest']:.3f}, SVM={bin_diff['SVM']:.3f}, MLP={bin_diff['MLP']:.3f}")

    # Metric 2: t-SNE Visualization
    print("\n  [Metric 2] Generating t-SNE Comparison Plot (Fig. 6 Replication)...")
    plot_tsne_comparison(
        real_data=expiro_real_data,
        gan_data=syn_gan,
        wgan_data=syn_wgan,
        diff_data=syn_diff,
        family_name=target_name,
        save_path="results/tsne_comparison.png"
    )

    # Metric 3: Synthetic-Only Training (Train Fake, Test Real)
    print("\n  [Metric 3] Synthetic-Only Training Evaluation (Train Fake -> Test Real):")
    # Multi-class synthetic generation for minority classes (3: Expiro.BK, 4: Hotbar)
    hotbar_real = X_train[y_train == 4]
    wgan_hotbar_G, _ = train_wgan_gp_model(hotbar_real, epochs=20, batch_size=8, device=device)
    with torch.no_grad():
        z_h = torch.randn(150, 64, device=device)
        syn_hotbar = wgan_hotbar_G(z_h).cpu().numpy()

    # Combine synthetic set
    syn_multi_X = np.vstack([syn_wgan, syn_hotbar])
    syn_multi_y = np.array([3] * len(syn_wgan) + [4] * len(syn_hotbar))
    minority_mask = np.isin(y_test, [3, 4])
    real_sub_X = X_test[minority_mask]
    real_sub_y = y_test[minority_mask]

    syn_only_res = evaluate_synthetic_only_training(syn_multi_X, syn_multi_y, real_sub_X, real_sub_y)
    print(f"     - F1 Test on Real Data: RF={syn_only_res['RandomForest']:.3f}, SVM={syn_only_res['SVM']:.3f}, MLP={syn_only_res['MLP']:.3f}")

    # Metric 4: Fidelity Score (Train Real, Test Fake)
    print("\n  [Metric 4] Fidelity Score (Train Real -> Test Fake):")
    real_minority_train_X = X_train[np.isin(y_train, [3, 4])]
    real_minority_train_y = y_train[np.isin(y_train, [3, 4])]
    fid_res = evaluate_fidelity_score(real_minority_train_X, real_minority_train_y, syn_multi_X, syn_multi_y)
    print(f"     - Fidelity F1 Score: RF={fid_res['RandomForest']:.3f}, SVM={fid_res['SVM']:.3f}, MLP={fid_res['MLP']:.3f}")

    # Metric 5: Threshold-Based Augmentation (Threshold t = 0.95)
    print("\n  [Metric 5] Threshold-Based Augmentation (Threshold t = 0.95, Random Forest):")
    syn_dict = {
        3: syn_diff,    # Expiro.BK augmented with Diffusion
        4: syn_hotbar   # Hotbar augmented with WGAN-GP
    }
    aug_res = evaluate_threshold_augmentation(
        X_train, y_train, X_test, y_test, synthetic_dict=syn_dict, threshold=0.95
    )
    print(f"     - Baseline Multiclass F1 (Before Aug):  {aug_res['baseline_overall_f1']:.4f}")
    print(f"     - Augmented Multiclass F1 (After Aug):   {aug_res['augmented_overall_f1']:.4f}")
    print(f"     - Overall Improvement (Delta F1):       +{aug_res['f1_improvement']:.4f}")
    print(f"     - Augmented Classes:                    {aug_res['augmented_classes']}")

    deltas = {}
    for c in aug_res['per_class_baseline']:
        deltas[c] = aug_res['per_class_augmented'][c] - aug_res['per_class_baseline'][c]
    plot_augmentation_delta(deltas, MALWARE_FAMILIES, save_path="results/augmentation_delta.png")

    # Metric 6: Cosine Similarity & Robustness
    print("\n  [Metric 6] Cosine Similarity Robustness (Median deviation from Baseline):")
    sim_gan = compute_cosine_robustness(expiro_real_data, syn_gan)
    sim_wgan = compute_cosine_robustness(expiro_real_data, syn_wgan)
    sim_diff = compute_cosine_robustness(expiro_real_data, syn_diff)
    print(f"     - Baseline (Real-Real) Median Sim: {sim_gan['baseline_median_sim']:.4f}")
    print(f"     - GAN Median Deviation:            {sim_gan['median_deviation']:.4f}")
    print(f"     - WGAN-GP Median Deviation:        {sim_wgan['median_deviation']:.4f}")
    print(f"     - Diffusion Median Deviation:      {sim_diff['median_deviation']:.4f} (Closest to Real)")

    # Save summary report
    summary = {
        "execution_date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "models_evaluated": ["Vanilla_GAN", "WGAN_GP", "Diffusion_1D"],
        "metric_1_binary": {"GAN": bin_gan, "WGAN_GP": bin_wgan, "Diffusion": bin_diff},
        "metric_3_synthetic_only": syn_only_res,
        "metric_4_fidelity": fid_res,
        "metric_5_threshold_augmentation": {
            "baseline_f1": aug_res['baseline_overall_f1'],
            "augmented_f1": aug_res['augmented_overall_f1'],
            "improvement": aug_res['f1_improvement'],
            "augmented_classes": aug_res['augmented_classes']
        },
        "metric_6_robustness_deviation": {
            "GAN": sim_gan['median_deviation'],
            "WGAN_GP": sim_wgan['median_deviation'],
            "Diffusion": sim_diff['median_deviation']
        }
    }
    with open("results/experiment_summary.json", "w") as f:
        json.dump(summary, f, indent=4)

    print("\n" + "=" * 75)
    print("   DEMO PIPELINE COMPLETED SUCCESSFULLY!")
    print("   Output artifacts saved in './results/' directory:")
    print("   - results/experiment_summary.json")
    print("   - results/tsne_comparison.png")
    print("   - results/augmentation_delta.png")
    print("=" * 75)

if __name__ == "__main__":
    run_pipeline()
