import numpy as np
from typing import Dict, Tuple, Any
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score, classification_report
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split

def evaluate_binary_classification(real_data: np.ndarray, synthetic_data: np.ndarray) -> Dict[str, float]:
    """
    Metric 1: Binary Classification (Real vs Synthetic).
    Labels: Real = 1, Synthetic = 0.
    Ideal F1-Score: 0.5 (Discriminators cannot distinguish between real and synthetic).
    Classifiers evaluated: Random Forest, SVM, MLP.
    """
    n_real = len(real_data)
    n_syn = len(synthetic_data)

    X = np.vstack([real_data, synthetic_data])
    y = np.array([1] * n_real + [0] * n_syn)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    models = {
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
        "SVM": SVC(kernel='rbf', probability=True, random_state=42),
        "MLP": MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=200, random_state=42)
    }

    results = {}
    for name, clf in models.items():
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
        results[name] = float(f1_score(y_test, preds, average='macro'))

    return results


def evaluate_synthetic_only_training(synthetic_X: np.ndarray, synthetic_y: np.ndarray,
                                    real_X_test: np.ndarray, real_y_test: np.ndarray) -> Dict[str, float]:
    """
    Metric 3: Synthetic-Only Training Evaluation.
    Train classifiers exclusively on synthetic samples, test on real malware data.
    Adapts the concept of Frechet Inception Distance (FID).
    """
    models = {
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
        "SVM": SVC(kernel='rbf', random_state=42),
        "MLP": MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=200, random_state=42)
    }

    results = {}
    for name, clf in models.items():
        clf.fit(synthetic_X, synthetic_y)
        preds = clf.predict(real_X_test)
        results[name] = float(f1_score(real_y_test, preds, average='weighted', zero_division=0))

    return results


def evaluate_fidelity_score(real_X_train: np.ndarray, real_y_train: np.ndarray,
                            synthetic_X: np.ndarray, synthetic_y: np.ndarray) -> Dict[str, float]:
    """
    Metric 4: Fidelity Score.
    Train classifiers on real malware data, test exclusively on synthetic samples.
    High F1-Score indicates high realism and consistency of generated samples.
    """
    models = {
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
        "SVM": SVC(kernel='rbf', random_state=42),
        "MLP": MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=200, random_state=42)
    }

    results = {}
    for name, clf in models.items():
        clf.fit(real_X_train, real_y_train)
        preds = clf.predict(synthetic_X)
        results[name] = float(f1_score(synthetic_y, preds, average='weighted', zero_division=0))

    return results


def evaluate_threshold_augmentation(real_X_train: np.ndarray, real_y_train: np.ndarray,
                                   real_X_test: np.ndarray, real_y_test: np.ndarray,
                                   synthetic_dict: Dict[int, np.ndarray],
                                   threshold: float = 0.95) -> Dict[str, Any]:
    """
    Metric 5: Threshold-Based Malware Classification Augmentation.
    - Baseline: Train RF on real_X_train, evaluate per-class F1 on real_X_test.
    - Identify classes where baseline F1 < threshold.
    - Augment only those underperforming classes with synthetic data.
    - Retrain RF and compare overall F1 and per-class delta.
    """
    # 1. Baseline training
    rf_baseline = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_baseline.fit(real_X_train, real_y_train)
    baseline_preds = rf_baseline.predict(real_X_test)

    unique_classes = np.unique(real_y_test)
    class_f1_baseline = f1_score(real_y_test, baseline_preds, average=None, labels=unique_classes, zero_division=0)
    baseline_overall_f1 = float(f1_score(real_y_test, baseline_preds, average='weighted', zero_division=0))

    # 2. Augment classes below threshold
    aug_X_list = [real_X_train]
    aug_y_list = [real_y_train]
    augmented_classes = []

    for cls_idx, f1 in zip(unique_classes, class_f1_baseline):
        if f1 < threshold and cls_idx in synthetic_dict:
            syn_samples = synthetic_dict[cls_idx]
            if len(syn_samples) > 0:
                aug_X_list.append(syn_samples)
                aug_y_list.append(np.full(len(syn_samples), cls_idx))
                augmented_classes.append(int(cls_idx))

    X_train_augmented = np.vstack(aug_X_list)
    y_train_augmented = np.concatenate(aug_y_list)

    # 3. Retrain on augmented set
    rf_aug = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_aug.fit(X_train_augmented, y_train_augmented)
    aug_preds = rf_aug.predict(real_X_test)

    class_f1_augmented = f1_score(real_y_test, aug_preds, average=None, labels=unique_classes, zero_division=0)
    augmented_overall_f1 = float(f1_score(real_y_test, aug_preds, average='weighted', zero_division=0))

    return {
        "baseline_overall_f1": baseline_overall_f1,
        "augmented_overall_f1": augmented_overall_f1,
        "f1_improvement": augmented_overall_f1 - baseline_overall_f1,
        "augmented_classes": augmented_classes,
        "per_class_baseline": dict(zip(map(int, unique_classes), map(float, class_f1_baseline))),
        "per_class_augmented": dict(zip(map(int, unique_classes), map(float, class_f1_augmented)))
    }


def compute_cosine_robustness(real_data: np.ndarray, synthetic_data: np.ndarray) -> Dict[str, float]:
    """
    Metric 6: Cosine Similarity & Robustness.
    Calculates cross-comparison cosine similarity:
    - Baseline: Real vs Real (internal pairwise similarity)
    - Synthetic: Real vs Synthetic
    Computes median deviation from baseline.
    """
    # Sample subset for fast matrix calculation if data is large
    n = min(len(real_data), 300)
    idx_real = np.random.choice(len(real_data), n, replace=False)
    real_sub = real_data[idx_real]

    m = min(len(synthetic_data), 300)
    idx_syn = np.random.choice(len(synthetic_data), m, replace=False)
    syn_sub = synthetic_data[idx_syn]

    # Baseline Real vs Real
    sim_real_real = cosine_similarity(real_sub, real_sub)
    # Mask out self-similarity diagonal
    np.fill_diagonal(sim_real_real, np.nan)
    baseline_median = float(np.nanmedian(sim_real_real))

    # Real vs Synthetic
    sim_real_syn = cosine_similarity(real_sub, syn_sub)
    syn_median = float(np.median(sim_real_syn))
    deviation = float(abs(syn_median - baseline_median))

    return {
        "baseline_median_sim": baseline_median,
        "synthetic_median_sim": syn_median,
        "median_deviation": deviation
    }
