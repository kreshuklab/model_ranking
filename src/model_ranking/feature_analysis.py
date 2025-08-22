"""
Feature space analysis utilities for model ranking.

This module provides functions for visualizing and analyzing feature spaces
extracted from pretrained models, including dimensionality reduction,
variance analysis, and distance calculations.
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple, Optional, Any
from numpy.typing import NDArray
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.metrics import (
    silhouette_score,  # pyright: ignore[reportUnknownVariableType]
)
import json
import warnings
from umap import UMAP  # pyright: ignore[reportMissingTypeStubs]

from model_ranking.feature_ranking import get_precomputed_feature_path
from model_ranking.utils import is_ndarray, load_h5

# Reset to matplotlib/seaborn defaults
plt.rcdefaults()  # Reset all rcParams to matplotlib defaults
sns.reset_defaults()  # Reset seaborn to defaults
sns.reset_orig()  # Reset to original matplotlib defaults


def plot_umap_features(
    features: NDArray[Any],
    labels: NDArray[Any],
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    n_components: int = 2,
    random_state: int = 42,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8),
    alpha: float = 0.7,
    s: int = 1,
) -> plt.Figure:  # type: ignore[name-defined]
    """
    Plot UMAP visualization of features colored by label class.

    Args:
        features: Feature array of shape (n_samples, n_features)
        labels: Label array of shape (n_samples,)
        n_neighbors: Number of nearest neighbors for UMAP
        min_dist: Minimum distance parameter for UMAP
        n_components: Number of components for UMAP (2 or 3)
        random_state: Random state for reproducibility
        title: Optional title for the plot
        save_path: Optional path to save the plot
        figsize: Figure size tuple
        alpha: Transparency of points
        s: Size of points

    Returns:
        matplotlib.figure.Figure: The created figure
    """

    # Validate inputs
    if features.ndim != 2:
        raise ValueError(f"Features must be 2D array, got shape {features.shape}")
    if labels.ndim != 1:
        raise ValueError(f"Labels must be 1D array, got shape {labels.shape}")
    if len(features) != len(labels):
        raise ValueError(
            f"Features and labels must have same length: {len(features)} vs {len(labels)}"
        )

    # Standardize features for better UMAP performance
    scaler = StandardScaler()
    features_scaled = (  # pyright: ignore[reportUnknownVariableType]
        scaler.fit_transform(features)
    )

    assert is_ndarray(features_scaled), "Features must be a numpy array."

    # Fit UMAP
    print(
        f"Computing UMAP embedding with {len(features)} samples and {features.shape[1]} features..."
    )
    umap_model = UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        random_state=random_state,
        verbose=True,
    )

    embedding = umap_model.fit_transform(  # pyright: ignore[reportUnknownVariableType]
        features_scaled
    )

    # Create plot
    fig, ax = plt.subplots(figsize=figsize)

    # Get unique classes and create colormap
    unique_classes = np.unique(labels)
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_classes)))  # type: ignore[attr-defined]

    # Plot each class separately for better legend
    for i, class_label in enumerate(unique_classes):
        mask = labels == class_label
        if n_components == 2:
            _ = ax.scatter(
                embedding[mask, 0],  # pyright: ignore
                embedding[mask, 1],  # pyright: ignore
                c=[colors[i]],  # pyright: ignore
                label=f"Class {class_label}",
                alpha=alpha,
                s=s,
            )
        elif n_components == 3:
            # For 3D, we need a 3D subplot
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111, projection="3d")
            _ = ax.scatter(  # pyright: ignore
                embedding[mask, 0],  # pyright: ignore
                embedding[mask, 1],  # pyright: ignore
                embedding[mask, 2],  # pyright: ignore
                c=[colors[i]],
                label=f"Class {class_label}",
                alpha=alpha,
                s=s,  # pyright: ignore
            )

    # Customize plot
    if title is None:
        title = f"UMAP Visualization of Features ({len(features)} samples)"
    _ = ax.set_title(title, fontsize=14, fontweight="bold")

    if n_components == 2:
        _ = ax.set_xlabel("UMAP 1", fontsize=12)
        _ = ax.set_ylabel("UMAP 2", fontsize=12)
    elif n_components == 3:
        _ = ax.set_xlabel("UMAP 1", fontsize=12)
        _ = ax.set_ylabel("UMAP 2", fontsize=12)
        _ = ax.set_zlabel("UMAP 3", fontsize=12)  # type: ignore[attr-defined]

    _ = ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()

    # Save if path provided
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"UMAP plot saved to {save_path}")

    return fig


def plot_tsne_features(
    features: NDArray[Any],
    labels: NDArray[Any],
    perplexity: float = 30.0,
    n_components: int = 2,
    random_state: int = 42,
    n_iter: int = 1000,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8),
    alpha: float = 0.7,
    s: int = 1,
) -> plt.Figure:  # type: ignore[name-defined]
    """
    Plot t-SNE visualization of features colored by label class.

    Args:
        features: Feature array of shape (n_samples, n_features)
        labels: Label array of shape (n_samples,)
        perplexity: Perplexity parameter for t-SNE
        n_components: Number of components for t-SNE (2 or 3)
        random_state: Random state for reproducibility
        n_iter: Number of iterations for t-SNE
        title: Optional title for the plot
        save_path: Optional path to save the plot
        figsize: Figure size tuple
        alpha: Transparency of points
        s: Size of points

    Returns:
        matplotlib.figure.Figure: The created figure
    """

    # Validate inputs
    if features.ndim != 2:
        raise ValueError(f"Features must be 2D array, got shape {features.shape}")
    if labels.ndim != 1:
        raise ValueError(f"Labels must be 1D array, got shape {labels.shape}")
    if len(features) != len(labels):
        raise ValueError(
            f"Features and labels must have same length: {len(features)} vs {len(labels)}"
        )

    # Check if perplexity is appropriate for dataset size
    if perplexity >= len(features) / 3:
        new_perplexity = max(5, len(features) // 4)
        warnings.warn(
            f"Perplexity {perplexity} too large for {len(features)} samples. Using {new_perplexity}"
        )
        perplexity = new_perplexity

    # Standardize features for better t-SNE performance
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)  # pyright: ignore

    assert is_ndarray(features_scaled), "Features must be a numpy array."

    # Fit t-SNE
    print(
        f"Computing t-SNE embedding with {len(features)} samples and {features.shape[1]} features..."
    )
    tsne_model = TSNE(
        perplexity=perplexity,
        n_components=n_components,
        random_state=random_state,
        # n_iter=n_iter,
        verbose=1,
    )

    embedding = tsne_model.fit_transform(features_scaled)  # pyright: ignore

    # Create plot
    fig, ax = plt.subplots(figsize=figsize)

    # Get unique classes and create colormap
    unique_classes = np.unique(labels)
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_classes)))  # type: ignore[attr-defined]

    # Plot each class separately for better legend
    for i, class_label in enumerate(unique_classes):
        mask = labels == class_label
        if n_components == 2:
            _ = ax.scatter(
                embedding[mask, 0],
                embedding[mask, 1],
                c=[colors[i]],  # pyright: ignore
                label=f"Class {class_label}",
                alpha=alpha,
                s=s,
            )
        elif n_components == 3:
            # For 3D, we need a 3D subplot
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111, projection="3d")
            _ = ax.scatter(  # pyright: ignore
                embedding[mask, 0],
                embedding[mask, 1],
                embedding[mask, 2],
                c=[colors[i]],
                label=f"Class {class_label}",
                alpha=alpha,
                s=s,  # pyright: ignore
            )

    # Customize plot
    if title is None:
        title = f"t-SNE Visualization of Features ({len(features)} samples)"
    _ = ax.set_title(title, fontsize=14, fontweight="bold")

    if n_components == 2:
        _ = ax.set_xlabel("t-SNE 1", fontsize=12)
        _ = ax.set_ylabel("t-SNE 2", fontsize=12)
    elif n_components == 3:
        _ = ax.set_xlabel("t-SNE 1", fontsize=12)
        _ = ax.set_ylabel("t-SNE 2", fontsize=12)
        _ = ax.set_zlabel("t-SNE 3", fontsize=12)  # type: ignore[attr-defined]

    _ = ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()

    # Save if path provided
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"t-SNE plot saved to {save_path}")

    return fig


def calculate_intra_class_variance(
    features: NDArray[Any],
    labels: NDArray[Any],
) -> Dict[int, float]:
    """
    Calculate intra-class variance for each class.

    Computes the average squared distance of each feature vector from its class mean.

    Args:
        features: Feature array of shape (n_samples, n_features)
        labels: Label array of shape (n_samples,)

    Returns:
        Dict mapping class labels to their intra-class variance
    """
    # Validate inputs
    if features.ndim != 2:
        raise ValueError(f"Features must be 2D array, got shape {features.shape}")
    if labels.ndim != 1:
        raise ValueError(f"Labels must be 1D array, got shape {labels.shape}")
    if len(features) != len(labels):
        raise ValueError(
            f"Features and labels must have same length: {len(features)} vs {len(labels)}"
        )

    intra_class_variances: Dict[int, float] = {}
    unique_classes = np.unique(labels)

    for class_label in unique_classes:
        # Get features for this class
        class_mask = labels == class_label
        class_features = features[class_mask]

        if len(class_features) == 0:
            intra_class_variances[int(class_label)] = 0.0
            continue

        # Calculate class mean
        class_mean = np.mean(class_features, axis=0)

        # Calculate squared distances from mean
        squared_distances = np.sum((class_features - class_mean) ** 2, axis=1)

        # Average squared distance is the intra-class variance
        intra_class_variance = np.mean(squared_distances)  # pyright: ignore
        intra_class_variances[int(class_label)] = float(
            intra_class_variance  # pyright: ignore
        )

    return intra_class_variances


def calculate_inter_class_distances(
    features: NDArray[Any], labels: NDArray[Any], distance_metric: str = "euclidean"
) -> Dict[str, float]:
    """
    Calculate inter-class distances between class means.

    Args:
        features: Feature array of shape (n_samples, n_features)
        labels: Label array of shape (n_samples,)
        distance_metric: Distance metric to use ('euclidean', 'cosine', 'manhattan')

    Returns:
        Dict mapping class pairs to their inter-class distance
    """
    # Validate inputs
    if features.ndim != 2:
        raise ValueError(f"Features must be 2D array, got shape {features.shape}")
    if labels.ndim != 1:
        raise ValueError(f"Labels must be 1D array, got shape {labels.shape}")
    if len(features) != len(labels):
        raise ValueError(
            f"Features and labels must have same length: {len(features)} vs {len(labels)}"
        )

    unique_classes = np.unique(labels)

    # Calculate class means
    class_means: Dict[int, NDArray[Any]] = {}
    for class_label in unique_classes:
        class_mask = labels == class_label
        class_features = features[class_mask]
        if len(class_features) > 0:
            class_means[int(class_label)] = np.mean(class_features, axis=0)

    # Calculate pairwise distances between class means
    inter_class_distances: Dict[str, float] = {}

    for i, class1 in enumerate(unique_classes):
        for j, class2 in enumerate(unique_classes):
            if i < j:  # Only calculate upper triangle to avoid duplicates
                if int(class1) in class_means and int(class2) in class_means:
                    mean1 = class_means[int(class1)]
                    mean2 = class_means[int(class2)]

                    if distance_metric == "euclidean":
                        distance = np.sqrt(np.sum((mean1 - mean2) ** 2))
                    elif distance_metric == "cosine":
                        # Cosine distance = 1 - cosine similarity
                        dot_product = np.dot(mean1, mean2)
                        norms = np.linalg.norm(mean1) * np.linalg.norm(mean2)
                        if norms > 0:
                            cosine_similarity = dot_product / norms
                            distance = 1 - cosine_similarity
                        else:
                            distance = 1.0
                    elif distance_metric == "manhattan":
                        distance = np.sum(np.abs(mean1 - mean2))
                    else:
                        raise ValueError(f"Unknown distance metric: {distance_metric}")

                    inter_class_distances[f"{int(class1)}to{int(class2)}"] = float(
                        distance
                    )

    return inter_class_distances


def calculate_silhouette_score(
    features: NDArray[Any],
    labels: NDArray[Any],
    metric: str = "euclidean",
    sample_size: Optional[int] = None,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Calculate silhouette score to quantify clustering quality.

    The silhouette score measures how similar an object is to its own cluster
    compared to other clusters. Ranges from -1 to +1, where:
    - +1: Sample is far from neighboring clusters (good clustering)
    - 0: Sample is on or very close to decision boundary between clusters
    - -1: Sample might have been assigned to wrong cluster

    Args:
        features: Feature array of shape (n_samples, n_features)
        labels: Label array of shape (n_samples,) - the cluster assignments
        metric: Distance metric to use ('euclidean', 'manhattan', 'cosine', etc.)
        sample_size: If provided, subsample this many points for efficiency
        random_state: Random state for reproducible subsampling

    Returns:
        Dictionary containing:
        - 'overall_score': Mean silhouette score across all samples
        - 'n_samples': Number of samples used in calculation
        - 'n_classes': Number of unique classes
    """
    # Validate inputs
    if features.ndim != 2:
        raise ValueError(f"Features must be 2D array, got shape {features.shape}")
    if labels.ndim != 1:
        raise ValueError(f"Labels must be 1D array, got shape {labels.shape}")
    if len(features) != len(labels):
        raise ValueError(
            f"Features and labels must have same length: {len(features)} vs {len(labels)}"
        )

    unique_classes = np.unique(labels)
    n_classes = len(unique_classes)

    # Need at least 2 classes for silhouette score
    if n_classes < 2:
        print(
            f"Warning: Only {n_classes} unique class found. Silhouette score requires at least 2 classes."
        )
        return {
            "overall_score": 0.0,
            "per_class_scores": {int(unique_classes[0]): 0.0} if n_classes == 1 else {},
            "per_sample_scores": np.zeros(len(features)),
            "n_samples": len(features),
            "n_classes": n_classes,
        }

    # Subsample if requested for efficiency
    if sample_size and len(features) > sample_size:
        print(
            f"Subsampling from {len(features)} to {sample_size} samples for silhouette calculation..."
        )
        np.random.seed(random_state)
        indices = np.random.choice(len(features), sample_size, replace=False)
        features_subset = features[indices]
        labels_subset = labels[indices]
        print(f"Subsampled dataset has {len(np.unique(labels_subset))} classes")
    else:
        features_subset = features
        labels_subset = labels
        indices = np.arange(len(features))

    # Calculate overall silhouette score
    try:
        overall_score = silhouette_score(features_subset, labels_subset, metric=metric)
        print(f"Overall silhouette score: {overall_score:.4f}")
    except Exception as e:
        print(f"Error calculating overall silhouette score: {e}")
        overall_score = 0.0

    # Print detailed results
    print("\nSilhouette Score Analysis:")
    print(f"  Overall score: {overall_score:.4f}")

    return {
        "overall_score": float(overall_score),
        "n_samples": len(features_subset),
        "n_classes": len(np.unique(labels_subset)),
    }


def analyze_feature_space(
    features: NDArray[Any],
    labels: NDArray[Any],
    title_prefix: str = "Feature Analysis",
    save_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Comprehensive analysis of feature space including visualizations and statistics.

    Args:
        features: Feature array of shape (n_samples, n_features)
        labels: Label array of shape (n_samples,)
        title_prefix: Prefix for plot titles
        save_dir: Directory to save plots and results

    Returns:
        Dictionary containing analysis results:
        - 'intra_class_variances': Dict of intra-class variances per class
        - 'inter_class_distances': Dict of inter-class distances between class pairs
        - 'silhouette_analysis': Dict with overall and per-class silhouette scores
        - 'umap_figure': UMAP visualization figure (if save_dir provided)
        - 'tsne_figure': t-SNE visualization figure (if save_dir provided)
    """
    print(
        f"Analyzing feature space with {len(features)} samples and {features.shape[1]} features..."
    )

    results: Dict[str, Any] = {}

    # Calculate variance metrics
    print("Calculating intra-class variances...")
    intra_class_vars = calculate_intra_class_variance(features, labels)
    results["intra_class_variances"] = intra_class_vars

    print("Calculating inter-class distances...")
    inter_class_dists = calculate_inter_class_distances(features, labels)
    results["inter_class_distances"] = inter_class_dists

    print("Calculating silhouette score...")
    silhouette_results = calculate_silhouette_score(features, labels, sample_size=10000)
    results["silhouette_analysis"] = silhouette_results

    # Print summary statistics
    print("\n=== Feature Space Analysis Results ===")
    print("Intra-class variances:")
    for class_label, variance in intra_class_vars.items():
        print(f"  Class {class_label}: {variance:.6f}")

    print("\nInter-class distances (Euclidean):")
    for classes, distance in inter_class_dists.items():
        class1, class2 = classes.split("to")
        print(f"  Class {class1} <-> Class {class2}: {distance:.6f}")

    # print(f"\nSilhouette Score: {silhouette_results['overall_score']:.4f}")
    # print("Per-class silhouette scores:")
    # for class_label, score in silhouette_results["per_class_scores"].items():
    #     print(f"  Class {class_label}: {score:.4f}")

    # Generate visualizations if requested
    if save_dir:
        import os

        os.makedirs(save_dir, exist_ok=True)

        # save results to json file located in save_dir
        save_path = os.path.join(save_dir, "feature_analysis_results.json")
        with open(save_path, "w") as f:
            json.dump(results, f, indent=4)

        print("\nGenerating UMAP visualization...")
        umap_fig = plot_umap_features(
            features,
            labels,
            title=f"{title_prefix} - UMAP",
            save_path=f"{save_dir}/umap_visualization.png",
        )

        print("Generating t-SNE visualization...")
        tsne_fig = plot_tsne_features(
            features,
            labels,
            title=f"{title_prefix} - t-SNE",
            save_path=f"{save_dir}/tsne_visualization.png",
        )
        results["umap_figure"] = umap_fig
        results["tsne_figure"] = tsne_fig

        plt.close("all")  # Close figures to free memory

    return results


def reshape_features_for_analysis(
    features: NDArray[Any], labels: NDArray[Any], patch_index: Optional[int] = None
) -> Tuple[NDArray[Any], NDArray[Any]]:
    """
    Reshape features from (N_patches, Num_pixels_per_patch, Feature_dimension) format
    for analysis functions that expect (Total_features, Feature_dimension).

    Args:
        features: Features array of shape (N_patches, Num_pixels_per_patch, Feature_dimension)
        labels: Labels array of shape (N_patches, Num_pixels_per_patch)
        patch_index: If provided, analyze only this specific patch. Otherwise, analyze all patches.

    Returns:
        Tuple of (reshaped_features, reshaped_labels) ready for analysis
    """
    if features.ndim != 3:
        raise ValueError(f"Expected 3D features array, got shape {features.shape}")
    if labels.ndim != 2:
        raise ValueError(f"Expected 2D labels array, got shape {labels.shape}")

    n_patches, n_pixels, n_features = features.shape

    if patch_index is not None:
        if patch_index >= n_patches:
            raise ValueError(
                f"Patch index {patch_index} out of range for {n_patches} patches"
            )
        # Analyze single patch
        features_reshaped = features[patch_index]  # Shape: (n_pixels, n_features)
        labels_reshaped = labels[patch_index]  # Shape: (n_pixels,)
        print(f"Analyzing single patch {patch_index} with {n_pixels} pixels")
    else:
        # Analyze all patches
        features_reshaped = features.reshape(
            -1, n_features
        )  # Shape: (n_patches * n_pixels, n_features)
        labels_reshaped = labels.reshape(-1)  # Shape: (n_patches * n_pixels,)
        print(
            f"Analyzing all {n_patches} patches with {n_patches * n_pixels} total pixels"
        )

    return features_reshaped, labels_reshaped


def analyze_transfer_features(
    model_name: str = "E_model_NA2",
    target: str = "EPFL",
    layer_key: str = "decoders.2",
    base_path: str = "/scratch/talks/sampled_features/semantic_segmentation/mitochondria/1k_pixels_sampled",
    output_dir: Optional[str] = None,
    patch_index: Optional[int] = None,
    max_samples: Optional[int] = 10000,
):
    """
    Analyze features from a specific transfer (source model -> target dataset).

    Args:
        model_name: Name of the source model (e.g., "E_model_NA2")
        target: Target dataset name (e.g., "EPFL")
        layer_key: Layer from which features were extracted (e.g., "decoders.2")
        base_path: Base path where feature files are stored
        output_dir: Directory to save analysis results and plots
        patch_index: If provided, analyze only this specific patch
        max_samples: Maximum number of samples to use for analysis (for memory efficiency)
    """

    print(f"=== Analyzing Transfer: {model_name} -> {target} ===")
    print(f"Layer: {layer_key}")

    # Get the path to the precomputed features
    try:
        feature_path = get_precomputed_feature_path(
            model_name, target, base_path, filetype="h5"
        )
        print(f"Loading features from: {feature_path}")
    except Exception as e:
        print(f"Error finding feature path: {e}")
        print(
            f"Expected path pattern: {base_path}/**/{model_name}_to_{target}_features.h5"
        )
        return

    # Load features and labels
    try:
        features = load_h5(feature_path, f"{layer_key}_features")
        labels = load_h5(feature_path, f"{layer_key}_labels")
        print(f"Loaded features shape: {features.shape}")
        print(f"Loaded labels shape: {labels.shape}")
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # Reshape features for analysis
    features_reshaped, labels_reshaped = reshape_features_for_analysis(
        features, labels, patch_index=patch_index
    )

    # Subsample if dataset is too large
    if max_samples and len(features_reshaped) > max_samples:
        print(f"Subsampling from {len(features_reshaped)} to {max_samples} samples...")
        indices = np.random.choice(len(features_reshaped), max_samples, replace=False)
        features_reshaped = features_reshaped[indices]
        labels_reshaped = labels_reshaped[indices]

    # Print dataset info
    unique_labels, counts = np.unique(labels_reshaped, return_counts=True)
    print(f"\nDataset statistics:")
    print(f"Total samples: {len(features_reshaped)}")
    print(f"Feature dimension: {features_reshaped.shape[1]}")
    print(f"Classes: {unique_labels}")
    print(f"Class counts: {dict(zip(unique_labels, counts))}")

    # Set up output directory
    # if output_dir is None:
    #     output_dir = f"feature_analysis_{model_name}_to_{target}"
    #     if patch_index is not None:
    #         output_dir += f"_patch_{patch_index}"

    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Run comprehensive analysis
    analysis_title = f"{model_name} -> {target}"
    if patch_index is not None:
        analysis_title += f" (Patch {patch_index})"

    results = analyze_feature_space(
        features_reshaped,
        labels_reshaped,
        title_prefix=analysis_title,
        save_dir=output_dir,
    )

    print(f"\nAnalysis results saved to: {output_dir}")
    return results
