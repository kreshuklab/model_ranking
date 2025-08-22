#!/usr/bin/env python3
"""
Example script demonstrating how to use the feature analysis functions.

This script shows how to:
1. Load precomputed features from H5 files
2. Reshape features for analysis
3. Generate UMAP and t-SNE visualizations
4. Calculate intra-class variances and inter-class distances
"""

import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Any


from model_ranking.feature_analysis import (
    # plot_umap_features,
    # plot_tsne_features,
    # calculate_intra_class_variance,
    # calculate_inter_class_distances,
    analyze_feature_space,
    reshape_features_for_analysis,
)
from model_ranking.utils import load_h5
from model_ranking.feature_ranking import get_precomputed_feature_path


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
    if output_dir is None:
        output_dir = f"feature_analysis_{model_name}_to_{target}"
        if patch_index is not None:
            output_dir += f"_patch_{patch_index}"

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


def compare_patches_analysis(
    model_name: str = "E_model_NA2",
    target: str = "EPFL",
    layer_key: str = "decoders.2",
    base_path: str = "/scratch/talks/sampled_features/semantic_segmentation/mitochondria/1k_pixels_sampled",
    patch_indices: List[int] = [0, 1, 2],
    output_dir: Optional[str] = None,
):
    """
    Compare feature analysis across different patches.
    """
    print(f"=== Comparing Patches for {model_name} -> {target} ===")

    if output_dir is None:
        output_dir = f"patch_comparison_{model_name}_to_{target}"

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    patch_results: Dict[int, Dict[str, Any]] = {}

    for patch_idx in patch_indices:
        print(f"\n--- Analyzing Patch {patch_idx} ---")
        patch_output_dir = Path(output_dir) / f"patch_{patch_idx}"

        results = analyze_transfer_features(
            model_name=model_name,
            target=target,
            layer_key=layer_key,
            base_path=base_path,
            output_dir=str(patch_output_dir),
            patch_index=patch_idx,
            max_samples=5000,  # Smaller sample size for comparison
        )

        if results:
            patch_results[patch_idx] = results

    # Compare intra-class variances across patches
    print(f"\n=== Patch Comparison Summary ===")
    if patch_results:
        print("Intra-class variances by patch:")
        for patch_idx, results in patch_results.items():
            print(f"  Patch {patch_idx}:")
            for class_label, variance in results["intra_class_variances"].items():
                print(f"    Class {class_label}: {variance:.6f}")

    return patch_results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Analyze precomputed transfer features"
    )
    _ = parser.add_argument("--model", default="E_model_NA2", help="Source model name")
    _ = parser.add_argument("--target", default="EPFL", help="Target dataset name")
    _ = parser.add_argument("--layer", default="decoders.2", help="Layer key")
    _ = parser.add_argument(
        "--base_path",
        default="/scratch/talks/sampled_features/semantic_segmentation/mitochondria/1k_pixels_sampled",
        help="Base path to feature files",
    )
    _ = parser.add_argument("--output_dir", help="Output directory for results")
    _ = parser.add_argument("--patch_index", type=int, help="Specific patch to analyze")
    _ = parser.add_argument(
        "--max_samples", type=int, default=10000, help="Maximum samples for analysis"
    )
    _ = parser.add_argument(
        "--compare_patches",
        action="store_true",
        help="Compare multiple patches instead of single analysis",
    )

    args = parser.parse_args()

    if args.compare_patches:
        _ = compare_patches_analysis(
            model_name=args.model,
            target=args.target,
            layer_key=args.layer,
            base_path=args.base_path,
            output_dir=args.output_dir,
        )
    else:
        _ = analyze_transfer_features(
            model_name=args.model,
            target=args.target,
            layer_key=args.layer,
            base_path=args.base_path,
            output_dir=args.output_dir,
            patch_index=args.patch_index,
            max_samples=args.max_samples,
        )
