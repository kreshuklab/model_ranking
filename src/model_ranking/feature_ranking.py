import numpy as np
from numpy.typing import NDArray
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, List, Union
import torch
from torch.utils.data import DataLoader, ConcatDataset, Dataset

from CCFV.utils.sliding_window_sampling import (  # pyright: ignore[reportMissingTypeStubs]
    FeatureExtractor,
)
from tqdm import tqdm

from pytorch3dunet.unet3d.model import (
    get_model,  # pyright: ignore[reportUnknownVariableType]
)
from pytorch3dunet.unet3d.utils import (
    load_checkpoint,  # pyright: ignore[reportUnknownVariableType]
)

from model_ranking.dataclass import (
    TransferFeatureExtractionConfig,
    ModelSourceConfig,
    Pytorch3DUnetTrainLoaderConfig,
    mito_dataset_type,
)
from model_ranking.utils import (
    is_ndarray,
    is_torch_tensor,
    loader_classes,
)
from model_ranking.yaml_generators import (
    get_model_path,
    MODEL_ABBREVIATIONS_TO_DATASET,
)

per_layer_feature_type = Dict[str, NDArray[Any]]


def sample_pixels(
    label_map: NDArray[Any], num_samples: int = 1000, seed: Optional[int] = None
) -> NDArray[Any]:
    labels = label_map.flatten()
    classes, counts = np.unique(labels, return_counts=True)
    class_weights = {c: 1.0 / count for c, count in zip(classes, counts)}
    weights = np.array([class_weights[l] for l in labels])
    probs = weights / weights.sum()

    rng = np.random.default_rng(seed)  # Create a local random generator
    sampled_indices = rng.choice(len(labels), size=num_samples, replace=False, p=probs)

    return sampled_indices


def compute_class_frequencies(target_train_dataset: Dataset[Any]):
    """Compute the classes frequencies of a semantic segmentation dataset."""
    class_counts: Dict[int, int] = {}
    for _, labels in iter(target_train_dataset):
        assert is_ndarray(labels) or is_torch_tensor(
            labels
        ), "Labels must be a numpy array or torch tensor."

        if is_ndarray(labels):
            labels = np.reshape(labels, [-1])
            labels, counts = np.unique(labels, return_counts=True)
        elif is_torch_tensor(labels):
            labels = labels.flatten()
            labels, counts = torch.unique(  # pyright: ignore[reportUnknownVariableType]
                labels, return_counts=True
            )
        for label, count in zip(labels, counts):  # pyright: ignore
            if int(label) not in class_counts:
                class_counts[int(label)] = count
            else:
                class_counts[int(label)] += count
    return class_counts


def get_sampling_indices(
    sampling_seed: Optional[int],
    class_counts: Dict[int, int],
    labels: NDArray[Any],
    num_samples: int = 1000,
):
    """Function to generates sampling indices of the given labels.

    Args:
      sampling_seed: seed used for the sampling process.
      class_counts: instances count of every class in the dataset. If None,
        uniform sampling is applied.
      labels: ground-truth labels from the target dataset.
      num_samples: number of labels to sample from the labels.

    Returns:
      indices: indices for sampling the labels.

    """
    rng = np.random.default_rng(sampling_seed)
    num_samples = min(num_samples, labels.shape[0])  # To avoid sampling errors

    if not class_counts:  # Uniform sampling
        weights_labels = None
    else:  # Class balanced sampling
        weights_labels = [1 / class_counts[int(x)] for x in labels]
        weights_labels = weights_labels / np.sum(weights_labels)

    indices = rng.choice(
        np.arange(labels.shape[0]), num_samples, replace=False, p=weights_labels
    )
    return indices


def sample_from_image(
    outputs: NDArray[Any],
    labels: NDArray[Any],
    sampling_seed: Optional[int],
    class_counts: Dict[int, int],
    num_samples: int,
) -> tuple[NDArray[Any], NDArray[Any], NDArray[Any]]:
    """Function to sample images and labels for semantic segmentation.

    Args:
      outputs: either activations or predictions at the pixel level.
      labels: ground-truth labels from the target dataset.
      sampling_seed: seed used for the sampling process.
      class_counts: Instances count of every class in the dataset. If None, random
        sampling is applied.
      num_samples: number of labels to sample from the labels.

    Returns:
      sampled_outputs, sampled_labels, sampled_indices

    """

    outputs = np.reshape(outputs, [-1, outputs.shape[1]])
    labels = np.reshape(labels, [-1])

    # # Remove background pixels located at label==0
    # mask = np.not_equal(labels, 0)
    # labels = labels[mask]
    # outputs = outputs[mask]
    # if np.sum(mask) == 0:  # Just background pixels
    #     return np.array([]), np.array([])

    if num_samples:  # Sample num_sample pixels per image.
        sampling_indices = get_sampling_indices(
            sampling_seed, class_counts, labels, num_samples
        )

        outputs = outputs[sampling_indices]
        labels = labels[sampling_indices]

    else:  # Sample all pixels.
        sampling_indices = np.arange(labels.shape[0])

    return outputs, labels, sampling_indices


class TransferFeatureExtraction:
    def __init__(self, config: TransferFeatureExtractionConfig):
        super().__init__()
        self.source_model_cfgs = config.source_models
        self.source_model_base_path = config.source_model_base_path
        self.data_base_path = config.data_base_path
        self.feature_cfg = config.feature_cfg
        self.target_datasets, self.target_dataloaders = self.initialise_target_datasets(
            config.target_datasets
        )
        self.feature_indices, self.class_counts = self.initialise_feature_indices(
            config.target_datasets
        )

    def initialise_feature_indices(self, target_configs: Sequence[mito_dataset_type]):
        feature_indices: Dict[str, Optional[Dict[str, Optional[NDArray[Any]]]]] = {}
        class_counts: Dict[str, Optional[Dict[int, int]]] = {}
        for target_cfg in target_configs:
            if target_cfg.feature_indices_path:
                indices_file = np.load(target_cfg.feature_indices_path)
                arrays = indices_file.files
                target_indices: Dict[str, Optional[NDArray[Any]]] = {}
                for layer in self.feature_cfg.layers:
                    if f"{layer}_indices" in arrays:
                        # Load precomputed feature indices if available
                        target_indices[layer] = np.load(
                            target_cfg.feature_indices_path
                        )[f"{layer}_indices"]
                    else:
                        target_indices[layer] = None
                feature_indices[target_cfg.name] = target_indices
            else:
                feature_indices[target_cfg.name] = None

            for layer in self.feature_cfg.layers:
                if feature_indices[target_cfg.name] is None:
                    class_counts[target_cfg.name] = compute_class_frequencies(
                        self.target_datasets[target_cfg.name]
                    )

                else:
                    target_indices_temp = feature_indices[target_cfg.name]
                    assert (
                        target_indices_temp is not None
                    ), "Feature indices for target dataset must be defined."
                    target_indices: Dict[str, Optional[NDArray[Any]]] = (
                        target_indices_temp
                    )
                    if target_indices[layer] is None:
                        class_counts[target_cfg.name] = compute_class_frequencies(
                            self.target_datasets[target_cfg.name]
                        )
                    else:
                        class_counts[target_cfg.name] = None
        return feature_indices, class_counts

    def initialise_model(self, model_meta_cfg: ModelSourceConfig, model_dir_path: str):

        source_model_path = get_model_path(
            source_data=model_meta_cfg.source_name,
            model_name=model_meta_cfg.model_name,
            base_dir_path=model_dir_path,
            checkpoint_name=model_meta_cfg.checkpoint_name,
        )

        if model_meta_cfg.model_type == "UnetrWrapper":
            model_cfg = model_meta_cfg.create_unetr_config(
                feature_perturbation=None, img_size=256
            )
        else:
            model_cfg = model_meta_cfg.create_config(feature_perturbation=None)

        model = get_model(model_cfg.model_dump())

        # Load model state
        if source_model_path.endswith(".pt"):
            model_key = "model_state"
        else:
            model_key = "model_state_dict"
        load_checkpoint(source_model_path, model, model_key=model_key)

        model = model.to("cuda:0" if torch.cuda.is_available() else "cpu")

        return model.eval()

    def get_datasets(self, config: Pytorch3DUnetTrainLoaderConfig):
        # get dataset class
        dataset_cls_str = config.dataset
        dataset_class = loader_classes(dataset_cls_str)
        datasets = dataset_class.create_datasets(config.model_dump(), phase="train")
        return datasets

    def get_dataloaders(  # pyright: ignore[reportUnknownParameterType]
        self, config: Pytorch3DUnetTrainLoaderConfig
    ):
        datasets = self.get_datasets(config)

        num_workers = config.num_workers
        batch_size = config.batch_size

        # when training with volumetric data use batch_size of 1 due to GPU memory constraints
        return [  # pyright: ignore[reportUnknownVariableType]
            DataLoader(
                ConcatDataset(datasets),  # pyright: ignore[reportUnknownArgumentType]
                batch_size=batch_size,
                shuffle=False,
                pin_memory=True,
                num_workers=num_workers,
            )
        ]

    def initialise_target_datasets(self, target_configs: Sequence[mito_dataset_type]):
        target_datasets: Dict[str, ConcatDataset[Any]] = {}
        target_dataloaders: Dict[str, DataLoader[Any]] = {}
        for target_cfg in target_configs:
            target_dataset_cfg = target_cfg.feature_loader.create_config(
                output_dir=None,
                data_base_path=self.data_base_path,
                phase="train",
            )
            assert isinstance(
                target_dataset_cfg, Pytorch3DUnetTrainLoaderConfig
            ), "Expected Pytorch3DUnetTrainLoaderConfig for target dataset config."
            datasets = self.get_datasets(target_dataset_cfg)
            target_datasets[target_cfg.name] = ConcatDataset(datasets)
            target_dataloaders[target_cfg.name] = DataLoader(
                ConcatDataset(datasets),
                batch_size=target_dataset_cfg.batch_size,
                shuffle=False,
                pin_memory=True,
                num_workers=target_dataset_cfg.num_workers,
            )

        return target_datasets, target_dataloaders

    def save_features(
        self,
        source_model_config: ModelSourceConfig,
        target: str,
        per_target_features: Dict[str, per_layer_feature_type],
        per_target_labels: Dict[str, per_layer_feature_type],
        per_target_indices: Dict[str, per_layer_feature_type],
    ):
        # Save the sampled features and labels
        assert (
            self.feature_cfg.output_dir_path
        ), "Output directory path must be specified in the feature configuration."
        output_path = (
            Path(self.feature_cfg.output_dir_path)
            / f"{source_model_config.source_name}_to_{target}"
            / f"{source_model_config.model_name}_to_{target}_features.npz"
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            output_path,
            **{
                f"{layer}_features": per_target_features[target][layer]
                for layer in per_target_features[target]
            },
            **{
                f"{layer}_labels": per_target_labels[target][layer]
                for layer in per_target_labels[target]
            },
            **{
                f"{layer}_indices": per_target_indices[target][layer]
                for layer in per_target_indices[target]
            },
        )
        print(f"Saved features to {output_path}")

    def run_transfer_ranking(self):
        for source_model_config in self.source_model_cfgs:
            model = self.initialise_model(
                source_model_config, model_dir_path=self.source_model_base_path
            )
            per_target_features: Dict[str, per_layer_feature_type] = {}
            per_target_labels: Dict[str, per_layer_feature_type] = {}
            per_target_indices: Dict[str, per_layer_feature_type] = {}
            # Iterate over all target datasets
            for target, target_dataset in tqdm(self.target_datasets.items()):
                features, labels, indices = self.extract_features_sampled(
                    model, target, target_dataset
                )
                per_target_features[target] = features
                per_target_labels[target] = labels
                per_target_indices[target] = indices

                if self.feature_cfg.output_dir_path:
                    self.save_features(
                        source_model_config,
                        target,
                        per_target_features,
                        per_target_labels,
                        per_target_indices,
                    )

    def run_transfer_ranking_batched(self):
        """
        Efficient version of run_transfer_ranking with batch processing.

        """
        for source_model_config in self.source_model_cfgs:
            model = self.initialise_model(
                source_model_config, model_dir_path=self.source_model_base_path
            )
            per_target_features: Dict[str, per_layer_feature_type] = {}
            per_target_labels: Dict[str, per_layer_feature_type] = {}
            per_target_indices: Dict[str, per_layer_feature_type] = {}
            for target, target_dataloader in tqdm(self.target_dataloaders.items()):
                features, labels, indices = self.extract_features_sampled_batched(
                    model, target, target_dataloader
                )
                per_target_features[target] = features
                per_target_labels[target] = labels
                per_target_indices[target] = indices

                if self.feature_cfg.output_dir_path:
                    self.save_features(
                        source_model_config,
                        target,
                        per_target_features,
                        per_target_labels,
                        per_target_indices,
                    )

    def extract_features_sampled(
        self,
        model: torch.nn.Module,
        target: str,
        target_dataset: ConcatDataset[Any],
    ):
        per_image_features: Dict[str, List[NDArray[Any]]] = {}
        per_image_labels: Dict[str, List[NDArray[Any]]] = {}
        per_image_indices: Dict[str, List[NDArray[Any]]] = {}
        feature_extractor = FeatureExtractor(model, layers=self.feature_cfg.layers)
        precomputed_indices = self.feature_indices[target]
        with torch.no_grad():
            for image, label in tqdm(iter(target_dataset)):
                image = image.to("cuda:0" if torch.cuda.is_available() else "cpu")
                features = feature_extractor(image)
                label = label.numpy()

                sampled_PL_features: per_layer_feature_type = {}
                sampled_PL_targets: per_layer_feature_type = {}
                sampled_PL_indices: per_layer_feature_type = {}
                for layer, feature in features.items():
                    feature = feature.detach().cpu().numpy()
                    if precomputed_indices is None:
                        class_frequencies = self.class_counts[target]
                        assert (
                            class_frequencies is not None
                        ), "Class counts must be computed before sampling."
                        sampled_outputs, sampled_labels, sampled_indices = (
                            sample_from_image(
                                feature,
                                label,
                                self.feature_cfg.sampling_seed,
                                class_frequencies,
                                self.feature_cfg.num_samples,
                            )
                        )
                    else:
                        sampled_indices = self.feature_indices[target]
                        assert (
                            sampled_indices is not None
                        ), "Feature indices for target dataset must be defined."
                        if sampled_indices[layer] is None:
                            class_frequencies = self.class_counts[target]
                            assert (
                                class_frequencies is not None
                            ), "Class counts must be computed before sampling."
                            sampled_outputs, sampled_labels, sampled_indices = (
                                sample_from_image(
                                    feature,
                                    label,
                                    self.feature_cfg.sampling_seed,
                                    class_frequencies,
                                    self.feature_cfg.num_samples,
                                )
                            )
                        else:
                            sampled_indices = sampled_indices[layer]
                            assert (
                                sampled_indices is not None
                            ), "Precomputed indices for the layer must be defined."
                            sampled_outputs = np.reshape(
                                feature, [-1, feature.shape[1]]
                            )[sampled_indices]
                            sampled_labels = np.reshape(label, [-1])[sampled_indices]
                    sampled_PL_features[layer] = sampled_outputs
                    sampled_PL_targets[layer] = sampled_labels
                    sampled_PL_indices[layer] = sampled_indices
                # Append sampled features and labels per layer
                for layer in sampled_PL_features:
                    if layer not in per_image_features:
                        per_image_features[layer] = []
                        per_image_labels[layer] = []
                        per_image_indices[layer] = []
                    per_image_features[layer].append(sampled_PL_features[layer])
                    per_image_labels[layer].append(sampled_PL_targets[layer])
                    per_image_indices[layer].append(sampled_PL_indices[layer])

            feature_extractor.remove_handler()

        # Concatenate features and labels across all images
        concatenated_features: Dict[str, NDArray[Any]] = {}
        concatenated_labels: Dict[str, NDArray[Any]] = {}
        concatenated_indices: Dict[str, NDArray[Any]] = {}
        for layer in per_image_features:
            concatenated_features[layer] = np.stack(per_image_features[layer])
            concatenated_labels[layer] = np.vstack(per_image_labels[layer])
            concatenated_indices[layer] = np.vstack(per_image_indices[layer])

        return concatenated_features, concatenated_labels, concatenated_indices

    def extract_features_sampled_batched(
        self,
        model: torch.nn.Module,
        target: str,
        target_dataloader: DataLoader[Any],
    ):

        # Precompute sampling strategy if needed
        precomputed_indices = self.feature_indices[target]
        class_frequencies = self.class_counts[target]

        # Initialize storage with pre-allocated lists
        all_features: Dict[str, List[NDArray[Any]]] = {}
        all_labels: Dict[str, List[NDArray[Any]]] = {}
        all_indices: Dict[str, List[NDArray[Any]]] = {}

        feature_extractor = FeatureExtractor(model, layers=self.feature_cfg.layers)
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        batch_size = target_dataloader.batch_size
        assert batch_size is not None, "Batch size must be defined."

        with torch.no_grad():
            for i, (batch_images, batch_labels) in enumerate(
                tqdm(target_dataloader, desc="Processing batches")
            ):
                batch_images = batch_images.to(device)
                # 2D model requires 4D input (B, C, H, W)
                batch_images = torch.squeeze(
                    batch_images, dim=-3
                )  # Ensure correct shape remove z spatial dimension

                # Single forward pass for the entire batch
                batch_features = feature_extractor(batch_images)

                # Process each image in the batch
                for j in range(batch_images.size(0)):
                    label = batch_labels[j].numpy()

                    # Process all layers for this image
                    for layer_name, layer_features in batch_features.items():
                        # Extract features for this specific image
                        image_features = layer_features[j].detach().cpu().numpy()

                        # Sample features for this image and layer
                        if precomputed_indices is not None:
                            if precomputed_indices[layer_name] is not None:
                                # Use precomputed indices
                                layer_indices = precomputed_indices[layer_name]
                                assert (
                                    layer_indices is not None
                                ), "Precomputed indices for the layer must be defined."
                                sampled_features, sampled_labels, sampled_indices = (
                                    self._sample_with_precomputed_indices(
                                        image_features,
                                        label,
                                        layer_indices[i * batch_size + j],
                                    )
                                )
                            else:
                                # Compute sampling on-the-fly
                                sampled_features, sampled_labels, sampled_indices = (
                                    self._sample_features_vectorized(
                                        image_features, label, class_frequencies
                                    )
                                )
                        else:
                            # Compute sampling on-the-fly
                            sampled_features, sampled_labels, sampled_indices = (
                                self._sample_features_vectorized(
                                    image_features, label, class_frequencies
                                )
                            )

                        # Store results
                        if layer_name not in all_features:
                            all_features[layer_name] = []
                            all_labels[layer_name] = []
                            all_indices[layer_name] = []

                        all_features[layer_name].append(sampled_features)
                        all_labels[layer_name].append(sampled_labels)
                        all_indices[layer_name].append(sampled_indices)

        feature_extractor.remove_handler()

        # Efficiently concatenate all results
        concatenated_features: Dict[str, NDArray[Any]] = {}
        concatenated_labels: Dict[str, NDArray[Any]] = {}
        concatenated_indices: Dict[str, NDArray[Any]] = {}

        for layer_name in all_features:
            concatenated_features[layer_name] = np.stack(all_features[layer_name])
            concatenated_labels[layer_name] = np.vstack(all_labels[layer_name])
            concatenated_indices[layer_name] = np.vstack(all_indices[layer_name])

        return concatenated_features, concatenated_labels, concatenated_indices

    def _sample_with_precomputed_indices(
        self, features: NDArray[Any], labels: NDArray[Any], indices: NDArray[Any]
    ) -> tuple[NDArray[Any], NDArray[Any], NDArray[Any]]:
        """Fast sampling using precomputed indices."""
        features_flat = features.reshape(-1, features.shape[0])
        labels_flat = labels.reshape(-1)

        # Ensure indices are within bounds
        valid_indices = indices[indices < len(labels_flat)]

        return (features_flat[valid_indices], labels_flat[valid_indices], valid_indices)

    def _sample_features_vectorized(
        self,
        features: NDArray[Any],
        labels: NDArray[Any],
        class_frequencies: Optional[Dict[int, int]],
    ) -> tuple[NDArray[Any], NDArray[Any], NDArray[Any]]:
        """Vectorized feature sampling with reduced overhead."""
        features_flat = features.reshape(-1, features.shape[0])
        labels_flat = labels.reshape(-1)

        num_samples = min(self.feature_cfg.num_samples, len(labels_flat))

        if class_frequencies is None or not class_frequencies:
            # Uniform sampling - much faster
            rng = np.random.default_rng(self.feature_cfg.sampling_seed)
            indices = rng.choice(len(labels_flat), size=num_samples, replace=False)
        else:
            # Class-balanced sampling - optimized version
            indices = self._fast_class_balanced_sampling(
                labels_flat, class_frequencies, num_samples
            )

        return (features_flat[indices], labels_flat[indices], indices)

    def _fast_class_balanced_sampling(
        self, labels: NDArray[Any], class_frequencies: Dict[int, int], num_samples: int
    ) -> NDArray[Any]:
        """Optimized class-balanced sampling."""
        # Pre-compute weights for all labels at once
        label_weights = np.array(
            [1.0 / class_frequencies.get(int(label), 1) for label in labels]
        )
        probabilities = label_weights / label_weights.sum()

        rng = np.random.default_rng(self.feature_cfg.sampling_seed)
        return rng.choice(len(labels), size=num_samples, replace=False, p=probabilities)


def get_precomputed_feature_path(
    model_name: str, target: str, base_path: Union[str, Path]
):
    if isinstance(base_path, str):
        base_path = Path(base_path)
    source = MODEL_ABBREVIATIONS_TO_DATASET[model_name.split("_")[0]]
    paths = list(
        base_path.rglob(
            f"{source}_to_{target}/**/{model_name}_to_{target}_features.npz"
        )
    )
    assert (
        len(paths) == 1
    ), f"Expected exactly one path for {model_name} to {target}, found {len(paths)}"
    return paths[0]
