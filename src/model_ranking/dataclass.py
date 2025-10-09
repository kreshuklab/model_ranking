from pydantic import BaseModel, Discriminator
from pathlib import Path
from typing import (
    Annotated,
    Literal,
    Mapping,
    Sequence,
    Union,
    Optional,
    List,
    Tuple,
    Dict,
    Any,
    assert_never,
)
import numpy as np
from numpy.typing import NDArray
import torch

from model_ranking.metrics import (
    MultiClassF1Eval,
    BinaryF1Eval,
    SoftF1Eval,
    AdaptedRandErrorEval,
    DifferenceImageEval,
    EffectiveInvarianceEval,
    EntropyEval,
    KLDivergenceEval,
    CrossEntropyEval,
    HammingDistanceEval,
)
from pytorch3dunet.unet3d.metrics import (
    InstanceAveragePrecision,
)

DEFAULT_SCHEDULER_KWARGS: Dict[str, Any] = {
    "mode": "max",
    "factor": 0.5,
    "patience": 10,
}

transformer_type = Mapping[
    str,
    List[Mapping[str, Optional[Union[str, bool, int, Sequence[int], Sequence[float]]]]],
]

transferability_metric_names = Literal[
    "GBC", "LEEP", "Gaussian_LEEP", "Hscore", "Regularized_Hscore", "LogME", "NCTI"
]


class ConsistencyMetricConfig(BaseModel):
    metric: Literal[
        "Diff",
        "EI",
        "Entropy",
        "Cross-Entropy",
        "KL-Divergence",
        "Hamming-Distance",
        "Rand-Index",
        "Adapted-Rand-Error",
        "AdaRand-Error",
    ]
    threshold: List[float]
    pred_key: str
    save_key: str
    entr_base: Optional[int]
    diff_alpha: Optional[float]
    select_alphas: Optional[Dict[str, List[str]]]
    save_mask: bool
    ignore_path: Optional[str]
    ignore_key: Optional[str]
    border_parameters: Optional[Dict[str, int]]
    remove_background: bool
    zero_largest_instance: bool


class ConsistencyMetaConfig(BaseModel, frozen=True):
    save_key: Optional[str]
    save_mask: Optional[bool]
    mask_threshold: float
    overwrite_score: bool
    bckg_consistency: bool = False
    # invert_threshold: bool = False
    # greater_than_threshold: bool = True


class EvalDatasetConfig(BaseModel):
    name: Literal["StandardEvalDataset"]
    aug_name: str
    pred_path: Sequence[str]
    gt_path: Sequence[str]
    pred_key: str
    gt_key: str
    patch_key: str
    roi: Optional[Sequence[Sequence[int]]]
    ignore_index: Optional[int]
    ignore_path: Optional[str]
    ignore_key: Optional[str]
    convert_to_binary_label: bool
    convert_to_boundary_label: bool
    gt_zero_largest_instance: bool
    gt_zero_large_instances: bool
    min_object_size: Optional[int]
    zero_large_instances: bool
    zero_largest_instance: bool
    largest_obj_multiplier: Optional[float]
    max_obj_size: Optional[int]


class Pytorch3DUnetSliceBuilderConfig(BaseModel):
    name: Literal["SliceBuilder"]
    patch_shape: Tuple[int, int, int]
    stride_shape: Tuple[int, int, int]
    halo_shape: Tuple[int, int, int]


class Pytorch3DUnetFilterSliceBuilderConfig(BaseModel):
    name: Literal["FilterSliceBuilder"]
    patch_shape: Tuple[int, int, int]
    stride_shape: Tuple[int, int, int]
    halo_shape: Tuple[int, int, int]
    threshold: float
    ignore_index: Optional[int]
    slack_acceptance: float


class TIFPhaseConfig(BaseModel):
    # dataset_name: Literal["Standard_TIF_Dataset", "HeLaNuc_Dataset", "Hoechst_Dataset"]
    image_dir: Sequence[str]
    mask_dir: Sequence[str]
    transformer: transformer_type


class TIFtxtPhaseConfig(BaseModel, frozen=True):
    # dataset_name: Literal["TIF_txt_Dataset"]
    image_dir: Sequence[str]
    mask_dir: Sequence[str]
    filenames_path: str
    transformer: transformer_type


class TIFEvalDatasetConfig(BaseModel):
    name: Literal[
        "TIF_txt_Dataset", "Standard_TIF_Dataset", "HeLaNuc_Dataset", "Hoechst_Dataset"
    ]
    eval: Union[TIFPhaseConfig, TIFtxtPhaseConfig]
    expand_dims: bool
    global_norm: bool
    percentiles: Optional[Sequence[Union[float, int]]]
    image_key: Optional[str]
    mask_key: Optional[str]
    min_object_size: Optional[int]
    zero_large_instances: bool


class SBIAD1410PhaseConfig(BaseModel):
    img_paths: Sequence[str]
    mask_paths: Sequence[str]
    roi: Optional[Sequence[Sequence[int]]]
    transformer: transformer_type
    slice_builder: Optional[
        Union[Pytorch3DUnetFilterSliceBuilderConfig, Pytorch3DUnetSliceBuilderConfig]
    ]


class SBIAD1410PhaseMetaConfig(BaseModel):
    mask_paths: Optional[Sequence[str]]
    roi: Optional[Sequence[Sequence[int]]]
    transformer: transformer_type
    slice_builder: Optional[
        Annotated[
            Union[
                Pytorch3DUnetFilterSliceBuilderConfig, Pytorch3DUnetSliceBuilderConfig
            ],
            Discriminator("name"),
        ]
    ]


class SBIAD1410EvalDatasetConfig(BaseModel):
    name: Literal["S_BIAD1410_Dataset"]
    eval: SBIAD1410PhaseConfig
    global_normalization: bool
    global_percentiles: Optional[Sequence[Union[float, int]]]
    image_key: Optional[str]
    mask_key: Optional[str]
    zero_large_instances: bool = True


class EvalDataloaderConfig(BaseModel):
    eval_dataset: Annotated[
        Union[EvalDatasetConfig, TIFEvalDatasetConfig, SBIAD1410EvalDatasetConfig],
        Discriminator("name"),
    ]
    batch_size: int
    num_workers: int


class EvalDataloaderMetaConfig(BaseModel):
    name: Literal["StandardEvalDataset"]
    gt_path: Optional[Sequence[str]]
    pred_key: str
    gt_key: str
    patch_key: str
    roi: Optional[Sequence[Sequence[int]]]
    ignore_index: Optional[int]
    ignore_path: Optional[str]
    ignore_key: Optional[str]
    convert_to_boundary_label: bool
    convert_to_binary_label: bool
    min_object_size: Optional[int]
    gt_zero_largest_instance: bool
    gt_zero_large_instances: bool
    zero_large_instances: bool
    zero_largest_instance: bool
    largest_obj_multiplier: Optional[float]
    max_obj_size: Optional[int]
    batch_size: int
    num_workers: int

    def create_config(
        self, aug_name: str, pred_path: Sequence[str], data_base_path: str
    ):
        gt_path: List[str] = []
        assert self.gt_path is not None, "gt_path is None"
        for i in range(len(self.gt_path)):
            gt_path.append(data_base_path + self.gt_path[i])
        if self.ignore_path is not None:
            ignore_path = data_base_path + self.ignore_path
        else:
            ignore_path = None
        return EvalDataloaderConfig(
            eval_dataset=EvalDatasetConfig(
                name=self.name,
                aug_name=aug_name,
                pred_path=pred_path,
                gt_path=gt_path,
                pred_key=self.pred_key,
                gt_key=self.gt_key,
                patch_key=self.patch_key,
                roi=self.roi,
                ignore_index=self.ignore_index,
                ignore_path=ignore_path,
                ignore_key=self.ignore_key,
                convert_to_boundary_label=self.convert_to_boundary_label,
                convert_to_binary_label=self.convert_to_binary_label,
                min_object_size=self.min_object_size,
                gt_zero_largest_instance=self.gt_zero_largest_instance,
                gt_zero_large_instances=self.gt_zero_large_instances,
                zero_large_instances=self.zero_large_instances,
                zero_largest_instance=self.zero_largest_instance,
                largest_obj_multiplier=self.largest_obj_multiplier,
                max_obj_size=self.max_obj_size,
            ),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def create_consis_config(
        self,
        aug_name: str,
        perturbed_path: Sequence[str],
        unperturbed_path: Sequence[str],
        data_base_path: str,
    ):
        if self.ignore_path is not None:
            ignore_path = data_base_path + self.ignore_path
        else:
            ignore_path = None
        return EvalDataloaderConfig(
            eval_dataset=EvalDatasetConfig(
                name=self.name,
                aug_name=aug_name,
                pred_path=perturbed_path,
                gt_path=unperturbed_path,
                pred_key=self.pred_key,
                gt_key=self.gt_key,
                patch_key=self.patch_key,
                roi=self.roi,
                ignore_index=self.ignore_index,
                ignore_path=ignore_path,
                ignore_key=self.ignore_key,
                convert_to_boundary_label=self.convert_to_boundary_label,
                convert_to_binary_label=self.convert_to_binary_label,
                min_object_size=self.min_object_size,
                gt_zero_largest_instance=self.gt_zero_largest_instance,
                gt_zero_large_instances=self.gt_zero_large_instances,
                zero_large_instances=self.zero_large_instances,
                zero_largest_instance=self.zero_largest_instance,
                largest_obj_multiplier=self.largest_obj_multiplier,
                max_obj_size=self.max_obj_size,
            ),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )


class EvalSB1410DataloaderMetaConfig(BaseModel):
    name: Literal["S_BIAD1410_Dataset"]
    eval: SBIAD1410PhaseMetaConfig
    global_normalization: bool
    global_percentiles: Optional[Sequence[Union[float, int]]]
    image_key: Optional[str]
    mask_key: Optional[str]
    zero_large_instances: bool
    batch_size: int
    num_workers: int

    def create_config(self, img_paths: Sequence[str], data_base_path: str):
        mask_paths: List[str] = []
        assert self.eval.mask_paths is not None, "mask_paths is None"
        for i in range(len(self.eval.mask_paths)):
            mask_paths.append(data_base_path + self.eval.mask_paths[i])
        return EvalDataloaderConfig(
            eval_dataset=SBIAD1410EvalDatasetConfig(
                name=self.name,
                eval=SBIAD1410PhaseConfig(
                    img_paths=img_paths,
                    mask_paths=mask_paths,
                    roi=self.eval.roi,
                    transformer=self.eval.transformer,
                    slice_builder=self.eval.slice_builder,
                ),
                global_normalization=self.global_normalization,
                global_percentiles=self.global_percentiles,
                image_key=self.image_key,
                mask_key=self.mask_key,
                zero_large_instances=self.zero_large_instances,
            ),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def create_consis_config(
        self,
        perturbed_paths: Sequence[str],
        unperturbed_paths: Sequence[str],
    ):
        return EvalDataloaderConfig(
            eval_dataset=SBIAD1410EvalDatasetConfig(
                name=self.name,
                eval=SBIAD1410PhaseConfig(
                    img_paths=perturbed_paths,
                    mask_paths=unperturbed_paths,
                    roi=self.eval.roi,
                    transformer=self.eval.transformer,
                    slice_builder=self.eval.slice_builder,
                ),
                global_normalization=self.global_normalization,
                global_percentiles=self.global_percentiles,
                image_key=self.image_key,
                mask_key=self.mask_key,
                zero_large_instances=self.zero_large_instances,
            ),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )


class EvalMetricConfig(BaseModel, frozen=True):
    eval_save_key: str
    overwrite_score: bool


class AdaptedRandErrorConfig(BaseModel, frozen=True):
    name: Literal["AdaptedRandError"] = "AdaptedRandError"
    num_dilations: Optional[int] = 1
    num_erosions: Optional[int] = 1

    def initialise_metric(self, incomplete_gt: bool) -> AdaptedRandErrorEval:
        return AdaptedRandErrorEval(
            incomplete_gt=incomplete_gt,
            num_dilations=self.num_dilations,
            num_erosions=self.num_erosions,
        )

    def initialise_score(self, num_samples: int) -> torch.Tensor:
        return torch.zeros((num_samples, 3), dtype=torch.float32)

    def initialise_score_array(self, num_samples: int) -> NDArray[Any]:
        return np.zeros((num_samples, 3), dtype=np.float32)


class AdaptedRandErrorEvalConfig(AdaptedRandErrorConfig, EvalMetricConfig, frozen=True):
    pass


class MeanAvgPrecisionConfig(EvalMetricConfig, frozen=True):
    name: Literal["MeanAvgPrecision"] = "MeanAvgPrecision"
    iou_range: Optional[List[Union[float, int]]] = [0.5, 0.95, 10]
    min_instance_size: Optional[int] = None

    def initialise_metric(self) -> InstanceAveragePrecision:
        return InstanceAveragePrecision(
            min_instance_size=self.min_instance_size,
            iou_range=self.iou_range,
        )

    def initialise_score(self, num_samples: int) -> torch.Tensor:
        return torch.zeros(num_samples, dtype=torch.float32)


class MultiClassF1Config(EvalMetricConfig, frozen=True):
    name: Literal["MultiClassF1"] = "MultiClassF1"
    threshold: float = 0.5

    def initialise_metric(self) -> MultiClassF1Eval:
        return MultiClassF1Eval(
            threshold=self.threshold,
        )

    def initialise_score(self, num_samples: int) -> torch.Tensor:
        return torch.zeros((num_samples, 2), dtype=torch.float32)


class BinaryF1Config(EvalMetricConfig, frozen=True):
    name: Literal["BinaryF1"] = "BinaryF1"
    threshold: float = 0.5

    def initialise_metric(self) -> BinaryF1Eval:
        return BinaryF1Eval(
            threshold=self.threshold,
        )

    def initialise_score(self, num_samples: int) -> torch.Tensor:
        return torch.zeros(num_samples, dtype=torch.float32)


class SoftF1Config(EvalMetricConfig, frozen=True):
    name: Literal["SoftF1"] = "SoftF1"

    def initialise_metric(self) -> SoftF1Eval:
        return SoftF1Eval()

    def initialise_score(self, num_samples: int) -> torch.Tensor:
        return torch.zeros(num_samples, dtype=torch.float32)


class AdaptedRandErrorConsisConfig(
    AdaptedRandErrorConfig, ConsistencyMetaConfig, frozen=True
):
    pass


class DifferenceImageConfig(ConsistencyMetaConfig, frozen=True):
    name: Literal["Diff"] = "Diff"
    diff_alpha: float = 1

    def initialise_metric(self) -> DifferenceImageEval:
        return DifferenceImageEval(
            diff_alpha=self.diff_alpha,
        )

    def initialise_score(
        self, num_samples: int, sample_shape: Sequence[int]
    ) -> NDArray[Any]:
        return np.zeros((num_samples, *sample_shape), dtype=np.float32)


class EffectiveInvarianceConfig(ConsistencyMetaConfig, frozen=True):
    name: Literal["EI"] = "EI"
    threshold: float = 0.5

    def initialise_metric(self) -> EffectiveInvarianceEval:
        return EffectiveInvarianceEval(
            threshold=self.threshold,
            invert_threshold=self.bckg_consistency,
        )

    def initialise_score(
        self, num_samples: int, sample_shape: Sequence[int]
    ) -> NDArray[Any]:
        return np.zeros((num_samples, *sample_shape), dtype=np.float32)


class EntropyConfig(ConsistencyMetaConfig, frozen=True):
    name: Literal["Entropy"] = "Entropy"
    entr_base: int = 2

    def initialise_metric(self) -> EntropyEval:
        return EntropyEval(
            entr_base=self.entr_base,
        )

    def initialise_score(
        self, num_samples: int, sample_shape: Sequence[int]
    ) -> NDArray[Any]:
        return np.zeros((num_samples, *sample_shape), dtype=np.float32)


class KLDivergenceConfig(ConsistencyMetaConfig, frozen=True):
    name: Literal["KL-Divergence"] = "KL-Divergence"
    eps: float = 1e-7
    entr_base: int = 2

    def initialise_metric(self) -> KLDivergenceEval:
        return KLDivergenceEval(
            eps=self.eps,
            entr_base=self.entr_base,
        )

    def initialise_score(
        self, num_samples: int, sample_shape: Sequence[int]
    ) -> NDArray[Any]:
        return np.zeros((num_samples, *sample_shape), dtype=np.float32)


class CrossEntropyConfig(ConsistencyMetaConfig, frozen=True):
    name: Literal["Cross-Entropy"] = "Cross-Entropy"
    eps: float = 1e-7
    entr_base: int = 2

    def initialise_metric(self) -> CrossEntropyEval:
        return CrossEntropyEval(
            eps=self.eps,
            entr_base=self.entr_base,
        )

    def initialise_score(
        self, num_samples: int, sample_shape: Sequence[int]
    ) -> NDArray[Any]:
        return np.zeros((num_samples, *sample_shape), dtype=np.float32)


class HammingDistanceConfig(ConsistencyMetaConfig, frozen=True):
    name: Literal["Hamming-Distance"] = "Hamming-Distance"
    threshold: float = 0.5

    def initialise_metric(self) -> HammingDistanceEval:
        return HammingDistanceEval(
            threshold=self.threshold,
            invert_threshold=self.bckg_consistency,
        )

    def initialise_score(self, num_samples: int) -> NDArray[Any]:
        return np.zeros(num_samples, dtype=np.float32)


consistency_metric_type = Annotated[
    Union[
        AdaptedRandErrorConsisConfig,
        CrossEntropyConfig,
        DifferenceImageConfig,
        EffectiveInvarianceConfig,
        EntropyConfig,
        HammingDistanceConfig,
        KLDivergenceConfig,
    ],
    Discriminator("name"),
]

eval_metric_type = Annotated[
    Union[
        AdaptedRandErrorEvalConfig,
        BinaryF1Config,
        MeanAvgPrecisionConfig,
        MultiClassF1Config,
        SoftF1Config,
    ],
    Discriminator("name"),
]


class EvaluateConfig(BaseModel, frozen=True):
    eval_dataloader: EvalDataloaderConfig
    eval_metric: eval_metric_type


class ConsistencyConfig(BaseModel, frozen=True):
    consistency_dataloader: EvalDataloaderConfig
    consistency_metric: consistency_metric_type


class WandbConfig(BaseModel):
    project: str
    name: str
    mode: Literal["disabled", "online", "offline"]
    run_id: Optional[str] = None
    resume: Union[bool, None, Literal["allow", "never", "must", "auto"]] = None


class FeaturePerturbationBaseConfig(BaseModel):
    layers: Sequence[int]
    random_seed: int


class DropOutPerturbationConfig(FeaturePerturbationBaseConfig):
    name: Literal["DropOutPerturbation"]
    drop_rate: float
    spatial_dropout: bool


class FeatureDropPerturbationConfig(FeaturePerturbationBaseConfig):
    name: Literal["FeatureDropPerturbation"]
    lower_th: float
    upper_th: float


class FeatureNoisePerturbationConfig(FeaturePerturbationBaseConfig):
    name: Literal["FeatureNoisePerturbation"]
    uniform_range: float


class InputPerturbationConfig(BaseModel):
    # name: Literal["RandomGamma", "RandomBrightness", "RandomContrast"]
    name: str
    execution_probability: float
    alpha: Tuple[float, float]
    clip_kwargs: Optional[Dict[str, float]]


class InputGaussianConfig(BaseModel):
    # name: Literal["AdditiveGaussianNoise"]
    name: str
    execution_probability: float
    scale: Tuple[float, float]


class Pytorch3DUnetPredictorMetaConfig(BaseModel):
    name: Literal["PatchWisePredictor"]
    save_segmentation: bool
    min_size: Optional[int]
    layer_id: Optional[int]
    beta: float = 0.5
    zero_largest_instance: bool
    zero_large_instances: bool
    large_instance_multiplier: float = 4
    max_obj_size: Optional[float] = None


class Pytorch3DUnetPredictorConfig(Pytorch3DUnetPredictorMetaConfig):
    save_suffix: str
    output_file_name: Optional[str]


class Pytorch3DUnetDatasetConfig(BaseModel, frozen=True):
    file_paths: Sequence[str]
    slice_builder: Annotated[
        Union[Pytorch3DUnetSliceBuilderConfig, Pytorch3DUnetFilterSliceBuilderConfig],
        Discriminator("name"),
    ]
    transformer: transformer_type
    roi: Optional[Union[Sequence[Sequence[int]], Sequence[int]]]


class Pytorch3DUnetLoaderConfig(BaseModel, frozen=True):
    dataset: Literal["StandardHDF5Dataset"]
    batch_size: int
    num_workers: int
    raw_internal_path: str
    label_internal_path: str
    global_normalization: bool
    global_percentiles: Optional[Sequence[float]]


class Pytorch3DUnetTestLoaderConfig(Pytorch3DUnetLoaderConfig, frozen=True):
    output_dir: str
    test: Pytorch3DUnetDatasetConfig


class Pytorch3DUnetTrainLoaderConfig(Pytorch3DUnetLoaderConfig, frozen=True):
    train: Pytorch3DUnetDatasetConfig


class SBIAD1410LoaderConfig(BaseModel, frozen=True):
    dataset: Literal["S_BIAD1410_Dataset"]
    batch_size: int
    num_workers: int
    global_normalization: bool
    global_percentiles: Optional[Sequence[Union[float, int]]]


class SBIAD1410LoaderTestConfig(SBIAD1410LoaderConfig, frozen=True):
    output_dir: str
    test: SBIAD1410PhaseConfig


class SBIAD1410LoaderTrainConfig(SBIAD1410LoaderConfig, frozen=True):
    train: SBIAD1410PhaseConfig


class SBIAD1410LoaderMetaConfig(BaseModel, frozen=True):
    dataset: Literal["S_BIAD1410_Dataset"]
    batch_size: int
    num_workers: int
    global_normalization: bool
    global_percentiles: Optional[Sequence[Union[float, int]]]
    img_paths: Sequence[str]
    mask_paths: Sequence[str]
    roi: Optional[Sequence[Sequence[int]]]
    transformer: transformer_type
    slice_builder: Optional[
        Union[Pytorch3DUnetFilterSliceBuilderConfig, Pytorch3DUnetSliceBuilderConfig]
    ]

    def create_config(
        self,
        output_dir: Optional[str],
        data_base_path: str,
        phase: Literal["train", "test"] = "test",
    ):
        img_paths: List[str] = []
        mask_paths: List[str] = []
        for i in range(len(self.img_paths)):
            img_paths.append(data_base_path + self.img_paths[i])
            mask_paths.append(data_base_path + self.mask_paths[i])
        if phase == "test":
            assert output_dir is not None, "output_dir must be given for test phase"
            loader = SBIAD1410LoaderTestConfig(
                dataset=self.dataset,
                output_dir=output_dir,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                global_normalization=self.global_normalization,
                global_percentiles=self.global_percentiles,
                test=SBIAD1410PhaseConfig(
                    img_paths=img_paths,
                    mask_paths=mask_paths,
                    roi=self.roi,
                    transformer=self.transformer,
                    slice_builder=self.slice_builder,
                ),
            )
        elif phase == "train":
            for i in range(len(img_paths)):
                img_paths[i] = img_paths[i].replace("test", "train")
                mask_paths[i] = mask_paths[i].replace("test", "train")
            loader = SBIAD1410LoaderTrainConfig(
                dataset=self.dataset,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                global_normalization=self.global_normalization,
                global_percentiles=self.global_percentiles,
                train=SBIAD1410PhaseConfig(
                    img_paths=img_paths,
                    mask_paths=mask_paths,
                    roi=self.roi,
                    transformer=self.transformer,
                    slice_builder=self.slice_builder,
                ),
            )
        else:
            assert_never(phase)

        return loader


class Pytorch3DUnetLoaderMetaConfig(BaseModel, frozen=True):
    dataset: Literal["StandardHDF5Dataset"]
    batch_size: int
    num_workers: int
    raw_internal_path: str
    label_internal_path: str
    global_normalization: bool
    global_percentiles: Optional[Sequence[float]]
    file_paths: Sequence[str]
    slice_builder: Annotated[
        Union[Pytorch3DUnetSliceBuilderConfig, Pytorch3DUnetFilterSliceBuilderConfig],
        Discriminator("name"),
    ]
    transformer: transformer_type
    roi: Optional[Union[Sequence[Sequence[int]], Sequence[int]]]

    def create_config(
        self,
        output_dir: Optional[str],
        data_base_path: str,
        phase: Literal["train", "test"] = "test",
    ):
        file_paths: List[str] = []
        for i in range(len(self.file_paths)):
            file_paths.append(data_base_path + self.file_paths[i])

        if phase == "test":
            assert output_dir is not None, "output_dir must be given for test phase"
            loader = Pytorch3DUnetTestLoaderConfig(
                dataset=self.dataset,
                output_dir=output_dir,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                raw_internal_path=self.raw_internal_path,
                label_internal_path=self.label_internal_path,
                global_normalization=self.global_normalization,
                global_percentiles=self.global_percentiles,
                test=Pytorch3DUnetDatasetConfig(
                    file_paths=file_paths,
                    slice_builder=self.slice_builder,
                    transformer=self.transformer,
                    roi=self.roi,
                ),
            )
            return loader
        elif phase == "train":
            # for i in range(len(file_paths)):
            #    file_paths[i] = file_paths[i].replace("test", "train")
            loader = Pytorch3DUnetTrainLoaderConfig(
                dataset=self.dataset,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                raw_internal_path=self.raw_internal_path,
                label_internal_path=self.label_internal_path,
                global_normalization=self.global_normalization,
                global_percentiles=self.global_percentiles,
                train=Pytorch3DUnetDatasetConfig(
                    file_paths=file_paths,
                    slice_builder=self.slice_builder,
                    transformer=self.transformer,
                    roi=self.roi,
                ),
            )
            return loader
        else:
            assert_never(phase)


class TIFNucleiSemanticPredictorConfig(BaseModel):
    name: Literal["DSB2018Predictor"]


class TIFNucleiInstancePredictorConfig(BaseModel):
    name: Literal["NucleiInstancePredictor"]
    save_segmentation: bool
    min_size: int
    zero_largest_instance: bool = False
    no_adjust_background: bool = False


class TIFLoadersConfig(BaseModel, frozen=True):
    dataset: Literal[
        "Standard_TIF_Dataset", "TIF_txt_Dataset", "HeLaNuc_Dataset", "Hoechst_Dataset"
    ]
    batch_size: int
    num_workers: int
    global_norm: bool
    percentiles: Optional[Sequence[Union[float, int]]]


class TIFPredictionLoadersConfig(TIFLoadersConfig, frozen=True):
    output_dir: str
    test: Union[TIFPhaseConfig, TIFtxtPhaseConfig]


class TIFTrainLoadersConfig(TIFLoadersConfig, frozen=True):
    train: Union[TIFPhaseConfig, TIFtxtPhaseConfig]


feature_perturbation_type = Optional[
    Union[
        DropOutPerturbationConfig,
        FeatureDropPerturbationConfig,
        FeatureNoisePerturbationConfig,
    ]
]


class Pytorch3DUnetModelMetaConfig(BaseModel, frozen=True):
    name: Literal[
        "UNet2D", "UNet2d_as3d", "ResidualUNet2D", "ResidualUNet2D_as_3D", "UNet3D"
    ]
    in_channels: int
    out_channels: int
    layer_order: str
    f_maps: Union[int, Sequence[int]]
    final_sigmoid: bool
    feature_return: bool
    is_segmentation: Optional[bool]


class UnetrModelMetaConfig(BaseModel, frozen=True):
    name: Literal["UnetrWrapper"]
    in_channels: int
    out_channels: int
    img_size: Union[Sequence[int], int]
    feature_size: int
    hidden_size: int
    mlp_dim: int
    num_heads: int
    proj_type: str
    norm_name: Union[Tuple[str, ...], str]
    conv_block: bool
    res_block: bool
    dropout_rate: float
    spatial_dims: int
    qkv_bias: bool
    save_attn: bool
    is_segmentation: bool
    final_sigmoid: bool


class UnetrModelConfig(UnetrModelMetaConfig, frozen=True):
    feature_perturbation: Annotated[feature_perturbation_type, Discriminator("name")]


class Pytorch3DUnetModelConfig(Pytorch3DUnetModelMetaConfig, frozen=True):
    feature_perturbation: Annotated[feature_perturbation_type, Discriminator("name")]


class FeaturePerturbationConfig(BaseModel):
    perturbation_types: Sequence[
        Literal[
            "DropOutPerturbation",
            "FeatureDropPerturbation",
            "FeatureNoisePerturbation",
            "None",
        ]
    ]
    layers: Sequence[int]
    dropOut_rates: Optional[Sequence[float]]
    spatial_dropout: Optional[bool]
    featureDrop_thresholds: Optional[Sequence[Tuple[float, float]]]
    featureNoise_ranges: Optional[Sequence[float]]
    random_seed: int


class OutputSettingsConfig(BaseModel):
    result_dir: Optional[str]
    approach: Optional[str]
    base_dir_path: str
    output_folder: Optional[str] = "norm"


class LoaderMetaConfig(BaseModel):
    batch_size: int
    num_workers: int
    global_norm: bool
    percentiles: Optional[Sequence[float]]
    image_dir: Sequence[str]
    mask_dir: Sequence[str]
    transformer: transformer_type


class TIFLoaderMetaConfig(LoaderMetaConfig):
    dataset: Literal["Standard_TIF_Dataset", "HeLaNuc_Dataset", "Hoechst_Dataset"]

    def create_config(
        self,
        output_dir: Optional[str],
        data_base_path: str,
        phase: Literal["train", "test"] = "test",
    ):
        mask_dir: List[str] = []
        image_dir: List[str] = []
        for i in range(len(self.mask_dir)):
            mask_dir.append(data_base_path + self.mask_dir[i])
            image_dir.append(data_base_path + self.image_dir[i])
        if phase == "test":
            assert output_dir is not None, "output_dir must be given for test phase"
            loader = TIFPredictionLoadersConfig(
                dataset=self.dataset,
                output_dir=output_dir,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                global_norm=self.global_norm,
                percentiles=self.percentiles,
                test=TIFPhaseConfig(
                    image_dir=image_dir,
                    mask_dir=mask_dir,
                    transformer=self.transformer,
                ),
            )
        elif phase == "train":
            for i in range(len(image_dir)):
                image_dir[i] = image_dir[i].replace("test", "train")
                mask_dir[i] = mask_dir[i].replace("test", "train")
            loader = TIFTrainLoadersConfig(
                dataset=self.dataset,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                global_norm=self.global_norm,
                percentiles=self.percentiles,
                train=TIFPhaseConfig(
                    image_dir=image_dir,
                    mask_dir=mask_dir,
                    transformer=self.transformer,
                ),
            )
        else:
            assert_never(phase)
        return loader


class TIFtxtLoaderMetaConfig(LoaderMetaConfig):
    dataset: Literal["TIF_txt_Dataset"]
    filenames_path: str

    def create_config(
        self,
        output_dir: Optional[str],
        data_base_path: str,
        phase: Literal["test", "train"] = "test",
    ):
        mask_dir: List[str] = []
        image_dir: List[str] = []
        for i in range(len(self.mask_dir)):
            mask_dir.append(data_base_path + self.mask_dir[i])
            image_dir.append(data_base_path + self.image_dir[i])
        if phase == "test":
            assert output_dir is not None, "output_dir must be given for test phase"
            loader = TIFPredictionLoadersConfig(
                dataset=self.dataset,
                output_dir=output_dir,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                global_norm=self.global_norm,
                percentiles=self.percentiles,
                test=TIFtxtPhaseConfig(
                    image_dir=image_dir,
                    mask_dir=mask_dir,
                    filenames_path=data_base_path + self.filenames_path,
                    transformer=self.transformer,
                ),
            )
        elif phase == "train":
            loader = TIFTrainLoadersConfig(
                dataset=self.dataset,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                global_norm=self.global_norm,
                percentiles=self.percentiles,
                train=TIFtxtPhaseConfig(
                    image_dir=image_dir,
                    mask_dir=mask_dir,
                    filenames_path=data_base_path + self.filenames_path,
                    transformer=self.transformer,
                ),
            )
        else:
            assert_never(phase)
        return loader


class Eval_TIF_TxtDataloaderMetaConfig(BaseModel, frozen=True):
    name: Literal["TIF_txt_Dataset"]
    expand_dims: bool
    global_norm: bool
    percentiles: Optional[Sequence[Union[float, int]]]
    image_key: Optional[str]
    min_object_size: Optional[int]
    zero_large_instances: bool
    mask_dir: Optional[Sequence[str]]
    mask_key: Optional[str]
    filenames_path: str
    batch_size: int
    num_workers: int
    transformer: transformer_type

    def create_config(self, image_dir: Sequence[str], data_base_path: str):
        mask_dir: List[str] = []
        assert self.mask_dir is not None, "mask_dir must be given"
        for i in range(len(self.mask_dir)):
            mask_dir.append(data_base_path + self.mask_dir[i])
        return EvalDataloaderConfig(
            eval_dataset=TIFEvalDatasetConfig(
                name=self.name,
                eval=TIFtxtPhaseConfig(
                    image_dir=image_dir,
                    mask_dir=mask_dir,
                    filenames_path=data_base_path + self.filenames_path,
                    transformer=self.transformer,
                ),
                expand_dims=self.expand_dims,
                global_norm=self.global_norm,
                percentiles=self.percentiles,
                image_key=self.image_key,
                mask_key=self.mask_key,
                min_object_size=self.min_object_size,
                zero_large_instances=self.zero_large_instances,
            ),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def create_consis_config(
        self,
        perturbed_dir: Sequence[str],
        unperturbed_dir: Sequence[str],
        data_base_path: str,
    ):
        return EvalDataloaderConfig(
            eval_dataset=TIFEvalDatasetConfig(
                name=self.name,
                eval=TIFtxtPhaseConfig(
                    image_dir=perturbed_dir,
                    mask_dir=unperturbed_dir,
                    filenames_path=data_base_path + self.filenames_path,
                    transformer=self.transformer,
                ),
                expand_dims=self.expand_dims,
                global_norm=self.global_norm,
                percentiles=self.percentiles,
                image_key=self.image_key,
                mask_key=self.mask_key,
                min_object_size=self.min_object_size,
                zero_large_instances=self.zero_large_instances,
            ),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )


class Eval_TIF_DataloaderMetaConfig(BaseModel, frozen=True):
    name: Literal["Standard_TIF_Dataset", "HeLaNuc_Dataset", "Hoechst_Dataset"]
    expand_dims: bool
    global_norm: bool
    percentiles: Optional[Sequence[float]]
    image_key: Optional[str]
    min_object_size: Optional[int]
    zero_large_instances: bool
    mask_dir: Optional[Sequence[str]]
    mask_key: Optional[str]
    batch_size: int
    num_workers: int
    transformer: transformer_type

    def create_config(self, image_dir: Sequence[str], data_base_path: str):
        mask_dir: List[str] = []
        assert self.mask_dir is not None, "mask_dir must be given"
        for i in range(len(self.mask_dir)):
            mask_dir.append(data_base_path + self.mask_dir[i])
        return EvalDataloaderConfig(
            eval_dataset=TIFEvalDatasetConfig(
                name=self.name,
                eval=TIFPhaseConfig(
                    image_dir=image_dir,
                    mask_dir=mask_dir,
                    transformer=self.transformer,
                ),
                expand_dims=self.expand_dims,
                global_norm=self.global_norm,
                percentiles=self.percentiles,
                image_key=self.image_key,
                mask_key=self.mask_key,
                min_object_size=self.min_object_size,
                zero_large_instances=self.zero_large_instances,
            ),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def create_consis_config(
        self,
        perturbed_dir: Sequence[str],
        unperturbed_dir: Sequence[str],
    ):
        return EvalDataloaderConfig(
            eval_dataset=TIFEvalDatasetConfig(
                name=self.name,
                eval=TIFPhaseConfig(
                    image_dir=perturbed_dir,
                    mask_dir=unperturbed_dir,
                    transformer=self.transformer,
                ),
                expand_dims=self.expand_dims,
                global_norm=self.global_norm,
                percentiles=self.percentiles,
                image_key=self.image_key,
                mask_key=self.mask_key,
                min_object_size=self.min_object_size,
                zero_large_instances=self.zero_large_instances,
            ),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )


class ConsistencyMetricMetaConfig(BaseModel, frozen=True):
    save_mask: bool
    ignore_path: Optional[str]
    ignore_key: Optional[str]
    remove_background: bool
    zero_largest_instance: bool


class SourceModelConfigBase(BaseModel):
    # model: Pytorch3DUnetModelMetaConfig
    model_name: str
    model_type: Literal[
        "UNet2D",
        "ResidualUNet2D",
        "UnetrWrapper",
        "UNet2d_as3d",
        "ResidualUNet2D_as_3D",
        "Cellpose_SAM",
        "Micro_SAM",
        "SAM",
    ] = "UNet2D"
    checkpoint_name: str = "best_checkpoint"


UNET2D_3LAYER_ARCHITECTURE = Pytorch3DUnetModelMetaConfig(
    name="UNet2D",
    in_channels=1,
    out_channels=1,
    layer_order="bcr",
    f_maps=(32, 64, 128),
    final_sigmoid=True,
    feature_return=False,
    is_segmentation=True,
)

UNET2D_4LAYER_ARCHITECTURE = Pytorch3DUnetModelMetaConfig(
    name="UNet2D",
    in_channels=1,
    out_channels=1,
    layer_order="bcr",
    f_maps=32,
    final_sigmoid=True,
    feature_return=False,
    is_segmentation=True,
)

RESIDUALUNET2D_5LAYER_ARCHITECTURE = Pytorch3DUnetModelMetaConfig(
    name="ResidualUNet2D",
    in_channels=1,
    out_channels=1,
    layer_order="bcr",
    f_maps=64,
    final_sigmoid=True,
    feature_return=False,
    is_segmentation=True,
)


UNETR_DEFAULT_ARCHITECTURE = UnetrModelMetaConfig(
    name="UnetrWrapper",
    in_channels=1,
    out_channels=1,
    img_size=256,  ### Place holder size will be overwritten on creation of UnetrModelConfig
    feature_size=16,
    hidden_size=768,
    mlp_dim=3072,
    num_heads=12,
    proj_type="conv",
    norm_name="batch",
    conv_block=True,
    res_block=True,
    dropout_rate=0.0,
    spatial_dims=2,
    qkv_bias=False,
    save_attn=False,
    is_segmentation=True,
    final_sigmoid=True,
)

dataset_names = Literal[
    "Go-Nuclear",
    "S_BIAD1196",
    "S_BIAD1410",
    "FlyWing",
    "Ovules",
    "PNAS",
    "EPFL",
    "Hmito",
    "Rmito",
    "VNC",
    "BBBC039",
    "DSB2018",
    "HeLaNuc",
    "Hoechst",
    "S_BIAD634",
    "S_BIAD895",
    "Covid_IF",
]


class ModelSourceConfig(SourceModelConfigBase):
    source_name: dataset_names

    def create_config(
        self,
        feature_perturbation: Optional[
            Union[
                DropOutPerturbationConfig,
                FeatureDropPerturbationConfig,
                FeatureNoisePerturbationConfig,
            ]
        ],
    ):
        assert self.model_type in [
            "UNet2D",
            "ResidualUNet2D",
            "UNet2d_as3d",
            "ResidualUNet2D_as_3D",
        ], f"Invalid model type: {self.model_type}"
        if self.model_type == "UNet2D" or self.model_type == "UNet2d_as3d":
            if self.source_name in [
                "Go-Nuclear",
                "S_BIAD1196",
                "S_BIAD1410",
                "FlyWing",
                "Ovules",
                "PNAS",
                "EPFL",
                "Hmito",
                "Rmito",
                "VNC",
            ]:
                model = UNET2D_4LAYER_ARCHITECTURE.model_copy(
                    update={"name": self.model_type}
                )
            else:
                model = UNET2D_3LAYER_ARCHITECTURE.model_copy(
                    update={"name": self.model_type}
                )

        else:
            model = RESIDUALUNET2D_5LAYER_ARCHITECTURE.model_copy(
                update={"name": self.model_type}
            )
        return Pytorch3DUnetModelConfig(
            name=model.name,
            in_channels=model.in_channels,
            out_channels=model.out_channels,
            layer_order=model.layer_order,
            f_maps=model.f_maps,
            final_sigmoid=model.final_sigmoid,
            feature_return=model.feature_return,
            is_segmentation=model.is_segmentation,
            feature_perturbation=feature_perturbation,
        )

    def create_unetr_config(
        self,
        feature_perturbation: feature_perturbation_type,
        img_size: Union[Sequence[int], int],
    ):
        assert (
            self.model_type == "UnetrWrapper"
        ), f"Invalid model type for UnetrConfig: {self.model_type}"
        model = UNETR_DEFAULT_ARCHITECTURE
        return UnetrModelConfig(
            name=model.name,
            in_channels=model.in_channels,
            out_channels=model.out_channels,
            img_size=img_size,
            feature_size=model.feature_size,
            hidden_size=model.hidden_size,
            mlp_dim=model.mlp_dim,
            num_heads=model.num_heads,
            proj_type=model.proj_type,
            norm_name=model.norm_name,
            conv_block=model.conv_block,
            res_block=model.res_block,
            dropout_rate=model.dropout_rate,
            spatial_dims=model.spatial_dims,
            qkv_bias=model.qkv_bias,
            save_attn=model.save_attn,
            is_segmentation=model.is_segmentation,
            final_sigmoid=model.final_sigmoid,
            feature_perturbation=feature_perturbation,
        )


class ForegroundFilterConfig(BaseModel):
    name: Literal["ForegroundFilter"]
    foreground_threshold: float
    gt_dir_path: str
    gt_key: Optional[str]
    roi: Optional[Sequence[Sequence[int]]]
    save_selection: bool
    overwrite: bool

    def create_config(self, data_base_path: str):
        gt_dir_path = data_base_path + self.gt_dir_path
        return ForegroundFilterConfig(
            name=self.name,
            foreground_threshold=self.foreground_threshold,
            gt_dir_path=gt_dir_path,
            gt_key=self.gt_key,
            roi=self.roi,
            save_selection=self.save_selection,
            overwrite=self.overwrite,
        )


class TargetDatasetConfigBase(BaseModel, frozen=True):
    name: Literal[
        "BBBC039",
        "DSB2018",
        "HeLaNuc",
        "Hoechst",
        "S_BIAD634",
        "S_BIAD895",
        "S_BIAD1196",
        "S_BIAD1410",
        "Go-Nuclear",
        "FlyWing",
        "Ovules",
        "PNAS",
        "EPFL",
        "Hmito",
        "Rmito",
        "VNC",
        "Covid_IF",
    ]
    loader: Annotated[
        Union[
            Pytorch3DUnetLoaderMetaConfig,
            TIFLoaderMetaConfig,
            TIFtxtLoaderMetaConfig,
            SBIAD1410LoaderMetaConfig,
        ],
        Discriminator("dataset"),
    ]
    predictor_semantic: Optional[
        Annotated[
            Union[
                Pytorch3DUnetPredictorMetaConfig,
                TIFNucleiSemanticPredictorConfig,
            ],
            Discriminator("name"),
        ]
    ]
    predictor_instance: Optional[
        Annotated[
            Union[
                Pytorch3DUnetPredictorMetaConfig,
                TIFNucleiInstancePredictorConfig,
            ],
            Discriminator("name"),
        ]
    ]
    eval_dataloader_instance: Optional[
        Annotated[
            Union[
                Eval_TIF_TxtDataloaderMetaConfig,
                Eval_TIF_DataloaderMetaConfig,
                EvalDataloaderMetaConfig,
                EvalSB1410DataloaderMetaConfig,
            ],
            Discriminator("name"),
        ]
    ]
    eval_dataloader_semantic: Optional[
        Annotated[
            Union[
                Eval_TIF_TxtDataloaderMetaConfig,
                Eval_TIF_DataloaderMetaConfig,
                EvalDataloaderMetaConfig,
                EvalSB1410DataloaderMetaConfig,
            ],
            Discriminator("name"),
        ]
    ]
    consis_dataloader_instance: Optional[
        Annotated[
            Union[
                Eval_TIF_TxtDataloaderMetaConfig,
                Eval_TIF_DataloaderMetaConfig,
                EvalDataloaderMetaConfig,
                EvalSB1410DataloaderMetaConfig,
            ],
            Discriminator("name"),
        ]
    ]
    consis_dataloader_semantic: Optional[
        Annotated[
            Union[
                Eval_TIF_TxtDataloaderMetaConfig,
                Eval_TIF_DataloaderMetaConfig,
                EvalDataloaderMetaConfig,
                EvalSB1410DataloaderMetaConfig,
            ],
            Discriminator("name"),
        ]
    ]

    filter_results: Optional[ForegroundFilterConfig]


class BBBC039TargetConfig(TargetDatasetConfigBase, frozen=True):
    name: Literal["BBBC039"] = "BBBC039"
    loader: TIFtxtLoaderMetaConfig = TIFtxtLoaderMetaConfig(
        batch_size=2,
        num_workers=8,
        global_norm=True,
        percentiles=(5, 98),
        dataset="TIF_txt_Dataset",
        image_dir=("/BBBC039/images",),
        mask_dir=("/BBBC039/instance_annotations/instance_labels",),
        filenames_path="/BBBC039/test.txt",
        transformer={
            "raw": [
                {"name": "PercentileNormalizer"},
                {"name": "FixedClipping", "min_value": None, "max_value": 2},
                {"name": "CropToFixed", "size": (512, 512), "centered": True},
                {"name": "ToTensor", "expand_dims": True},
            ],
        },
    )
    predictor_instance: TIFNucleiInstancePredictorConfig = (
        TIFNucleiInstancePredictorConfig(
            name="NucleiInstancePredictor",
            save_segmentation=True,
            min_size=50,
            zero_largest_instance=False,
            no_adjust_background=False,
        )
    )
    predictor_semantic: TIFNucleiSemanticPredictorConfig = (
        TIFNucleiSemanticPredictorConfig(
            name="DSB2018Predictor",
        )
    )
    eval_dataloader_semantic: Eval_TIF_TxtDataloaderMetaConfig = (
        Eval_TIF_TxtDataloaderMetaConfig(
            batch_size=2,
            num_workers=8,
            name="TIF_txt_Dataset",
            expand_dims=True,
            global_norm=False,
            percentiles=None,
            image_key="predictions",
            min_object_size=None,
            zero_large_instances=False,
            mask_dir=("/BBBC039/instance_annotations/instance_labels",),
            mask_key=None,
            filenames_path="/BBBC039/test.txt",
            transformer={
                "raw": [
                    {"name": "ToTensor", "expand_dims": True},
                ],
                "label": [
                    {"name": "CropToFixed", "size": [512, 512], "centered": True},
                    {"name": "Relabel"},
                    {"name": "BlobsToMask", "append_label": False},
                    {"name": "ToTensor", "expand_dims": True},
                ],
            },
        )
    )
    eval_dataloader_instance: Eval_TIF_TxtDataloaderMetaConfig = (
        Eval_TIF_TxtDataloaderMetaConfig(
            batch_size=2,
            num_workers=8,
            name="TIF_txt_Dataset",
            expand_dims=True,
            global_norm=False,
            percentiles=None,
            image_key="segmentation",
            min_object_size=50,
            zero_large_instances=False,
            mask_dir=("/BBBC039/instance_annotations/instance_labels",),
            mask_key=None,
            filenames_path="/BBBC039/test.txt",
            transformer={
                "raw": [
                    {"name": "ToTensor", "expand_dims": True},
                ],
                "label": [
                    {"name": "CropToFixed", "size": (512, 512), "centered": True},
                    {"name": "ToTensor", "expand_dims": True},
                ],
            },
        )
    )
    consis_dataloader_semantic: Eval_TIF_TxtDataloaderMetaConfig = (
        Eval_TIF_TxtDataloaderMetaConfig(
            batch_size=2,
            num_workers=8,
            name="TIF_txt_Dataset",
            expand_dims=True,
            global_norm=False,
            percentiles=None,
            image_key="predictions",
            min_object_size=None,
            zero_large_instances=False,
            mask_dir=None,
            mask_key="predictions",
            filenames_path="/BBBC039/test.txt",
            transformer={
                "raw": [
                    {"name": "ToTensor", "expand_dims": True},
                ],
                "label": [
                    {"name": "ToTensor", "expand_dims": True},
                ],
            },
        )
    )
    consis_dataloader_instance: Eval_TIF_TxtDataloaderMetaConfig = (
        Eval_TIF_TxtDataloaderMetaConfig(
            batch_size=2,
            num_workers=8,
            name="TIF_txt_Dataset",
            expand_dims=True,
            global_norm=False,
            percentiles=None,
            image_key="segmentation",
            min_object_size=50,
            zero_large_instances=False,
            mask_dir=None,
            mask_key="segmentation",
            filenames_path="/BBBC039/test.txt",
            transformer={
                "raw": [
                    {"name": "ToTensor", "expand_dims": True},
                ],
                "label": [
                    {"name": "ToTensor", "expand_dims": True},
                ],
            },
        )
    )
    filter_results: None = None


class HeLaNucTargetConfig(TargetDatasetConfigBase, frozen=True):
    name: Literal["HeLaNuc"] = "HeLaNuc"
    loader: TIFLoaderMetaConfig = TIFLoaderMetaConfig(
        dataset="HeLaNuc_Dataset",
        batch_size=2,
        num_workers=8,
        global_norm=True,
        percentiles=(5, 99.6),
        image_dir=("/HeLaCytoNuc/test/images",),
        mask_dir=("/HeLaCytoNuc/test/nuclei_masks",),
        transformer={
            "raw": [
                {"name": "PercentileNormalizer"},
                {"name": "FixedClipping", "min_value": None, "max_value": 2},
                {"name": "ToTensor", "expand_dims": True},
            ]
        },
    )
    predictor_semantic: TIFNucleiSemanticPredictorConfig = (
        TIFNucleiSemanticPredictorConfig(
            name="DSB2018Predictor",
        )
    )
    predictor_instance: TIFNucleiInstancePredictorConfig = (
        TIFNucleiInstancePredictorConfig(
            name="NucleiInstancePredictor",
            save_segmentation=True,
            min_size=50,
            zero_largest_instance=False,
            no_adjust_background=False,
        )
    )
    eval_dataloader_instance: Eval_TIF_DataloaderMetaConfig = (
        Eval_TIF_DataloaderMetaConfig(
            name="HeLaNuc_Dataset",
            expand_dims=True,
            global_norm=False,
            percentiles=None,
            image_key="segmentation",
            min_object_size=50,
            zero_large_instances=False,
            mask_dir=("/HeLaCytoNuc/test/nuclei_masks",),
            mask_key=None,
            batch_size=1,
            num_workers=8,
            transformer={
                "raw": [{"name": "ToTensor", "expand_dims": True}],
                "label": [{"name": "ToTensor", "expand_dims": True}],
            },
        )
    )
    eval_dataloader_semantic: Eval_TIF_DataloaderMetaConfig = (
        Eval_TIF_DataloaderMetaConfig(
            name="HeLaNuc_Dataset",
            expand_dims=True,
            global_norm=False,
            percentiles=None,
            image_key="predictions",
            min_object_size=None,
            zero_large_instances=False,
            mask_dir=("/HeLaCytoNuc/test/nuclei_masks",),
            mask_key=None,
            batch_size=1,
            num_workers=8,
            transformer={
                "raw": [{"name": "ToTensor", "expand_dims": True}],
                "label": [
                    {"name": "Relabel"},
                    {"name": "BlobsToMask", "append_label": False},
                    {"name": "ToTensor", "expand_dims": True},
                ],
            },
        )
    )
    consis_dataloader_instance: Eval_TIF_DataloaderMetaConfig = (
        Eval_TIF_DataloaderMetaConfig(
            name="HeLaNuc_Dataset",
            expand_dims=True,
            global_norm=False,
            percentiles=None,
            image_key="segmentation",
            min_object_size=50,
            zero_large_instances=False,
            mask_dir=None,
            mask_key="segmentation",
            batch_size=1,
            num_workers=8,
            transformer={
                "raw": [
                    {"name": "ToTensor", "expand_dims": True},
                ],
                "label": [
                    {"name": "ToTensor", "expand_dims": True},
                ],
            },
        )
    )
    consis_dataloader_semantic: Eval_TIF_DataloaderMetaConfig = (
        Eval_TIF_DataloaderMetaConfig(
            name="HeLaNuc_Dataset",
            expand_dims=True,
            global_norm=False,
            percentiles=None,
            image_key="predictions",
            min_object_size=None,
            zero_large_instances=False,
            mask_dir=None,
            mask_key="predictions",
            batch_size=1,
            num_workers=8,
            transformer={
                "raw": [
                    {"name": "ToTensor", "expand_dims": True},
                ],
                "label": [
                    {"name": "ToTensor", "expand_dims": True},
                ],
            },
        )
    )
    filter_results: None = None


class HoechstTargetConfig(TargetDatasetConfigBase, frozen=True):
    name: Literal["Hoechst"] = "Hoechst"
    loader: TIFLoaderMetaConfig = TIFLoaderMetaConfig(
        dataset="Hoechst_Dataset",
        batch_size=2,
        num_workers=8,
        global_norm=True,
        percentiles=(5, 98),
        image_dir=("/Hoechst/test_nuclei/images/png",),
        mask_dir=("/Hoechst/test_nuclei/annotations",),
        transformer={
            "raw": [
                {"name": "PercentileNormalizer"},
                {"name": "ToTensor", "expand_dims": True},
            ],
        },
    )
    predictor_semantic: TIFNucleiSemanticPredictorConfig = (
        TIFNucleiSemanticPredictorConfig(
            name="DSB2018Predictor",
        )
    )
    predictor_instance: TIFNucleiInstancePredictorConfig = (
        TIFNucleiInstancePredictorConfig(
            name="NucleiInstancePredictor",
            save_segmentation=True,
            min_size=80,
            zero_largest_instance=False,
            no_adjust_background=False,
        )
    )
    eval_dataloader_semantic: Eval_TIF_DataloaderMetaConfig = (
        Eval_TIF_DataloaderMetaConfig(
            name="Hoechst_Dataset",
            expand_dims=True,
            global_norm=False,
            percentiles=None,
            image_key="predictions",
            min_object_size=None,
            zero_large_instances=False,
            mask_dir=("/Hoechst/test_nuclei/annotations",),
            mask_key=None,
            batch_size=1,
            num_workers=8,
            transformer={
                "raw": [{"name": "ToTensor", "expand_dims": True}],
                "label": [
                    {"name": "Relabel"},
                    {"name": "BlobsToMask", "append_label": False},
                    {"name": "ToTensor", "expand_dims": True},
                ],
            },
        )
    )
    eval_dataloader_instance: Eval_TIF_DataloaderMetaConfig = (
        Eval_TIF_DataloaderMetaConfig(
            name="Hoechst_Dataset",
            expand_dims=True,
            global_norm=False,
            percentiles=None,
            image_key="segmentation",
            min_object_size=80,
            zero_large_instances=False,
            mask_dir=("/Hoechst/test_nuclei/annotations",),
            mask_key=None,
            batch_size=1,
            num_workers=8,
            transformer={
                "raw": [{"name": "ToTensor", "expand_dims": True}],
                "label": [
                    {"name": "Relabel"},
                    {"name": "ToTensor", "expand_dims": True},
                ],
            },
        )
    )
    consis_dataloader_semantic: Eval_TIF_DataloaderMetaConfig = (
        Eval_TIF_DataloaderMetaConfig(
            name="Hoechst_Dataset",
            expand_dims=True,
            global_norm=False,
            percentiles=None,
            image_key="predictions",
            min_object_size=None,
            zero_large_instances=False,
            mask_dir=None,
            mask_key="predictions",
            batch_size=1,
            num_workers=8,
            transformer={
                "raw": [
                    {"name": "ToTensor", "expand_dims": True},
                ],
                "label": [
                    {"name": "ToTensor", "expand_dims": True},
                ],
            },
        )
    )
    consis_dataloader_instance: Eval_TIF_DataloaderMetaConfig = (
        Eval_TIF_DataloaderMetaConfig(
            name="Hoechst_Dataset",
            expand_dims=True,
            global_norm=False,
            percentiles=None,
            image_key="segmentation",
            min_object_size=80,
            zero_large_instances=False,
            mask_dir=None,
            mask_key="segmentation",
            batch_size=1,
            num_workers=8,
            transformer={
                "raw": [
                    {"name": "ToTensor", "expand_dims": True},
                ],
                "label": [
                    {"name": "ToTensor", "expand_dims": True},
                ],
            },
        )
    )
    filter_results: None = None


class SBIAD634TargetConfig(TargetDatasetConfigBase, frozen=True):
    name: Literal["S_BIAD634"] = "S_BIAD634"
    loader: TIFtxtLoaderMetaConfig = TIFtxtLoaderMetaConfig(
        dataset="TIF_txt_Dataset",
        batch_size=2,
        num_workers=8,
        global_norm=False,
        percentiles=(5, 98),
        image_dir=("/S-BIAD634/dataset/rawimages",),
        mask_dir=("/S-BIAD634/dataset/groundtruth",),
        filenames_path="/S-BIAD634/dataset/test.txt",
        transformer={
            "raw": [
                {"name": "PercentileNormalizer"},
                {"name": "ToTensor", "expand_dims": True},
            ],
        },
    )
    predictor_semantic: TIFNucleiSemanticPredictorConfig = (
        TIFNucleiSemanticPredictorConfig(
            name="DSB2018Predictor",
        )
    )
    predictor_instance: TIFNucleiInstancePredictorConfig = (
        TIFNucleiInstancePredictorConfig(
            name="NucleiInstancePredictor",
            save_segmentation=True,
            min_size=0,
            zero_largest_instance=False,
            no_adjust_background=False,
        )
    )
    eval_dataloader_semantic: Eval_TIF_TxtDataloaderMetaConfig = (
        Eval_TIF_TxtDataloaderMetaConfig(
            batch_size=1,
            num_workers=8,
            name="TIF_txt_Dataset",
            expand_dims=True,
            global_norm=False,
            percentiles=None,
            image_key="predictions",
            min_object_size=None,
            zero_large_instances=False,
            mask_dir=("/S-BIAD634/dataset/groundtruth",),
            mask_key=None,
            filenames_path="/S-BIAD634/dataset/test.txt",
            transformer={
                "raw": [{"name": "ToTensor", "expand_dims": True}],
                "label": [
                    {"name": "Relabel"},
                    {"name": "BlobsToMask", "append_label": False},
                    {"name": "ToTensor", "expand_dims": True},
                ],
            },
        )
    )
    eval_dataloader_instance: Eval_TIF_TxtDataloaderMetaConfig = (
        Eval_TIF_TxtDataloaderMetaConfig(
            batch_size=1,
            num_workers=8,
            name="TIF_txt_Dataset",
            expand_dims=True,
            global_norm=False,
            percentiles=None,
            image_key="segmentation",
            min_object_size=1,
            zero_large_instances=False,
            mask_dir=("/S-BIAD634/dataset/groundtruth",),
            mask_key=None,
            filenames_path="/S-BIAD634/dataset/test.txt",
            transformer={
                "raw": [{"name": "ToTensor", "expand_dims": True}],
                "label": [
                    {"name": "ToTensor", "expand_dims": True},
                ],
            },
        )
    )
    consis_dataloader_semantic: Eval_TIF_TxtDataloaderMetaConfig = (
        Eval_TIF_TxtDataloaderMetaConfig(
            batch_size=1,
            num_workers=8,
            name="TIF_txt_Dataset",
            expand_dims=True,
            global_norm=False,
            percentiles=None,
            image_key="predictions",
            min_object_size=None,
            zero_large_instances=False,
            mask_dir=None,
            mask_key="predictions",
            filenames_path="/S-BIAD634/dataset/test.txt",
            transformer={
                "raw": [
                    {"name": "ToTensor", "expand_dims": True},
                ],
                "label": [
                    {"name": "ToTensor", "expand_dims": True},
                ],
            },
        )
    )
    consis_dataloader_instance: Eval_TIF_TxtDataloaderMetaConfig = (
        Eval_TIF_TxtDataloaderMetaConfig(
            batch_size=1,
            num_workers=8,
            name="TIF_txt_Dataset",
            expand_dims=True,
            global_norm=False,
            percentiles=None,
            image_key="segmentation",
            min_object_size=1,
            zero_large_instances=False,
            mask_dir=None,
            mask_key="segmentation",
            filenames_path="/S-BIAD634/dataset/test.txt",
            transformer={
                "raw": [
                    {"name": "ToTensor", "expand_dims": True},
                ],
                "label": [
                    {"name": "ToTensor", "expand_dims": True},
                ],
            },
        )
    )
    filter_results: None = None


class SBIAD895TargetConfig(TargetDatasetConfigBase, frozen=True):
    name: Literal["S_BIAD895"] = "S_BIAD895"
    loader: TIFLoaderMetaConfig = TIFLoaderMetaConfig(
        dataset="Standard_TIF_Dataset",
        batch_size=5,
        num_workers=8,
        global_norm=False,
        percentiles=(5, 98),
        image_dir=("/S-BIAD895/ZeroCostDL4Mic/Stardist_v2/Stardist/Train/Raw",),
        mask_dir=("/S-BIAD895/ZeroCostDL4Mic/Stardist_v2/Stardist/Train/Masks",),
        transformer={
            "raw": [
                {"name": "PercentileNormalizer"},
                {"name": "FixedClipping", "min_value": None, "max_value": 2},
                {"name": "ToTensor", "expand_dims": True},
            ],
        },
    )
    predictor_semantic: TIFNucleiSemanticPredictorConfig = (
        TIFNucleiSemanticPredictorConfig(
            name="DSB2018Predictor",
        )
    )
    predictor_instance: TIFNucleiInstancePredictorConfig = (
        TIFNucleiInstancePredictorConfig(
            name="NucleiInstancePredictor",
            save_segmentation=True,
            min_size=50,
            zero_largest_instance=False,
            no_adjust_background=False,
        )
    )
    eval_dataloader_semantic: Eval_TIF_DataloaderMetaConfig = (
        Eval_TIF_DataloaderMetaConfig(
            batch_size=1,
            num_workers=8,
            name="Standard_TIF_Dataset",
            expand_dims=True,
            global_norm=False,
            percentiles=None,
            image_key="predictions",
            min_object_size=None,
            zero_large_instances=False,
            mask_dir=("/S-BIAD895/ZeroCostDL4Mic/Stardist_v2/Stardist/Train/Masks",),
            mask_key=None,
            transformer={
                "raw": [{"name": "ToTensor", "expand_dims": True}],
                "label": [
                    {"name": "Relabel"},
                    {"name": "BlobsToMask", "append_label": False},
                    {"name": "ToTensor", "expand_dims": True},
                ],
            },
        )
    )
    eval_dataloader_instance: Eval_TIF_DataloaderMetaConfig = (
        Eval_TIF_DataloaderMetaConfig(
            batch_size=1,
            num_workers=8,
            name="Standard_TIF_Dataset",
            expand_dims=True,
            global_norm=False,
            percentiles=None,
            image_key="segmentation",
            min_object_size=50,
            zero_large_instances=False,
            mask_dir=("/S-BIAD895/ZeroCostDL4Mic/Stardist_v2/Stardist/Train/Masks",),
            mask_key=None,
            transformer={
                "raw": [{"name": "ToTensor", "expand_dims": True}],
                "label": [{"name": "ToTensor", "expand_dims": True}],
            },
        )
    )
    consis_dataloader_semantic: Eval_TIF_DataloaderMetaConfig = (
        Eval_TIF_DataloaderMetaConfig(
            batch_size=1,
            num_workers=8,
            name="Standard_TIF_Dataset",
            expand_dims=True,
            global_norm=False,
            percentiles=None,
            image_key="predictions",
            min_object_size=None,
            zero_large_instances=False,
            mask_dir=None,
            mask_key="predictions",
            transformer={
                "raw": [
                    {"name": "ToTensor", "expand_dims": True},
                ],
                "label": [
                    {"name": "ToTensor", "expand_dims": True},
                ],
            },
        )
    )
    consis_dataloader_instance: Eval_TIF_DataloaderMetaConfig = (
        Eval_TIF_DataloaderMetaConfig(
            batch_size=1,
            num_workers=8,
            name="Standard_TIF_Dataset",
            expand_dims=True,
            global_norm=False,
            percentiles=None,
            image_key="segmentation",
            min_object_size=50,
            zero_large_instances=False,
            mask_dir=None,
            mask_key="segmentation",
            transformer={
                "raw": [
                    {"name": "ToTensor", "expand_dims": True},
                ],
                "label": [
                    {"name": "ToTensor", "expand_dims": True},
                ],
            },
        )
    )
    filter_results: None = None


class SBIAD1196TargetConfig(TargetDatasetConfigBase, frozen=True):
    name: Literal["S_BIAD1196"] = "S_BIAD1196"
    loader: Pytorch3DUnetLoaderMetaConfig = Pytorch3DUnetLoaderMetaConfig(
        dataset="StandardHDF5Dataset",
        batch_size=32,
        num_workers=8,
        raw_internal_path="raw",
        label_internal_path="label",
        global_normalization=True,
        global_percentiles=(5, 98),
        file_paths=("/S-BIAD1196/SELMA3D_training_annotated/shannel_cells/h5/test",),
        roi=None,
        transformer={
            "raw": [
                {"name": "PercentileNormalizer"},
                {"name": "ToTensor", "expand_dims": True},
            ],
        },
        slice_builder=Pytorch3DUnetSliceBuilderConfig(
            name="SliceBuilder",
            patch_shape=(1, 200, 200),
            stride_shape=(1, 200, 200),
            halo_shape=(0, 32, 32),
        ),
    )
    predictor_semantic: Pytorch3DUnetPredictorMetaConfig = (
        Pytorch3DUnetPredictorMetaConfig(
            name="PatchWisePredictor",
            save_segmentation=False,
            min_size=None,
            layer_id=None,
            beta=0.5,
            zero_largest_instance=False,
            zero_large_instances=False,
            large_instance_multiplier=4,
            max_obj_size=None,
        )
    )
    predictor_instance: Pytorch3DUnetPredictorMetaConfig = (
        Pytorch3DUnetPredictorMetaConfig(
            name="PatchWisePredictor",
            save_segmentation=True,
            min_size=1,
            layer_id=None,
            beta=0.5,
            zero_largest_instance=False,
            zero_large_instances=True,
            large_instance_multiplier=4,
            max_obj_size=None,
        )
    )
    eval_dataloader_semantic: EvalDataloaderMetaConfig = EvalDataloaderMetaConfig(
        name="StandardEvalDataset",
        gt_path=("/S-BIAD1196/SELMA3D_training_annotated/shannel_cells/h5/test",),
        pred_key="predictions",
        gt_key="label",
        patch_key="patch_index",
        roi=None,
        ignore_index=None,
        ignore_path=None,
        ignore_key=None,
        convert_to_boundary_label=False,
        convert_to_binary_label=False,
        min_object_size=None,
        gt_zero_largest_instance=False,
        gt_zero_large_instances=False,
        zero_large_instances=False,
        zero_largest_instance=False,
        largest_obj_multiplier=4,
        max_obj_size=None,
        batch_size=1,
        num_workers=8,
    )
    eval_dataloader_instance: EvalDataloaderMetaConfig = EvalDataloaderMetaConfig(
        name="StandardEvalDataset",
        gt_path=("/S-BIAD1196/SELMA3D_training_annotated/shannel_cells/h5/test",),
        pred_key="segmentation",
        gt_key="label",
        patch_key="patch_index",
        roi=None,
        ignore_index=None,
        ignore_path=None,
        ignore_key=None,
        convert_to_boundary_label=False,
        convert_to_binary_label=False,
        min_object_size=1,
        gt_zero_largest_instance=False,
        gt_zero_large_instances=False,
        zero_large_instances=False,
        zero_largest_instance=False,
        largest_obj_multiplier=4,
        max_obj_size=None,
        batch_size=1,
        num_workers=8,
    )
    consis_dataloader_semantic: EvalDataloaderMetaConfig = EvalDataloaderMetaConfig(
        name="StandardEvalDataset",
        gt_path=None,
        pred_key="predictions",
        gt_key="predictions",
        patch_key="patch_index",
        roi=None,
        ignore_index=None,
        ignore_path=None,
        ignore_key=None,
        convert_to_boundary_label=False,
        convert_to_binary_label=False,
        min_object_size=None,
        gt_zero_largest_instance=False,
        gt_zero_large_instances=False,
        zero_large_instances=False,
        zero_largest_instance=False,
        largest_obj_multiplier=4,
        max_obj_size=None,
        batch_size=1,
        num_workers=8,
    )
    consis_dataloader_instance: EvalDataloaderMetaConfig = EvalDataloaderMetaConfig(
        name="StandardEvalDataset",
        gt_path=None,
        pred_key="segmentation",
        gt_key="segmentation",
        patch_key="patch_index",
        roi=None,
        ignore_index=None,
        ignore_path=None,
        ignore_key=None,
        convert_to_boundary_label=False,
        convert_to_binary_label=False,
        min_object_size=1,
        gt_zero_largest_instance=False,
        gt_zero_large_instances=False,
        zero_large_instances=False,
        zero_largest_instance=False,
        largest_obj_multiplier=4,
        max_obj_size=None,
        batch_size=1,
        num_workers=8,
    )
    filter_results: ForegroundFilterConfig = ForegroundFilterConfig(
        name="ForegroundFilter",
        foreground_threshold=0,
        gt_dir_path="/S-BIAD1196/SELMA3D_training_annotated/shannel_cells/h5/test/",
        gt_key="label",
        roi=None,
        save_selection=True,
        overwrite=False,
    )


class SBIAD1410TargetConfig(TargetDatasetConfigBase, frozen=True):
    name: Literal["S_BIAD1410"] = "S_BIAD1410"
    loader: SBIAD1410LoaderMetaConfig = SBIAD1410LoaderMetaConfig(
        dataset="S_BIAD1410_Dataset",
        batch_size=32,
        num_workers=8,
        global_normalization=True,
        global_percentiles=(5, 98),
        img_paths=("/S-BIAD1410/cardioblast_nuclei/cardioblast_nuclei_test",),
        mask_paths=("/S-BIAD1410/cardioblast_nuclei/cardioblast_nuclei_test",),
        roi=None,
        transformer={
            "raw": [
                {"name": "PercentileNormalizer"},
                {"name": "ToTensor", "expand_dims": True},
            ],
        },
        slice_builder=Pytorch3DUnetSliceBuilderConfig(
            name="SliceBuilder",
            patch_shape=(1, 256, 256),
            stride_shape=(1, 256, 256),
            halo_shape=(0, 32, 32),
        ),
    )
    predictor_semantic: Pytorch3DUnetPredictorMetaConfig = (
        Pytorch3DUnetPredictorMetaConfig(
            name="PatchWisePredictor",
            save_segmentation=False,
            min_size=None,
            layer_id=None,
            beta=0.5,
            zero_largest_instance=False,
            zero_large_instances=False,
            large_instance_multiplier=4,
            max_obj_size=None,
        )
    )
    predictor_instance: Pytorch3DUnetPredictorMetaConfig = (
        Pytorch3DUnetPredictorMetaConfig(
            name="PatchWisePredictor",
            save_segmentation=True,
            min_size=1,
            layer_id=None,
            beta=0.5,
            zero_largest_instance=False,
            zero_large_instances=True,
            large_instance_multiplier=4,
            max_obj_size=None,
        )
    )
    eval_dataloader_semantic: EvalSB1410DataloaderMetaConfig = (
        EvalSB1410DataloaderMetaConfig(
            name="S_BIAD1410_Dataset",
            eval=SBIAD1410PhaseMetaConfig(
                mask_paths=("/S-BIAD1410/cardioblast_nuclei/cardioblast_nuclei_test",),
                roi=None,
                transformer={
                    "raw": [{"name": "ToTensor", "expand_dims": True}],
                    "label": [
                        {"name": "Relabel"},
                        {"name": "BlobsToMask", "append_label": False},
                        {"name": "ToTensor", "expand_dims": True},
                    ],
                },
                slice_builder=Pytorch3DUnetSliceBuilderConfig(
                    name="SliceBuilder",
                    patch_shape=(1, 256, 256),
                    stride_shape=(1, 256, 256),
                    halo_shape=(0, 32, 32),
                ),
            ),
            global_normalization=False,
            global_percentiles=None,
            image_key="predictions",
            mask_key=None,
            zero_large_instances=False,
            batch_size=1,
            num_workers=8,
        )
    )
    eval_dataloader_instance: EvalSB1410DataloaderMetaConfig = (
        EvalSB1410DataloaderMetaConfig(
            name="S_BIAD1410_Dataset",
            eval=SBIAD1410PhaseMetaConfig(
                mask_paths=("/S-BIAD1410/cardioblast_nuclei/cardioblast_nuclei_test",),
                roi=None,
                transformer={
                    "raw": [{"name": "ToTensor", "expand_dims": True}],
                    "label": [{"name": "ToTensor", "expand_dims": True}],
                },
                slice_builder=Pytorch3DUnetSliceBuilderConfig(
                    name="SliceBuilder",
                    patch_shape=(1, 256, 256),
                    stride_shape=(1, 256, 256),
                    halo_shape=(0, 32, 32),
                ),
            ),
            global_normalization=False,
            global_percentiles=None,
            image_key="segmentation",
            mask_key=None,
            zero_large_instances=False,
            batch_size=1,
            num_workers=8,
        )
    )
    consis_dataloader_semantic: EvalSB1410DataloaderMetaConfig = (
        EvalSB1410DataloaderMetaConfig(
            name="S_BIAD1410_Dataset",
            eval=SBIAD1410PhaseMetaConfig(
                mask_paths=None,
                roi=None,
                transformer={
                    "raw": [
                        {"name": "ToTensor", "expand_dims": True},
                    ],
                    "label": [
                        {"name": "ToTensor", "expand_dims": True},
                    ],
                },
                slice_builder=Pytorch3DUnetSliceBuilderConfig(
                    name="SliceBuilder",
                    patch_shape=(1, 256, 256),
                    stride_shape=(1, 256, 256),
                    halo_shape=(0, 32, 32),
                ),
            ),
            global_normalization=False,
            global_percentiles=None,
            image_key="predictions",
            mask_key=None,
            zero_large_instances=False,
            batch_size=1,
            num_workers=8,
        )
    )
    consis_dataloader_instance: EvalSB1410DataloaderMetaConfig = (
        EvalSB1410DataloaderMetaConfig(
            name="S_BIAD1410_Dataset",
            eval=SBIAD1410PhaseMetaConfig(
                mask_paths=("/S-BIAD1410/cardioblast_nuclei/cardioblast_nuclei_test",),
                roi=None,
                transformer={
                    "raw": [{"name": "ToTensor", "expand_dims": True}],
                    "label": [{"name": "ToTensor", "expand_dims": True}],
                },
                slice_builder=Pytorch3DUnetSliceBuilderConfig(
                    name="SliceBuilder",
                    patch_shape=(1, 256, 256),
                    stride_shape=(1, 256, 256),
                    halo_shape=(0, 32, 32),
                ),
            ),
            global_normalization=False,
            global_percentiles=None,
            image_key="segmentation",
            mask_key=None,
            zero_large_instances=False,
            batch_size=1,
            num_workers=8,
        )
    )
    filter_results: ForegroundFilterConfig = ForegroundFilterConfig(
        name="ForegroundFilter",
        foreground_threshold=0.01,
        gt_dir_path="/S-BIAD1410/cardioblast_nuclei/cardioblast_nuclei_test/",
        gt_key=None,
        roi=None,
        save_selection=True,
        overwrite=False,
    )


class DSB2018TargetConfig(TargetDatasetConfigBase, frozen=True):
    name: Literal["DSB2018"] = "DSB2018"
    loader: TIFLoaderMetaConfig = TIFLoaderMetaConfig(
        batch_size=32,
        num_workers=8,
        global_norm=False,
        percentiles=(5, 98),
        dataset="Standard_TIF_Dataset",
        image_dir=("/dsb2018_fluorescence/test/images",),
        mask_dir=("/dsb2018_fluorescence/test/masks",),
        transformer={
            "raw": [
                {"name": "PercentileNormalizer"},
                {"name": "ToTensor", "expand_dims": True},
            ],
        },
    )
    predictor_semantic: TIFNucleiSemanticPredictorConfig = (
        TIFNucleiSemanticPredictorConfig(
            name="DSB2018Predictor",
        )
    )
    predictor_instance: TIFNucleiInstancePredictorConfig = (
        TIFNucleiInstancePredictorConfig(
            name="NucleiInstancePredictor",
            save_segmentation=True,
            min_size=1,
            zero_largest_instance=False,
            no_adjust_background=False,
        )
    )
    eval_dataloader_semantic: Eval_TIF_DataloaderMetaConfig = (
        Eval_TIF_DataloaderMetaConfig(
            batch_size=1,
            num_workers=8,
            name="Standard_TIF_Dataset",
            expand_dims=True,
            global_norm=False,
            percentiles=None,
            image_key="predictions",
            min_object_size=None,
            zero_large_instances=False,
            mask_dir=("/dsb2018_fluorescence/test/masks",),
            mask_key=None,
            transformer={
                "raw": [
                    {"name": "ToTensor", "expand_dims": True},
                ],
                "label": [
                    {"name": "Relabel"},
                    {"name": "BlobsToMask", "append_label": False},
                    {"name": "ToTensor", "expand_dims": True},
                ],
            },
        )
    )
    eval_dataloader_instance: Eval_TIF_DataloaderMetaConfig = (
        Eval_TIF_DataloaderMetaConfig(
            batch_size=1,
            num_workers=8,
            name="Standard_TIF_Dataset",
            expand_dims=True,
            global_norm=False,
            percentiles=None,
            image_key="segmentation",
            min_object_size=None,
            zero_large_instances=False,
            mask_dir=("/dsb2018_fluorescence/test/masks",),
            mask_key=None,
            transformer={
                "raw": [
                    {"name": "ToTensor", "expand_dims": True},
                ],
                "label": [
                    {"name": "ToTensor", "expand_dims": True},
                ],
            },
        )
    )
    consis_dataloader_semantic: Eval_TIF_DataloaderMetaConfig = (
        Eval_TIF_DataloaderMetaConfig(
            batch_size=1,
            num_workers=8,
            name="Standard_TIF_Dataset",
            expand_dims=True,
            global_norm=False,
            percentiles=None,
            image_key="predictions",
            min_object_size=None,
            zero_large_instances=False,
            mask_dir=None,
            mask_key="predictions",
            transformer={
                "raw": [
                    {"name": "ToTensor", "expand_dims": True},
                ],
                "label": [
                    {"name": "ToTensor", "expand_dims": True},
                ],
            },
        )
    )
    consis_dataloader_instance: Eval_TIF_DataloaderMetaConfig = (
        Eval_TIF_DataloaderMetaConfig(
            batch_size=1,
            num_workers=8,
            name="Standard_TIF_Dataset",
            expand_dims=True,
            global_norm=False,
            percentiles=None,
            image_key="segmentation",
            min_object_size=None,
            zero_large_instances=False,
            mask_dir=None,
            mask_key="segmentation",
            transformer={
                "raw": [
                    {"name": "ToTensor", "expand_dims": True},
                ],
                "label": [
                    {"name": "ToTensor", "expand_dims": True},
                ],
            },
        )
    )
    filter_results: None = None


class GoNuclearTargetConfig(TargetDatasetConfigBase, frozen=True):
    name: Literal["Go-Nuclear"] = "Go-Nuclear"
    loader: Pytorch3DUnetLoaderMetaConfig = Pytorch3DUnetLoaderMetaConfig(
        dataset="StandardHDF5Dataset",
        batch_size=32,
        num_workers=8,
        raw_internal_path="raw/clear",
        label_internal_path="label/gold",
        global_normalization=True,
        global_percentiles=(0, 99.8),
        file_paths=("/Go-Nuclear/3d_all_in_one/1170.h5",),
        roi=[[50, 170]],
        transformer={
            "raw": [
                {"name": "PercentileNormalizer"},
                {"name": "ToTensor", "expand_dims": True},
            ]
        },
        slice_builder=Pytorch3DUnetSliceBuilderConfig(
            name="SliceBuilder",
            patch_shape=(1, 256, 256),
            stride_shape=(1, 256, 256),
            halo_shape=(0, 32, 32),
        ),
    )
    predictor_semantic: Pytorch3DUnetPredictorMetaConfig = (
        Pytorch3DUnetPredictorMetaConfig(
            name="PatchWisePredictor",
            save_segmentation=False,
            min_size=None,
            layer_id=None,
            beta=0.5,
            zero_largest_instance=False,
            zero_large_instances=False,
            large_instance_multiplier=4,
            max_obj_size=None,
        )
    )
    predictor_instance: Pytorch3DUnetPredictorMetaConfig = (
        Pytorch3DUnetPredictorMetaConfig(
            name="PatchWisePredictor",
            save_segmentation=True,
            min_size=50,
            layer_id=None,
            beta=0.5,
            zero_largest_instance=False,
            zero_large_instances=True,
            large_instance_multiplier=4,
            max_obj_size=None,
        )
    )

    eval_dataloader_semantic: EvalDataloaderMetaConfig = EvalDataloaderMetaConfig(
        name="StandardEvalDataset",
        gt_path=("/Go-Nuclear/3d_all_in_one/1170.h5",),
        pred_key="predictions",
        gt_key="label/gold",
        patch_key="patch_index",
        roi=[[50, 170]],
        ignore_index=None,
        ignore_path=None,
        ignore_key=None,
        convert_to_boundary_label=False,
        convert_to_binary_label=True,
        min_object_size=None,
        gt_zero_largest_instance=False,
        gt_zero_large_instances=False,
        zero_large_instances=False,
        zero_largest_instance=False,
        largest_obj_multiplier=4,
        max_obj_size=None,
        batch_size=1,
        num_workers=8,
    )
    eval_dataloader_instance: EvalDataloaderMetaConfig = EvalDataloaderMetaConfig(
        name="StandardEvalDataset",
        gt_path=("/Go-Nuclear/3d_all_in_one/1170.h5",),
        pred_key="segmentation",
        gt_key="label/gold",
        patch_key="patch_index",
        roi=[[50, 170]],
        ignore_index=None,
        ignore_path=None,
        ignore_key=None,
        convert_to_boundary_label=False,
        convert_to_binary_label=False,
        min_object_size=50,
        gt_zero_largest_instance=False,
        gt_zero_large_instances=False,
        zero_large_instances=False,
        zero_largest_instance=False,
        largest_obj_multiplier=4,
        max_obj_size=None,
        batch_size=1,
        num_workers=8,
    )
    consis_dataloader_semantic: EvalDataloaderMetaConfig = EvalDataloaderMetaConfig(
        name="StandardEvalDataset",
        gt_path=None,
        pred_key="predictions",
        gt_key="predictions",
        patch_key="patch_index",
        roi=[[50, 170]],
        ignore_index=None,
        ignore_path=None,
        ignore_key=None,
        convert_to_boundary_label=False,
        convert_to_binary_label=False,
        min_object_size=None,
        gt_zero_largest_instance=False,
        gt_zero_large_instances=False,
        zero_large_instances=False,
        zero_largest_instance=False,
        largest_obj_multiplier=4,
        max_obj_size=None,
        batch_size=1,
        num_workers=8,
    )
    consis_dataloader_instance: EvalDataloaderMetaConfig = EvalDataloaderMetaConfig(
        name="StandardEvalDataset",
        gt_path=None,
        pred_key="segmentation",
        gt_key="segmentation",
        patch_key="patch_index",
        roi=[[50, 170]],
        ignore_index=None,
        ignore_path=None,
        ignore_key=None,
        convert_to_boundary_label=False,
        convert_to_binary_label=False,
        min_object_size=50,
        gt_zero_largest_instance=False,
        gt_zero_large_instances=False,
        zero_large_instances=False,
        zero_largest_instance=False,
        largest_obj_multiplier=4,
        max_obj_size=None,
        batch_size=1,
        num_workers=8,
    )
    filter_results: ForegroundFilterConfig = ForegroundFilterConfig(
        name="ForegroundFilter",
        foreground_threshold=0.05,
        gt_dir_path="/Go-Nuclear/3d_all_in_one/",
        gt_key="label/gold",
        roi=[[50, 170]],
        save_selection=True,
        overwrite=False,
    )


class FlyWingTargetConfig(TargetDatasetConfigBase, frozen=True):
    name: Literal["FlyWing"] = "FlyWing"
    loader: Pytorch3DUnetLoaderMetaConfig = Pytorch3DUnetLoaderMetaConfig(
        dataset="StandardHDF5Dataset",
        batch_size=32,
        num_workers=8,
        raw_internal_path="volumes/raw",
        label_internal_path="volumes/labels/expanded_cells_with_ignore",
        global_normalization=True,
        global_percentiles=(5, 95),
        file_paths=("/FlyWing/GT/test/per03.h5",),
        roi=None,
        transformer={
            "raw": [
                {"name": "PercentileNormalizer"},
                {"name": "ToTensor", "expand_dims": True},
            ]
        },
        slice_builder=Pytorch3DUnetSliceBuilderConfig(
            name="SliceBuilder",
            patch_shape=(1, 256, 256),
            stride_shape=(1, 256, 256),
            halo_shape=(0, 32, 32),
            # halo_shape=(0, 0, 0),
        ),
    )
    predictor_semantic: None = None
    predictor_instance: Pytorch3DUnetPredictorMetaConfig = (
        Pytorch3DUnetPredictorMetaConfig(
            name="PatchWisePredictor",
            save_segmentation=True,
            min_size=50,
            layer_id=None,
            # beta=0.8,
            beta=0.5,
            zero_largest_instance=False,
            zero_large_instances=True,
            # zero_large_instances=False,
            large_instance_multiplier=2,
            max_obj_size=3303,
        )
    )
    eval_dataloader_semantic: None = None
    eval_dataloader_instance: EvalDataloaderMetaConfig = EvalDataloaderMetaConfig(
        name="StandardEvalDataset",
        gt_path=("/FlyWing/GT/test/per03.h5",),
        pred_key="segmentation",
        gt_key="volumes/labels/expanded_cells_with_ignore",
        patch_key="patch_index",
        roi=None,
        ignore_index=-1,
        ignore_path=None,
        ignore_key=None,
        convert_to_boundary_label=False,
        convert_to_binary_label=False,
        min_object_size=50,
        gt_zero_largest_instance=False,
        gt_zero_large_instances=False,
        zero_large_instances=False,
        zero_largest_instance=False,
        largest_obj_multiplier=1.5,
        max_obj_size=3303,
        batch_size=32,
        num_workers=8,
    )
    consis_dataloader_semantic: None = None
    consis_dataloader_instance: EvalDataloaderMetaConfig = EvalDataloaderMetaConfig(
        name="StandardEvalDataset",
        gt_path=None,
        pred_key="segmentation",
        gt_key="segmentation",
        patch_key="patch_index",
        roi=None,
        ignore_index=None,
        ignore_path="/FlyWing/GT/test/per03.h5",
        ignore_key="volumes/labels/ignore",
        convert_to_boundary_label=False,
        convert_to_binary_label=False,
        min_object_size=50,
        gt_zero_largest_instance=False,
        gt_zero_large_instances=False,
        zero_large_instances=False,
        zero_largest_instance=False,
        largest_obj_multiplier=1.5,
        max_obj_size=3303,
        batch_size=32,
        num_workers=8,
    )
    filter_results: ForegroundFilterConfig = ForegroundFilterConfig(
        name="ForegroundFilter",
        foreground_threshold=0.01,
        gt_dir_path="/FlyWing/GT/test/",
        gt_key="volumes/labels/cells",
        roi=None,
        save_selection=True,
        overwrite=False,
    )


class OvulesTargetConfig(TargetDatasetConfigBase, frozen=True):
    name: Literal["Ovules"] = "Ovules"
    loader: Pytorch3DUnetLoaderMetaConfig = Pytorch3DUnetLoaderMetaConfig(
        dataset="StandardHDF5Dataset",
        batch_size=32,
        num_workers=8,
        raw_internal_path="raw",
        label_internal_path="label_with_ignore",
        global_normalization=True,
        global_percentiles=(5, 95),
        file_paths=("/Ovules/GT2x/test/N_294_final_crop_ds2.h5",),
        roi=None,
        transformer={
            "raw": [
                {"name": "PercentileNormalizer"},
                {"name": "ToTensor", "expand_dims": True},
            ]
        },
        slice_builder=Pytorch3DUnetSliceBuilderConfig(
            name="SliceBuilder",
            patch_shape=(1, 256, 256),
            stride_shape=(1, 256, 256),
            halo_shape=(0, 32, 32),
            # halo_shape=(0, 0, 0),
        ),
    )
    predictor_semantic: None = None
    predictor_instance: Pytorch3DUnetPredictorMetaConfig = (
        Pytorch3DUnetPredictorMetaConfig(
            name="PatchWisePredictor",
            save_segmentation=True,
            min_size=1,
            layer_id=None,
            beta=0.5,
            zero_largest_instance=False,
            zero_large_instances=True,
            large_instance_multiplier=1.5,
            max_obj_size=5867,
        )
    )
    eval_dataloader_semantic: None = None
    eval_dataloader_instance: EvalDataloaderMetaConfig = EvalDataloaderMetaConfig(
        name="StandardEvalDataset",
        gt_path=("/Ovules/GT2x/test/N_294_final_crop_ds2.h5",),
        pred_key="segmentation",
        gt_key="label_with_ignore",
        patch_key="patch_index",
        roi=None,
        ignore_index=-1,
        ignore_path=None,
        ignore_key=None,
        convert_to_boundary_label=False,
        convert_to_binary_label=False,
        min_object_size=1,
        gt_zero_largest_instance=False,
        gt_zero_large_instances=False,
        zero_large_instances=False,
        zero_largest_instance=False,
        largest_obj_multiplier=1.5,
        max_obj_size=5867,
        batch_size=32,
        num_workers=8,
    )
    consis_dataloader_semantic: None = None
    consis_dataloader_instance: EvalDataloaderMetaConfig = EvalDataloaderMetaConfig(
        name="StandardEvalDataset",
        gt_path=None,
        pred_key="segmentation",
        gt_key="segmentation",
        patch_key="patch_index",
        roi=None,
        ignore_index=None,
        ignore_path="/Ovules/GT2x/test/N_294_final_crop_ds2.h5",
        ignore_key="ignore_mask",
        convert_to_boundary_label=False,
        convert_to_binary_label=False,
        min_object_size=1,
        gt_zero_large_instances=True,
        gt_zero_largest_instance=False,
        zero_large_instances=True,
        zero_largest_instance=False,
        largest_obj_multiplier=1.5,
        max_obj_size=5867,
        batch_size=32,
        num_workers=8,
    )
    filter_results: ForegroundFilterConfig = ForegroundFilterConfig(
        name="ForegroundFilter",
        foreground_threshold=0.3,
        gt_dir_path="/Ovules/GT2x/test/",
        gt_key="label",
        roi=None,
        save_selection=True,
        overwrite=False,
    )


class PNASTargetConfig(TargetDatasetConfigBase, frozen=True):
    name: Literal["PNAS"] = "PNAS"
    loader: Pytorch3DUnetLoaderMetaConfig = Pytorch3DUnetLoaderMetaConfig(
        dataset="StandardHDF5Dataset",
        batch_size=32,
        num_workers=8,
        raw_internal_path="raw",
        label_internal_path="label",
        global_normalization=True,
        global_percentiles=(5, 95),
        file_paths=(
            # "/PNAS/test/12hrs_plant18_trim-acylYFP.h5",
            "/PNAS/test/24hrs_plant18_trim-acylYFP.h5",
            "/PNAS/test/36hrs_plant18_trim-acylYFP.h5",
        ),
        roi=None,
        transformer={
            "raw": [
                {"name": "PercentileNormalizer"},
                {"name": "ToTensor", "expand_dims": True},
            ]
        },
        slice_builder=Pytorch3DUnetSliceBuilderConfig(
            name="SliceBuilder",
            patch_shape=(1, 256, 256),
            stride_shape=(1, 256, 256),
            halo_shape=(0, 32, 32),
            # halo_shape=(0, 0, 0),
        ),
    )
    predictor_semantic: None = None
    predictor_instance: Pytorch3DUnetPredictorMetaConfig = (
        Pytorch3DUnetPredictorMetaConfig(
            name="PatchWisePredictor",
            save_segmentation=True,
            min_size=50,
            layer_id=None,
            beta=0.5,
            zero_largest_instance=False,
            zero_large_instances=False,
            large_instance_multiplier=4,
            max_obj_size=None,
        )
    )
    eval_dataloader_semantic: None = None
    eval_dataloader_instance: EvalDataloaderMetaConfig = EvalDataloaderMetaConfig(
        name="StandardEvalDataset",
        gt_path=(
            # "/PNAS/test/12hrs_plant18_trim-acylYFP.h5",
            "/PNAS/test/24hrs_plant18_trim-acylYFP.h5",
            "/PNAS/test/36hrs_plant18_trim-acylYFP.h5",
        ),
        pred_key="segmentation",
        gt_key="label",
        patch_key="patch_index",
        roi=None,
        ignore_index=None,
        ignore_path=None,
        ignore_key=None,
        convert_to_boundary_label=False,
        convert_to_binary_label=False,
        min_object_size=50,
        gt_zero_largest_instance=True,
        gt_zero_large_instances=False,
        zero_large_instances=False,
        zero_largest_instance=False,
        largest_obj_multiplier=4,
        max_obj_size=None,
        batch_size=32,
        num_workers=8,
    )
    consis_dataloader_semantic: None = None
    consis_dataloader_instance: EvalDataloaderMetaConfig = EvalDataloaderMetaConfig(
        name="StandardEvalDataset",
        gt_path=None,
        pred_key="segmentation",
        gt_key="segmentation",
        patch_key="patch_index",
        roi=None,
        ignore_index=None,
        ignore_path=None,
        ignore_key=None,
        convert_to_boundary_label=False,
        convert_to_binary_label=False,
        min_object_size=50,
        gt_zero_largest_instance=False,
        gt_zero_large_instances=False,
        zero_large_instances=False,
        zero_largest_instance=False,
        largest_obj_multiplier=4,
        max_obj_size=None,
        batch_size=32,
        num_workers=8,
    )
    filter_results: ForegroundFilterConfig = ForegroundFilterConfig(
        name="ForegroundFilter",
        foreground_threshold=0.3,
        gt_dir_path="/PNAS/test/",
        gt_key="label",
        roi=None,
        save_selection=True,
        overwrite=False,
    )


class EPFLTargetConfig(TargetDatasetConfigBase, frozen=True):
    name: Literal["EPFL"] = "EPFL"
    loader: Pytorch3DUnetLoaderMetaConfig = Pytorch3DUnetLoaderMetaConfig(
        dataset="StandardHDF5Dataset",
        batch_size=32,
        num_workers=8,
        raw_internal_path="raw",
        label_internal_path="labels",
        global_normalization=True,
        # global_normalization=False,
        global_percentiles=None,
        file_paths=("/EPFL/test.h5",),
        roi=None,
        transformer={
            "raw": [
                {"name": "Normalize"},
                {"name": "ToTensor", "expand_dims": True},
            ]
        },
        slice_builder=Pytorch3DUnetSliceBuilderConfig(
            name="SliceBuilder",
            patch_shape=(1, 256, 256),
            stride_shape=(1, 256, 256),
            halo_shape=(0, 32, 32),
            # halo_shape=(0, 0, 0),
        ),
    )
    train_loader: Pytorch3DUnetLoaderMetaConfig = Pytorch3DUnetLoaderMetaConfig(
        dataset="StandardHDF5Dataset",
        batch_size=10,
        num_workers=8,
        raw_internal_path="raw",
        label_internal_path="labels",
        global_normalization=True,
        # global_normalization=False,
        global_percentiles=None,
        file_paths=("/EPFL/train.h5",),
        roi=None,
        transformer={
            "raw": [
                {"name": "Normalize"},
                {"name": "ToTensor", "expand_dims": True},
            ],
            "label": [
                {"name": "ToTensor", "expand_dims": True},
            ],
        },
        slice_builder=Pytorch3DUnetSliceBuilderConfig(
            name="SliceBuilder",
            patch_shape=(1, 256, 256),
            stride_shape=(1, 256, 256),
            halo_shape=(0, 32, 32),
        ),
    )
    feature_loader: Pytorch3DUnetLoaderMetaConfig = Pytorch3DUnetLoaderMetaConfig(
        dataset="StandardHDF5Dataset",
        batch_size=32,
        num_workers=8,
        raw_internal_path="raw",
        label_internal_path="labels",
        global_normalization=True,
        global_percentiles=None,
        file_paths=("/EPFL/test.h5",),
        # roi=[[0, 2], [0, 480], [0, 640]],
        roi=None,
        transformer={
            "raw": [
                {"name": "Normalize"},
                {"name": "ToTensor", "expand_dims": True},
            ],
            "label": [
                {"name": "ToTensor", "expand_dims": True},
            ],
        },
        slice_builder=Pytorch3DUnetSliceBuilderConfig(
            name="SliceBuilder",
            patch_shape=(1, 256, 256),
            stride_shape=(1, 256, 256),
            halo_shape=(0, 0, 0),
        ),
    )
    # feature_indices_path: Optional[str] = (
    #     "/scratch/talks/sampled_features/semantic_segmentation/mitochondria/feature_indices/EPFL_indices.npz"
    # )
    feature_indices_path: Optional[str] = None
    # feature_indices_path: Optional[str] = (
    #     "/g/kreshuk/talks/model_ranking/notebooks/checks/EPFL_to_EPFL/E_model5_to_EPFL_features.npz"
    # )
    predictor_semantic: Pytorch3DUnetPredictorMetaConfig = (
        Pytorch3DUnetPredictorMetaConfig(
            name="PatchWisePredictor",
            save_segmentation=False,
            min_size=None,
            layer_id=None,
            beta=0.5,
            zero_largest_instance=False,
            zero_large_instances=False,
            large_instance_multiplier=4,
            max_obj_size=None,
        )
    )
    predictor_instance: None = None
    eval_dataloader_semantic: EvalDataloaderMetaConfig = EvalDataloaderMetaConfig(
        name="StandardEvalDataset",
        gt_path=("/EPFL/test.h5",),
        pred_key="predictions",
        gt_key="labels",
        patch_key="patch_index",
        roi=None,
        ignore_index=None,
        ignore_path=None,
        ignore_key=None,
        convert_to_boundary_label=False,
        convert_to_binary_label=False,
        min_object_size=None,
        gt_zero_largest_instance=False,
        gt_zero_large_instances=False,
        zero_large_instances=False,
        zero_largest_instance=False,
        largest_obj_multiplier=4,
        max_obj_size=None,
        batch_size=32,
        num_workers=8,
    )
    eval_dataloader_instance: None = None
    consis_dataloader_semantic: EvalDataloaderMetaConfig = EvalDataloaderMetaConfig(
        name="StandardEvalDataset",
        gt_path=None,
        pred_key="predictions",
        gt_key="predictions",
        patch_key="patch_index",
        roi=None,
        ignore_index=None,
        ignore_path=None,
        ignore_key=None,
        convert_to_boundary_label=False,
        convert_to_binary_label=False,
        min_object_size=None,
        gt_zero_largest_instance=False,
        gt_zero_large_instances=False,
        zero_large_instances=False,
        zero_largest_instance=False,
        largest_obj_multiplier=4,
        max_obj_size=None,
        batch_size=32,
        num_workers=8,
    )
    consis_dataloader_instance: None = None
    filter_results: ForegroundFilterConfig = ForegroundFilterConfig(
        name="ForegroundFilter",
        foreground_threshold=0.02,
        gt_dir_path="/EPFL/",
        gt_key="labels",
        roi=None,
        save_selection=True,
        overwrite=False,
    )


class HmitoTargetConfig(TargetDatasetConfigBase, frozen=True):
    name: Literal["Hmito"] = "Hmito"
    loader: Pytorch3DUnetLoaderMetaConfig = Pytorch3DUnetLoaderMetaConfig(
        dataset="StandardHDF5Dataset",
        batch_size=32,
        num_workers=8,
        raw_internal_path="raw",
        label_internal_path="labels",
        global_normalization=True,
        # global_normalization=False,
        global_percentiles=None,
        file_paths=("/Hmito/test_converted.h5",),
        roi=[[0, 150], [0, 1280], [0, 1280]],
        transformer={
            "raw": [
                {"name": "Normalize"},
                {"name": "ToTensor", "expand_dims": True},
            ]
        },
        slice_builder=Pytorch3DUnetSliceBuilderConfig(
            name="SliceBuilder",
            patch_shape=(1, 256, 256),
            stride_shape=(1, 256, 256),
            halo_shape=(0, 32, 32),
            # halo_shape=(0, 0, 0),
        ),
    )
    train_loader: Pytorch3DUnetLoaderMetaConfig = Pytorch3DUnetLoaderMetaConfig(
        dataset="StandardHDF5Dataset",
        batch_size=32,
        num_workers=8,
        raw_internal_path="raw",
        label_internal_path="labels",
        global_normalization=True,
        # global_normalization=False,
        global_percentiles=None,
        file_paths=("/Hmito/train_converted.h5",),
        roi=[[0, 150], [0, 1280], [0, 1280]],
        transformer={
            "raw": [
                {"name": "Normalize"},
                {"name": "ToTensor", "expand_dims": True},
            ],
            "label": [
                {"name": "ToTensor", "expand_dims": True},
            ],
        },
        slice_builder=Pytorch3DUnetSliceBuilderConfig(
            name="SliceBuilder",
            patch_shape=(1, 256, 256),
            stride_shape=(1, 256, 256),
            halo_shape=(0, 0, 0),
        ),
    )
    feature_loader: Pytorch3DUnetLoaderMetaConfig = Pytorch3DUnetLoaderMetaConfig(
        dataset="StandardHDF5Dataset",
        batch_size=32,
        num_workers=8,
        raw_internal_path="raw",
        label_internal_path="labels",
        global_normalization=True,
        # global_normalization=False,
        global_percentiles=None,
        file_paths=("/Hmito/test_converted.h5",),
        roi=[[0, 150], [0, 1280], [0, 1280]],
        transformer={
            "raw": [
                {"name": "Normalize"},
                {"name": "ToTensor", "expand_dims": True},
            ],
            "label": [
                {"name": "ToTensor", "expand_dims": True},
            ],
        },
        slice_builder=Pytorch3DUnetSliceBuilderConfig(
            name="SliceBuilder",
            patch_shape=(1, 256, 256),
            stride_shape=(1, 256, 256),
            halo_shape=(0, 0, 0),
        ),
    )
    # feature_indices_path: Optional[str] = (
    #     "/scratch/talks/sampled_features/semantic_segmentation/mitochondria/feature_indices/Hmito_indices.npz"
    # )
    feature_indices_path: Optional[str] = None
    predictor_semantic: Pytorch3DUnetPredictorMetaConfig = (
        Pytorch3DUnetPredictorMetaConfig(
            name="PatchWisePredictor",
            save_segmentation=False,
            min_size=None,
            layer_id=None,
            beta=0.5,
            zero_largest_instance=False,
            zero_large_instances=False,
            large_instance_multiplier=4,
            max_obj_size=None,
        )
    )
    predictor_instance: None = None
    eval_dataloader_semantic: EvalDataloaderMetaConfig = EvalDataloaderMetaConfig(
        name="StandardEvalDataset",
        gt_path=("/Hmito/test_converted.h5",),
        pred_key="predictions",
        gt_key="labels",
        patch_key="patch_index",
        roi=[[0, 150], [0, 1280], [0, 1280]],
        ignore_index=None,
        ignore_path=None,
        ignore_key=None,
        convert_to_boundary_label=False,
        convert_to_binary_label=False,
        min_object_size=None,
        gt_zero_largest_instance=False,
        gt_zero_large_instances=False,
        zero_large_instances=False,
        zero_largest_instance=False,
        largest_obj_multiplier=4,
        max_obj_size=None,
        batch_size=32,
        num_workers=8,
    )
    eval_dataloader_instance: None = None
    consis_dataloader_semantic: EvalDataloaderMetaConfig = EvalDataloaderMetaConfig(
        name="StandardEvalDataset",
        gt_path=None,
        pred_key="predictions",
        gt_key="predictions",
        patch_key="patch_index",
        roi=None,
        ignore_index=None,
        ignore_path=None,
        ignore_key=None,
        convert_to_boundary_label=False,
        convert_to_binary_label=False,
        min_object_size=None,
        gt_zero_largest_instance=False,
        gt_zero_large_instances=False,
        zero_large_instances=False,
        zero_largest_instance=False,
        largest_obj_multiplier=4,
        max_obj_size=None,
        batch_size=32,
        num_workers=8,
    )
    consis_dataloader_instance: None = None
    filter_results: ForegroundFilterConfig = ForegroundFilterConfig(
        name="ForegroundFilter",
        foreground_threshold=0.1,
        gt_dir_path="/Hmito/",
        gt_key="labels",
        roi=None,
        save_selection=True,
        overwrite=False,
    )


class RmitoTargetConfig(TargetDatasetConfigBase, frozen=True):
    name: Literal["Rmito"] = "Rmito"
    loader: Pytorch3DUnetLoaderMetaConfig = Pytorch3DUnetLoaderMetaConfig(
        dataset="StandardHDF5Dataset",
        batch_size=32,
        num_workers=8,
        raw_internal_path="raw",
        label_internal_path="labels",
        global_normalization=True,
        # global_normalization=False,
        global_percentiles=None,
        file_paths=("/Rmito/test_converted.h5",),
        roi=[[0, 150], [0, 1280], [0, 1280]],
        transformer={
            "raw": [
                {"name": "Normalize"},
                {"name": "ToTensor", "expand_dims": True},
            ]
        },
        slice_builder=Pytorch3DUnetSliceBuilderConfig(
            name="SliceBuilder",
            patch_shape=(1, 256, 256),
            stride_shape=(1, 256, 256),
            halo_shape=(0, 32, 32),
            # halo_shape=(0, 0, 0),
        ),
    )
    train_loader: Pytorch3DUnetLoaderMetaConfig = Pytorch3DUnetLoaderMetaConfig(
        dataset="StandardHDF5Dataset",
        batch_size=32,
        num_workers=8,
        raw_internal_path="raw",
        label_internal_path="labels",
        global_normalization=True,
        # global_normalization=False,
        global_percentiles=None,
        file_paths=("/Rmito/train_converted.h5",),
        roi=[[0, 150], [0, 1280], [0, 1280]],
        transformer={
            "raw": [
                {"name": "Normalize"},
                {"name": "ToTensor", "expand_dims": True},
            ],
            "label": [
                {"name": "ToTensor", "expand_dims": True},
            ],
        },
        slice_builder=Pytorch3DUnetSliceBuilderConfig(
            name="SliceBuilder",
            patch_shape=(1, 256, 256),
            stride_shape=(1, 256, 256),
            halo_shape=(0, 0, 0),
        ),
    )
    feature_loader: Pytorch3DUnetLoaderMetaConfig = Pytorch3DUnetLoaderMetaConfig(
        dataset="StandardHDF5Dataset",
        batch_size=32,
        num_workers=8,
        raw_internal_path="raw",
        label_internal_path="labels",
        global_normalization=True,
        # global_normalization=False,
        global_percentiles=None,
        file_paths=("/Rmito/test_converted.h5",),
        roi=[[0, 150], [0, 1280], [0, 1280]],
        transformer={
            "raw": [
                {"name": "Normalize"},
                {"name": "ToTensor", "expand_dims": True},
            ],
            "label": [
                {"name": "ToTensor", "expand_dims": True},
            ],
        },
        slice_builder=Pytorch3DUnetSliceBuilderConfig(
            name="SliceBuilder",
            patch_shape=(1, 256, 256),
            stride_shape=(1, 256, 256),
            halo_shape=(0, 0, 0),
        ),
    )
    # feature_indices_path: Optional[str] = (
    #     "/scratch/talks/sampled_features/semantic_segmentation/mitochondria/feature_indices/Rmito_indices.npz"
    # )
    feature_indices_path: Optional[str] = None
    predictor_semantic: Pytorch3DUnetPredictorMetaConfig = (
        Pytorch3DUnetPredictorMetaConfig(
            name="PatchWisePredictor",
            save_segmentation=False,
            min_size=None,
            layer_id=None,
            beta=0.5,
            zero_largest_instance=False,
            zero_large_instances=False,
            large_instance_multiplier=4,
            max_obj_size=None,
        )
    )
    predictor_instance: None = None
    eval_dataloader_semantic: EvalDataloaderMetaConfig = EvalDataloaderMetaConfig(
        name="StandardEvalDataset",
        gt_path=("/Rmito/test_converted.h5",),
        pred_key="predictions",
        gt_key="labels",
        patch_key="patch_index",
        roi=[[0, 150], [0, 1280], [0, 1280]],
        ignore_index=None,
        ignore_path=None,
        ignore_key=None,
        convert_to_boundary_label=False,
        convert_to_binary_label=False,
        min_object_size=None,
        gt_zero_largest_instance=False,
        gt_zero_large_instances=False,
        zero_large_instances=False,
        zero_largest_instance=False,
        largest_obj_multiplier=4,
        max_obj_size=None,
        batch_size=32,
        num_workers=8,
    )
    eval_dataloader_instance: None = None
    consis_dataloader_semantic: EvalDataloaderMetaConfig = EvalDataloaderMetaConfig(
        name="StandardEvalDataset",
        gt_path=None,
        pred_key="predictions",
        gt_key="predictions",
        patch_key="patch_index",
        roi=None,
        ignore_index=None,
        ignore_path=None,
        ignore_key=None,
        convert_to_boundary_label=False,
        convert_to_binary_label=False,
        min_object_size=None,
        gt_zero_largest_instance=False,
        gt_zero_large_instances=False,
        zero_large_instances=False,
        zero_largest_instance=False,
        largest_obj_multiplier=4,
        max_obj_size=None,
        batch_size=32,
        num_workers=8,
    )
    consis_dataloader_instance: None = None
    filter_results: ForegroundFilterConfig = ForegroundFilterConfig(
        name="ForegroundFilter",
        foreground_threshold=0.1,
        gt_dir_path="/Rmito/",
        gt_key="labels",
        roi=None,
        save_selection=True,
        overwrite=False,
    )


class VNCTargetConfig(TargetDatasetConfigBase, frozen=True):
    name: Literal["VNC"] = "VNC"
    loader: Pytorch3DUnetLoaderMetaConfig = Pytorch3DUnetLoaderMetaConfig(
        dataset="StandardHDF5Dataset",
        batch_size=32,
        num_workers=8,
        # raw_internal_path="raw",
        # label_internal_path="label",
        raw_internal_path="resized_raw",
        label_internal_path="resized_labels",
        # raw_internal_path="raw",
        # label_internal_path="labels",
        global_normalization=True,
        # global_normalization=False,
        global_percentiles=None,
        # file_paths=("/VNC/data_labeled_mito.h5",),
        file_paths=("/VNC/resized_pixels/source_mitoEM_true.h5",),
        # file_paths=("/VNC/resized_pixels/test.h5",),
        roi=None,
        transformer={
            "raw": [
                {"name": "Normalize"},
                {"name": "ToTensor", "expand_dims": True},
            ]
        },
        slice_builder=Pytorch3DUnetSliceBuilderConfig(
            name="SliceBuilder",
            patch_shape=(1, 256, 256),
            stride_shape=(1, 256, 256),
            halo_shape=(0, 32, 32),
            # halo_shape=(0, 0, 0),
        ),
    )
    train_loader: Pytorch3DUnetLoaderMetaConfig = Pytorch3DUnetLoaderMetaConfig(
        dataset="StandardHDF5Dataset",
        batch_size=32,
        num_workers=8,
        # raw_internal_path="raw",
        # label_internal_path="label",
        # raw_internal_path="resized_raw",
        # label_internal_path="resized_labels",
        raw_internal_path="raw",
        label_internal_path="labels",
        # global_normalization=True,
        global_normalization=False,
        global_percentiles=None,
        # file_paths=("/VNC/data_labeled_mito.h5",),
        # file_paths=("/VNC/resized_pixels/source_mitoEM_true.h5",),
        file_paths=("/VNC/resized_pixels/binary_label/train_converted.h5",),
        roi=None,
        transformer={
            "raw": [
                {"name": "Normalize"},
                {"name": "ToTensor", "expand_dims": True},
            ],
            "label": [
                {"name": "ToTensor", "expand_dims": True},
            ],
        },
        slice_builder=Pytorch3DUnetSliceBuilderConfig(
            name="SliceBuilder",
            patch_shape=(1, 256, 256),
            stride_shape=(1, 64, 64),
            halo_shape=(0, 0, 0),
        ),
    )
    feature_loader: Pytorch3DUnetLoaderMetaConfig = Pytorch3DUnetLoaderMetaConfig(
        dataset="StandardHDF5Dataset",
        batch_size=32,
        num_workers=8,
        raw_internal_path="resized_raw",
        label_internal_path="resized_labels",
        global_normalization=True,
        global_percentiles=None,
        file_paths=("/VNC/resized_pixels/source_mitoEM_true.h5",),
        roi=None,
        transformer={
            "raw": [
                {"name": "Normalize"},
                {"name": "ToTensor", "expand_dims": True},
            ],
            "label": [
                {"name": "Relabel"},
                {"name": "BlobsToMask"},
                {"name": "ToTensor", "expand_dims": True},
            ],
        },
        slice_builder=Pytorch3DUnetSliceBuilderConfig(
            name="SliceBuilder",
            patch_shape=(1, 256, 256),
            stride_shape=(1, 256, 256),
            halo_shape=(0, 0, 0),
        ),
    )
    # feature_indices_path: Optional[str] = (
    #     "/scratch/talks/sampled_features/semantic_segmentation/mitochondria/feature_indices/VNC_indices.npz"
    # )
    feature_indices_path: Optional[str] = None
    # feature_indices_path: Optional[str] = None
    predictor_semantic: Pytorch3DUnetPredictorMetaConfig = (
        Pytorch3DUnetPredictorMetaConfig(
            name="PatchWisePredictor",
            save_segmentation=False,
            min_size=None,
            layer_id=None,
            beta=0.5,
            zero_largest_instance=False,
            zero_large_instances=False,
            large_instance_multiplier=4,
            max_obj_size=None,
        )
    )
    predictor_instance: None = None
    eval_dataloader_semantic: EvalDataloaderMetaConfig = EvalDataloaderMetaConfig(
        name="StandardEvalDataset",
        # gt_path=("/VNC/data_labeled_mito.h5",),
        gt_path=("/VNC/resized_pixels/source_mitoEM_true.h5",),
        # gt_path=("/VNC/resized_pixels/test.h5",),
        pred_key="predictions",
        # gt_key="label",
        gt_key="resized_labels",
        # gt_key="labels",
        patch_key="patch_index",
        roi=None,
        ignore_index=None,
        ignore_path=None,
        ignore_key=None,
        convert_to_boundary_label=False,
        convert_to_binary_label=True,
        min_object_size=None,
        gt_zero_largest_instance=False,
        gt_zero_large_instances=False,
        zero_large_instances=False,
        zero_largest_instance=False,
        largest_obj_multiplier=4,
        max_obj_size=None,
        batch_size=32,
        num_workers=8,
    )
    eval_dataloader_instance: None = None
    consis_dataloader_semantic: EvalDataloaderMetaConfig = EvalDataloaderMetaConfig(
        name="StandardEvalDataset",
        gt_path=None,
        pred_key="predictions",
        gt_key="predictions",
        patch_key="patch_index",
        roi=None,
        ignore_index=None,
        ignore_path=None,
        ignore_key=None,
        convert_to_boundary_label=False,
        convert_to_binary_label=False,
        min_object_size=None,
        gt_zero_largest_instance=False,
        gt_zero_large_instances=False,
        zero_large_instances=False,
        zero_largest_instance=False,
        largest_obj_multiplier=4,
        max_obj_size=None,
        batch_size=32,
        num_workers=8,
    )
    consis_dataloader_instance: None = None
    filter_results: ForegroundFilterConfig = ForegroundFilterConfig(
        name="ForegroundFilter",
        foreground_threshold=0.02,
        gt_dir_path="/VNC/resized_pixels/",
        # gt_key="labels",
        gt_key="resized_labels",
        roi=None,
        save_selection=True,
        overwrite=False,
    )


class CovidIFTargetConfig(TargetDatasetConfigBase, frozen=True):
    name: Literal["Covid_IF"] = "Covid_IF"
    loader: TIFLoaderMetaConfig = TIFLoaderMetaConfig(
        dataset="Standard_TIF_Dataset",
        batch_size=5,
        num_workers=8,
        global_norm=False,
        percentiles=None,
        image_dir=("/g/kreshuk/talks/data/covid_if",),
        mask_dir=("/g/kreshuk/talks/data/covid_if",),
        transformer={
            "raw": [
                {"name": "Normalize"},
                {"name": "ToTensor", "expand_dims": True},
            ],
        },
    )
    predictor_semantic: TIFNucleiSemanticPredictorConfig = (
        TIFNucleiSemanticPredictorConfig(
            name="DSB2018Predictor",
        )
    )
    predictor_instance: TIFNucleiInstancePredictorConfig = (
        TIFNucleiInstancePredictorConfig(
            name="NucleiInstancePredictor",
            save_segmentation=True,
            min_size=50,
            zero_largest_instance=False,
            no_adjust_background=False,
        )
    )
    eval_dataloader_semantic: Eval_TIF_DataloaderMetaConfig = (
        Eval_TIF_DataloaderMetaConfig(
            batch_size=1,
            num_workers=8,
            name="Standard_TIF_Dataset",
            expand_dims=True,
            global_norm=False,
            percentiles=None,
            image_key="prediction",
            min_object_size=None,
            zero_large_instances=False,
            mask_dir=("/g/kreshuk/talks/data/covid_if",),
            mask_key="labels/cells/s0",
            transformer={
                "raw": [{"name": "ToTensor", "expand_dims": True}],
                "label": [
                    {"name": "Relabel"},
                    {"name": "BlobsToMask", "append_label": False},
                    {"name": "ToTensor", "expand_dims": True},
                ],
            },
        )
    )
    eval_dataloader_instance: Eval_TIF_DataloaderMetaConfig = (
        Eval_TIF_DataloaderMetaConfig(
            batch_size=1,
            num_workers=8,
            name="Standard_TIF_Dataset",
            expand_dims=True,
            global_norm=False,
            percentiles=None,
            image_key="prediction",
            min_object_size=50,
            zero_large_instances=False,
            mask_dir=("/g/kreshuk/talks/data/covid_if",),
            mask_key="labels/cells/s0",
            transformer={
                "raw": [{"name": "ToTensor", "expand_dims": True}],
                "label": [{"name": "ToTensor", "expand_dims": True}],
            },
        )
    )
    consis_dataloader_semantic: Eval_TIF_DataloaderMetaConfig = (
        Eval_TIF_DataloaderMetaConfig(
            batch_size=1,
            num_workers=8,
            name="Standard_TIF_Dataset",
            expand_dims=True,
            global_norm=False,
            percentiles=None,
            image_key="prediction",
            min_object_size=None,
            zero_large_instances=False,
            mask_dir=None,
            mask_key="prediction",
            transformer={
                "raw": [
                    {"name": "ToTensor", "expand_dims": True},
                ],
                "label": [
                    {"name": "ToTensor", "expand_dims": True},
                ],
            },
        )
    )
    consis_dataloader_instance: Eval_TIF_DataloaderMetaConfig = (
        Eval_TIF_DataloaderMetaConfig(
            batch_size=1,
            num_workers=8,
            name="Standard_TIF_Dataset",
            expand_dims=True,
            global_norm=False,
            percentiles=None,
            image_key="prediction",
            min_object_size=0,
            zero_large_instances=False,
            mask_dir=None,
            mask_key="prediction",
            transformer={
                "raw": [
                    {"name": "ToTensor", "expand_dims": True},
                ],
                "label": [
                    {"name": "ToTensor", "expand_dims": True},
                ],
            },
        )
    )
    filter_results: None = None


class SummaryResultsMetaConfig(BaseModel):
    overwrite_scores: bool
    consis_key: Optional[str] = None
    eval_key: Optional[str] = None
    save_name_postfix: str = ""
    filter_patches: bool = False


target_dataset_type = Annotated[
    Union[
        BBBC039TargetConfig,
        DSB2018TargetConfig,
        GoNuclearTargetConfig,
        HeLaNucTargetConfig,
        HoechstTargetConfig,
        SBIAD634TargetConfig,
        SBIAD895TargetConfig,
        SBIAD1196TargetConfig,
        SBIAD1410TargetConfig,
        FlyWingTargetConfig,
        OvulesTargetConfig,
        PNASTargetConfig,
        EPFLTargetConfig,
        HmitoTargetConfig,
        RmitoTargetConfig,
        VNCTargetConfig,
        CovidIFTargetConfig,
    ],
    Discriminator("name"),
]

mito_dataset_type = Annotated[
    Union[
        EPFLTargetConfig,
        HmitoTargetConfig,
        RmitoTargetConfig,
        VNCTargetConfig,
    ],
    Discriminator("name"),
]


class MetaConfig(BaseModel):
    target_datasets: Sequence[target_dataset_type]
    source_models: Sequence[ModelSourceConfig]
    segmentation_mode: Literal["instance", "semantic"]
    run_mode: Literal[
        "full",
        "evaluation",
        "consistency",
        "pred_eval",
        "summary_results",
        "adaptive_batchnorm",
    ]
    summary_results: SummaryResultsMetaConfig
    overwrite_yaml: bool
    data_base_path: str
    model_dir_path: str
    feature_perturbations: Optional[FeaturePerturbationConfig]
    output_settings: OutputSettingsConfig
    input_augs: Dict[str, List[Tuple[float, float]]]
    eval_settings: Optional[eval_metric_type]
    consistency_settings: Optional[consistency_metric_type]


class TransformerConsistencyMetaConfig(BaseModel):
    source_models: Sequence[SourceModelConfigBase]
    target_datasets: Sequence[target_dataset_type]
    data_base_path: str
    overwrite_yaml: bool
    input_augs: Dict[str, List[Tuple[float, float]]]
    summary_results: SummaryResultsMetaConfig
    output_settings: OutputSettingsConfig
    consistency_settings: Optional[consistency_metric_type]


class SummaryResultsConfig(BaseModel):
    filter_patches: Optional[ForegroundFilterConfig]
    output_path: str
    eval_key: Optional[str]
    consis_key: Optional[str]
    overwrite_scores: bool
    save_name_postfix: str = ""


class ConfigFull(BaseModel):
    wandb: WandbConfig
    model_path: str
    summary_results: SummaryResultsConfig
    model: Pytorch3DUnetModelConfig
    predictor: Annotated[
        Union[
            Pytorch3DUnetPredictorMetaConfig,
            TIFNucleiSemanticPredictorConfig,
        ],
        Discriminator("name"),
    ]
    loaders: Annotated[
        Union[
            Pytorch3DUnetLoaderMetaConfig,
            TIFLoaderMetaConfig,
            TIFtxtLoaderMetaConfig,
            SBIAD1410LoaderMetaConfig,
        ],
        Discriminator("dataset"),
    ]
    evaluation: EvaluateConfig
    consistency: ConsistencyConfig


class ConfigEvaluation(BaseModel):
    save_results: SummaryResultsConfig
    evaluation: EvaluateConfig


class ConfigConsistency(BaseModel):
    save_results: SummaryResultsConfig
    consistency: ConsistencyConfig


class SemanticSegmentationConfig(BaseModel):
    name: Literal["semantic"] = "semantic"


class InstanceSegmentationConfig(BaseModel):
    name: Literal["instance"] = "instance"
    min_size: int = 50
    beta: float = 0.5
    zero_largest_instance: bool = False
    zero_large_instances: bool = False
    large_instance_multiplier: int = 4


segmentation_type = Annotated[
    Union[SemanticSegmentationConfig, InstanceSegmentationConfig],
    Discriminator("name"),
]


class ConsistencyPseudoLabelerConfig(BaseModel):
    # consistency_metric: consistency_metric_type
    consistency_threshold: Optional[float]
    seg_params: segmentation_type
    consistency_metric: consistency_metric_type
    activation: Optional[Literal["softmax", "sigmoid"]] = "sigmoid"


class InputConsisPseudoLabelerConfig(ConsistencyPseudoLabelerConfig):
    name: Literal["input_consistency"]
    transformer_cfg: Dict[str, List[Any]]
    stats_cfg: Dict[str, Any]


class ModelConsisPseudoLabelerConfig(ConsistencyPseudoLabelerConfig):
    name: Literal["model_consistency"]
    perturbed_model_config: Pytorch3DUnetModelConfig


class DefaultPseudoLabelerConfig(BaseModel):
    name: Literal["default_pseudo_labeler"]
    confidence_threshold: Optional[float] = None
    threshold_from_both_sides: bool = True
    mask_channel: Optional[int] = None
    activation: Optional[Literal["softmax", "sigmoid"]] = "sigmoid"


class DummyDirectEvalPseudoLabelerConfig(BaseModel):
    name: Literal["direct_eval_pseudo_labeler"]
    score_threshold: float = 0.5
    activation: Optional[Literal["softmax", "sigmoid"]] = "sigmoid"


class ScheduledPseudoLabelerConfig(BaseModel):
    name: Literal["scheduled_pseudo_labeler"]
    confidence_threshold: Optional[float] = None
    threshold_from_both_sides: bool = True
    mode: Literal["min", "max"] = "min"
    factor: float = 0.05
    patience: int = 10
    threshold: float = 1e-4
    threshold_mode: Literal["rel", "abs"] = "abs"
    min_ct: float = 0.5
    eps: float = 1e-8
    verbose: bool = True
    activation: Optional[Literal["softmax", "sigmoid"]] = "sigmoid"


pseudo_labeler_type = Annotated[
    Union[
        InputConsisPseudoLabelerConfig,
        ModelConsisPseudoLabelerConfig,
        DefaultPseudoLabelerConfig,
        ScheduledPseudoLabelerConfig,
        DummyDirectEvalPseudoLabelerConfig,
    ],
    Discriminator("name"),
]


class SelfTrainingDataConfig(BaseModel):
    unsupervised_train_paths: List[str]
    unsupervised_val_paths: List[str]
    patch_shape: Tuple[int, ...]
    raw_key: str
    label_key: Optional[str]
    batch_size: int
    num_workers: int
    n_samples_train: Optional[int]
    n_samples_val: Optional[int]
    roi_unsupervised_train: Optional[Sequence[Sequence[int]]] = None
    roi_unsupervised_val: Optional[Sequence[Sequence[int]]] = None
    global_normalization: bool = False
    norm01: bool = False


class SelfTrainingModelConfig(BaseModel):
    model: Annotated[
        Union[Pytorch3DUnetModelConfig, UnetrModelConfig], Discriminator("name")
    ]
    source_checkpoint: Optional[Union[str, Path]]


class SelfTrainingTrainConfig(BaseModel):
    lr: float
    n_iterations: Optional[int]
    epochs: Optional[int]
    save_ckpt_every_kth_epoch: Optional[int]
    mixed_precision: bool = True
    scheduler_kwargs: Dict[str, Any] = DEFAULT_SCHEDULER_KWARGS
    optimizer_kwargs: Dict[str, Any] = {}


class MeanTeacherConfig(BaseModel):
    name: str
    output_root_path: str
    data_cfg: SelfTrainingDataConfig
    pseudo_labeler_cfg: pseudo_labeler_type
    model_cfg: SelfTrainingModelConfig
    training_cfg: SelfTrainingTrainConfig
    wandb_cfg: Optional[WandbConfig]
    supervised_loader_cfg: Optional[Dict[str, Any]] = None


class SupervisedDataConfig(BaseModel):
    patch_shape: Tuple[int, ...]
    supervised_train_paths: List[str]
    supervised_val_paths: List[str]
    raw_key: str
    label_key: str
    batch_size: int
    num_workers: int
    n_samples_train: Optional[int]
    n_samples_val: Optional[int]
    roi_supervised_train: Optional[Sequence[Sequence[int]]] = None
    roi_supervised_val: Optional[Sequence[Sequence[int]]] = None


class SupervisedFinetuningConfig(BaseModel):
    name: str
    output_root_path: str
    model_cfg: SelfTrainingModelConfig
    training_cfg: SelfTrainingTrainConfig
    wandb_cfg: Optional[WandbConfig]
    loader_cfg: Dict[str, Any]


class FeatureSampleConfig(BaseModel):
    layers: List[str]
    sampling_seed: int
    num_samples: int
    output_dir_path: Optional[str]


class TransferFeatureExtractionConfig(BaseModel):
    target_datasets: Sequence[mito_dataset_type]
    source_models: Sequence[ModelSourceConfig]
    source_model_base_path: str
    data_base_path: str
    feature_cfg: FeatureSampleConfig


class PrecomputedPerformanceConfig(BaseModel):
    base_path: str
    key: str
    invert_score: bool = False


class PrecomputedDirectPerformanceConfig(PrecomputedPerformanceConfig):
    name: Literal["direct_performance"]
    approach: Literal["consistency", "feature_perturbation_consistency"]
    run_id: str


class PrecomputedFinetunedPerformanceConfig(PrecomputedPerformanceConfig):
    name: Literal["finetuned_performance"]
    finetuning_approach: Literal[
        "confidence_threshold",
        "direct_eval",
        "feature_perturbation",
        "default_selftraining",
    ]
    result_type: Literal["predictions", "checkpoints"]


class PrecomputedClassificationPerformanceConfig(PrecomputedPerformanceConfig):
    name: Literal["classification_performance"]


performance_type = Union[
    PrecomputedDirectPerformanceConfig,
    PrecomputedFinetunedPerformanceConfig,
    PrecomputedClassificationPerformanceConfig,
]

segmentation_performance_type = Union[
    PrecomputedDirectPerformanceConfig,
    PrecomputedFinetunedPerformanceConfig,
]


class PrecomputedFeatureConfig(BaseModel):
    base_path: str
    file_type: str
    layer_keys: Dict[str, str]
    n_PCA_components: Optional[int] = None


class TransferabilitySaveConfig(BaseModel):
    save_base_path: Optional[str]
    save_name: Optional[str]
    save_plot: bool


class TransferabilityMetricConfig(BaseModel):
    targets: Sequence[str]
    source_models: Dict[str, str]
    feature_config: PrecomputedFeatureConfig
    performance_config: Annotated[performance_type, Discriminator("name")]
    transferability_metrics: Sequence[transferability_metric_names]
    output_config: TransferabilitySaveConfig


class FSA_SaveConfig(BaseModel):
    save_base_path: str
    save_name: str


class FeatureSpaceAnalysisConfig(BaseModel):
    targets: Sequence[str]
    source_models: Sequence[str]
    feature_config: PrecomputedFeatureConfig
    output_config: FSA_SaveConfig
    max_samples: int


class LoadTransformerPredictionsConfig(BaseModel):
    model_name: str
    TTA_key: str
    base_dir_path: str
    file_identifier: str


class ConsistencyPatchedTransformerConfig(BaseModel):
    predictions_perturbed: LoadTransformerPredictionsConfig
    predictions_unperturbed: LoadTransformerPredictionsConfig
    metric_config: AdaptedRandErrorConsisConfig
    summary: SummaryResultsConfig
