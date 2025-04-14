from pydantic import BaseModel, Discriminator
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
    save_key: str
    save_mask: bool
    mask_threshold: float
    overwrite_score: bool


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
    relabel_background: bool
    min_object_size: Optional[int]
    instance_zero_background: bool
    zero_largest_instance: bool


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
    transformer: Mapping[
        str, List[Mapping[str, Optional[Union[str, bool, int, Sequence[int]]]]]
    ]


class TIFtxtPhaseConfig(BaseModel, frozen=True):
    # dataset_name: Literal["TIF_txt_Dataset"]
    image_dir: Sequence[str]
    mask_dir: Sequence[str]
    filenames_path: str
    transformer: Mapping[
        str, List[Mapping[str, Optional[Union[str, bool, int, Sequence[int]]]]]
    ]


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
    instance_zero_background: bool


class SBIAD1410PhaseConfig(BaseModel):
    img_paths: Sequence[str]
    mask_paths: Sequence[str]
    roi: Optional[Sequence[Sequence[int]]]
    transformer: Mapping[
        str, List[Mapping[str, Optional[Union[str, bool, int, Sequence[int]]]]]
    ]
    slice_builder: Optional[
        Union[Pytorch3DUnetFilterSliceBuilderConfig, Pytorch3DUnetSliceBuilderConfig]
    ]


class SBIAD1410PhaseMetaConfig(BaseModel):
    mask_paths: Optional[Sequence[str]]
    roi: Optional[Sequence[Sequence[int]]]
    transformer: Mapping[
        str, List[Mapping[str, Optional[Union[str, bool, int, Sequence[int]]]]]
    ]
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
    instance_zero_background: bool = True


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
    relabel_background: bool
    instance_zero_background: bool
    zero_largest_instance: bool
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
                relabel_background=self.relabel_background,
                instance_zero_background=self.instance_zero_background,
                zero_largest_instance=self.zero_largest_instance,
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
                relabel_background=self.relabel_background,
                instance_zero_background=self.instance_zero_background,
                zero_largest_instance=self.zero_largest_instance,
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
    instance_zero_background: bool
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
                instance_zero_background=self.instance_zero_background,
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
                instance_zero_background=self.instance_zero_background,
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

    def initialise_metric(self, dataset_name: str) -> AdaptedRandErrorEval:
        return AdaptedRandErrorEval(
            dataset_name=dataset_name,
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
        )

    def initialise_score(self, num_samples: int) -> NDArray[Any]:
        return np.zeros(num_samples, dtype=np.float32)


class EvaluateConfig(BaseModel, frozen=True):
    eval_dataloader: EvalDataloaderConfig
    eval_metric: Annotated[
        Union[
            MultiClassF1Config,
            BinaryF1Config,
            SoftF1Config,
            AdaptedRandErrorEvalConfig,
            MeanAvgPrecisionConfig,
        ],
        Discriminator("name"),
    ]


class ConsistencyConfig(BaseModel, frozen=True):
    consistency_dataloader: EvalDataloaderConfig
    consistency_metric: Annotated[
        Union[
            DifferenceImageConfig,
            EffectiveInvarianceConfig,
            EntropyConfig,
            KLDivergenceConfig,
            CrossEntropyConfig,
            HammingDistanceConfig,
            AdaptedRandErrorConsisConfig,
        ],
        Discriminator("name"),
    ]


class WandbConfig(BaseModel):
    project: str
    name: str
    mode: Literal["disabled", "online", "offline"]


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
    zero_largest_instance: bool
    no_adjust_background: bool


class Pytorch3DUnetPredictorConfig(Pytorch3DUnetPredictorMetaConfig):
    save_suffix: str
    output_file_name: Optional[str]


class Pytorch3DUnetDatasetConfig(BaseModel, frozen=True):
    file_paths: Sequence[str]
    slice_builder: Annotated[
        Union[Pytorch3DUnetSliceBuilderConfig, Pytorch3DUnetFilterSliceBuilderConfig],
        Discriminator("name"),
    ]
    transformer: Mapping[
        str, List[Mapping[str, Optional[Union[str, bool, int, Sequence[int]]]]]
    ]
    roi: Optional[Sequence[Sequence[int]]]


class Pytorch3DUnetLoaderConfig(BaseModel, frozen=True):
    dataset: Literal["StandardHDF5Dataset"]
    output_dir: str
    batch_size: int
    num_workers: int
    raw_internal_path: str
    label_internal_path: str
    global_normalization: bool
    global_percentiles: Optional[Sequence[float]]
    test: Pytorch3DUnetDatasetConfig


class SBIAD1410LoaderConfig(BaseModel, frozen=True):
    dataset: Literal["S_BIAD1410_Dataset"]
    output_dir: str
    batch_size: int
    num_workers: int
    global_normalization: bool
    global_percentiles: Optional[Sequence[Union[float, int]]]
    test: SBIAD1410PhaseConfig


class SBIAD1410LoaderMetaConfig(BaseModel, frozen=True):
    dataset: Literal["S_BIAD1410_Dataset"]
    batch_size: int
    num_workers: int
    global_normalization: bool
    global_percentiles: Optional[Sequence[Union[float, int]]]
    img_paths: Sequence[str]
    mask_paths: Sequence[str]
    roi: Optional[Sequence[Sequence[int]]]
    transformer: Mapping[
        str, List[Mapping[str, Optional[Union[str, bool, int, Sequence[int]]]]]
    ]
    slice_builder: Optional[
        Union[Pytorch3DUnetFilterSliceBuilderConfig, Pytorch3DUnetSliceBuilderConfig]
    ]

    def create_config(self, output_dir: str, data_base_path: str):
        img_paths: List[str] = []
        mask_paths: List[str] = []
        for i in range(len(self.img_paths)):
            img_paths.append(data_base_path + self.img_paths[i])
            mask_paths.append(data_base_path + self.mask_paths[i])
        return SBIAD1410LoaderConfig(
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
    transformer: Mapping[
        str, List[Mapping[str, Optional[Union[str, bool, int, Sequence[int]]]]]
    ]
    roi: Optional[Sequence[Sequence[int]]]

    def create_config(self, output_dir: str, data_base_path: str):
        file_paths: List[str] = []
        for i in range(len(self.file_paths)):
            file_paths.append(data_base_path + self.file_paths[i])
        return Pytorch3DUnetLoaderConfig(
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


class TIFNucleiSemanticPredictorConfig(BaseModel):
    name: Literal["DSB2018Predictor"]


class TIFNucleiInstancePredictorConfig(BaseModel):
    name: Literal["NucleiInstancePredictor"]
    save_segmentation: bool
    min_size: int
    zero_largest_instance: bool = False
    no_adjust_background: bool = False


class TIFPredictionLoadersConfig(BaseModel, frozen=True):
    dataset: Literal[
        "Standard_TIF_Dataset", "TIF_txt_Dataset", "HeLaNuc_Dataset", "Hoechst_Dataset"
    ]
    output_dir: str
    batch_size: int
    num_workers: int
    global_norm: bool
    percentiles: Optional[Sequence[Union[float, int]]]
    # test: Annotated[
    #    Union[TIFPhaseConfig, TIFtxtPhaseConfig], Discriminator("dataset_name")
    # ]
    test: Union[TIFPhaseConfig, TIFtxtPhaseConfig]


class Pytorch3DUnetModelMetaConfig(BaseModel, frozen=True):
    name: str
    in_channels: int
    out_channels: int
    layer_order: str
    f_maps: Union[int, Sequence[int]]
    final_sigmoid: bool
    feature_return: bool
    is_segmentation: Optional[bool]


class Pytorch3DUnetModelConfig(Pytorch3DUnetModelMetaConfig, frozen=True):
    # architecture: Pytorch3DUnetModelMetaConfig
    feature_perturbation: Annotated[
        Optional[
            Union[
                DropOutPerturbationConfig,
                FeatureDropPerturbationConfig,
                FeatureNoisePerturbationConfig,
            ]
        ],
        Discriminator("name"),
    ]


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
    result_dir: str
    approach: str
    base_dir_path: str


class LoaderMetaConfig(BaseModel):
    batch_size: int
    num_workers: int
    global_norm: bool
    percentiles: Optional[Sequence[float]]
    image_dir: Sequence[str]
    mask_dir: Sequence[str]
    transformer: Mapping[
        str, List[Mapping[str, Optional[Union[str, bool, int, Sequence[int]]]]]
    ]


class TIFLoaderMetaConfig(LoaderMetaConfig):
    dataset: Literal["Standard_TIF_Dataset", "HeLaNuc_Dataset", "Hoechst_Dataset"]

    def create_config(self, output_dir: str, data_base_path: str):
        mask_dir: List[str] = []
        image_dir: List[str] = []
        for i in range(len(self.mask_dir)):
            mask_dir.append(data_base_path + self.mask_dir[i])
            image_dir.append(data_base_path + self.image_dir[i])
        return TIFPredictionLoadersConfig(
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


class TIFtxtLoaderMetaConfig(LoaderMetaConfig):
    dataset: Literal["TIF_txt_Dataset"]
    filenames_path: str

    def create_config(self, output_dir: str, data_base_path: str):
        mask_dir: List[str] = []
        image_dir: List[str] = []
        for i in range(len(self.mask_dir)):
            mask_dir.append(data_base_path + self.mask_dir[i])
            image_dir.append(data_base_path + self.image_dir[i])
        return TIFPredictionLoadersConfig(
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


class Eval_TIF_TxtDataloaderMetaConfig(BaseModel, frozen=True):
    name: Literal["TIF_txt_Dataset"]
    expand_dims: bool
    global_norm: bool
    percentiles: Optional[Sequence[Union[float, int]]]
    image_key: Optional[str]
    min_object_size: Optional[int]
    instance_zero_background: bool
    mask_dir: Optional[Sequence[str]]
    mask_key: Optional[str]
    filenames_path: str
    batch_size: int
    num_workers: int
    transformer: Mapping[
        str, List[Mapping[str, Optional[Union[str, bool, int, Sequence[int]]]]]
    ]

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
                instance_zero_background=self.instance_zero_background,
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
                instance_zero_background=self.instance_zero_background,
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
    instance_zero_background: bool
    mask_dir: Optional[Sequence[str]]
    mask_key: Optional[str]
    batch_size: int
    num_workers: int
    transformer: Mapping[
        str, List[Mapping[str, Optional[Union[str, bool, int, Sequence[int]]]]]
    ]

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
                instance_zero_background=self.instance_zero_background,
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
                instance_zero_background=self.instance_zero_background,
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


class SourceModelConfigBase(BaseModel, frozen=True):
    model: Pytorch3DUnetModelMetaConfig
    model_name: str


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


class Model3LayerSourceConfig(SourceModelConfigBase, frozen=True):
    source_name: Literal[
        "BBBC039", "DSB2018", "HeLaNuc", "Hoechst", "S_BIAD634", "S_BIAD895"
    ]
    model: Pytorch3DUnetModelMetaConfig = UNET2D_3LAYER_ARCHITECTURE
    model_name: str

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
        return Pytorch3DUnetModelConfig(
            name=self.model.name,
            in_channels=self.model.in_channels,
            out_channels=self.model.out_channels,
            layer_order=self.model.layer_order,
            f_maps=self.model.f_maps,
            final_sigmoid=self.model.final_sigmoid,
            feature_return=self.model.feature_return,
            is_segmentation=self.model.is_segmentation,
            feature_perturbation=feature_perturbation,
        )


class Model4LayerSourceConfig(SourceModelConfigBase, frozen=True):
    source_name: Literal[
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
    ]
    model: Pytorch3DUnetModelMetaConfig = UNET2D_4LAYER_ARCHITECTURE
    model_name: str

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
        return Pytorch3DUnetModelConfig(
            name=self.model.name,
            in_channels=self.model.in_channels,
            out_channels=self.model.out_channels,
            layer_order=self.model.layer_order,
            f_maps=self.model.f_maps,
            final_sigmoid=self.model.final_sigmoid,
            feature_return=self.model.feature_return,
            is_segmentation=self.model.is_segmentation,
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
            batch_size=1,
            num_workers=8,
            name="TIF_txt_Dataset",
            expand_dims=True,
            global_norm=False,
            percentiles=None,
            image_key="predictions",
            min_object_size=None,
            instance_zero_background=False,
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
            batch_size=1,
            num_workers=8,
            name="TIF_txt_Dataset",
            expand_dims=True,
            global_norm=False,
            percentiles=None,
            image_key="segmentation",
            min_object_size=50,
            instance_zero_background=False,
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
            batch_size=1,
            num_workers=8,
            name="TIF_txt_Dataset",
            expand_dims=True,
            global_norm=False,
            percentiles=None,
            image_key="predictions",
            min_object_size=None,
            instance_zero_background=False,
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
            batch_size=1,
            num_workers=8,
            name="TIF_txt_Dataset",
            expand_dims=True,
            global_norm=False,
            percentiles=None,
            image_key="segmentation",
            min_object_size=50,
            instance_zero_background=False,
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
            instance_zero_background=False,
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
            instance_zero_background=False,
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
            instance_zero_background=False,
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
            instance_zero_background=False,
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
            instance_zero_background=False,
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
            instance_zero_background=False,
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
            instance_zero_background=False,
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
            instance_zero_background=False,
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
            instance_zero_background=False,
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
            instance_zero_background=False,
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
            instance_zero_background=False,
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
            instance_zero_background=False,
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
            instance_zero_background=False,
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
            instance_zero_background=False,
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
            instance_zero_background=False,
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
            instance_zero_background=False,
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
            zero_largest_instance=False,
            no_adjust_background=False,
        )
    )
    predictor_instance: Pytorch3DUnetPredictorMetaConfig = (
        Pytorch3DUnetPredictorMetaConfig(
            name="PatchWisePredictor",
            save_segmentation=True,
            min_size=1,
            layer_id=None,
            zero_largest_instance=False,
            no_adjust_background=False,
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
        relabel_background=False,
        instance_zero_background=False,
        zero_largest_instance=False,
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
        relabel_background=False,
        instance_zero_background=False,
        zero_largest_instance=False,
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
        relabel_background=False,
        instance_zero_background=False,
        zero_largest_instance=False,
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
        relabel_background=False,
        instance_zero_background=False,
        zero_largest_instance=False,
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
            zero_largest_instance=False,
            no_adjust_background=False,
        )
    )
    predictor_instance: Pytorch3DUnetPredictorMetaConfig = (
        Pytorch3DUnetPredictorMetaConfig(
            name="PatchWisePredictor",
            save_segmentation=True,
            min_size=1,
            layer_id=None,
            zero_largest_instance=False,
            no_adjust_background=False,
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
            instance_zero_background=False,
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
            instance_zero_background=False,
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
            instance_zero_background=False,
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
            instance_zero_background=False,
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
            instance_zero_background=False,
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
            instance_zero_background=False,
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
            instance_zero_background=False,
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
            instance_zero_background=False,
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
            zero_largest_instance=False,
            no_adjust_background=False,
        )
    )
    predictor_instance: Pytorch3DUnetPredictorMetaConfig = (
        Pytorch3DUnetPredictorMetaConfig(
            name="PatchWisePredictor",
            save_segmentation=True,
            min_size=50,
            layer_id=None,
            zero_largest_instance=False,
            no_adjust_background=False,
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
        relabel_background=False,
        instance_zero_background=False,
        zero_largest_instance=False,
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
        relabel_background=False,
        instance_zero_background=False,
        zero_largest_instance=False,
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
        relabel_background=False,
        instance_zero_background=False,
        zero_largest_instance=False,
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
        relabel_background=False,
        instance_zero_background=False,
        zero_largest_instance=False,
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
        label_internal_path="volumes/labels/cells_with_ignore",
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
        ),
    )
    predictor_semantic: None = None
    predictor_instance: Pytorch3DUnetPredictorMetaConfig = (
        Pytorch3DUnetPredictorMetaConfig(
            name="PatchWisePredictor",
            save_segmentation=True,
            min_size=50,
            layer_id=None,
            zero_largest_instance=False,
            no_adjust_background=True,
        )
    )
    eval_dataloader_semantic: None = None
    eval_dataloader_instance: EvalDataloaderMetaConfig = EvalDataloaderMetaConfig(
        name="StandardEvalDataset",
        gt_path=("/FlyWing/GT/test/per03.h5",),
        pred_key="segmentation",
        gt_key="volumes/labels/cells_with_ignore",
        patch_key="patch_index",
        roi=None,
        ignore_index=-1,
        ignore_path=None,
        ignore_key=None,
        convert_to_boundary_label=False,
        convert_to_binary_label=False,
        min_object_size=50,
        relabel_background=False,
        instance_zero_background=False,
        zero_largest_instance=True,
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
        relabel_background=False,
        instance_zero_background=False,
        zero_largest_instance=True,
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
        ),
    )
    predictor_semantic: None = None
    predictor_instance: Pytorch3DUnetPredictorMetaConfig = (
        Pytorch3DUnetPredictorMetaConfig(
            name="PatchWisePredictor",
            save_segmentation=True,
            min_size=50,
            layer_id=None,
            zero_largest_instance=False,
            no_adjust_background=True,
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
        min_object_size=50,
        relabel_background=False,
        instance_zero_background=False,
        zero_largest_instance=True,
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
        min_object_size=50,
        relabel_background=False,
        instance_zero_background=False,
        zero_largest_instance=True,
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
        file_paths=("/PNAS/test/12hrs_plant1_trim-acylYFP.h5",),
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
        ),
    )
    predictor_semantic: None = None
    predictor_instance: Pytorch3DUnetPredictorMetaConfig = (
        Pytorch3DUnetPredictorMetaConfig(
            name="PatchWisePredictor",
            save_segmentation=True,
            min_size=50,
            layer_id=None,
            zero_largest_instance=True,
            no_adjust_background=False,
        )
    )
    eval_dataloader_semantic: None = None
    eval_dataloader_instance: EvalDataloaderMetaConfig = EvalDataloaderMetaConfig(
        name="StandardEvalDataset",
        gt_path=("/PNAS/test/12hrs_plant1_trim-acylYFP.h5",),
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
        relabel_background=True,
        instance_zero_background=False,
        zero_largest_instance=False,
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
        relabel_background=False,
        instance_zero_background=False,
        zero_largest_instance=False,
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
        ),
    )
    predictor_semantic: Pytorch3DUnetPredictorMetaConfig = (
        Pytorch3DUnetPredictorMetaConfig(
            name="PatchWisePredictor",
            save_segmentation=False,
            min_size=None,
            layer_id=None,
            zero_largest_instance=False,
            no_adjust_background=False,
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
        relabel_background=False,
        instance_zero_background=False,
        zero_largest_instance=False,
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
        relabel_background=False,
        instance_zero_background=False,
        zero_largest_instance=False,
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
        ),
    )
    predictor_semantic: Pytorch3DUnetPredictorMetaConfig = (
        Pytorch3DUnetPredictorMetaConfig(
            name="PatchWisePredictor",
            save_segmentation=False,
            min_size=None,
            layer_id=None,
            zero_largest_instance=False,
            no_adjust_background=False,
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
        relabel_background=False,
        instance_zero_background=False,
        zero_largest_instance=False,
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
        relabel_background=False,
        instance_zero_background=False,
        zero_largest_instance=False,
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
        ),
    )
    predictor_semantic: Pytorch3DUnetPredictorMetaConfig = (
        Pytorch3DUnetPredictorMetaConfig(
            name="PatchWisePredictor",
            save_segmentation=False,
            min_size=None,
            layer_id=None,
            zero_largest_instance=False,
            no_adjust_background=False,
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
        relabel_background=False,
        instance_zero_background=False,
        zero_largest_instance=False,
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
        relabel_background=False,
        instance_zero_background=False,
        zero_largest_instance=False,
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
        raw_internal_path="raw",
        label_internal_path="label",
        global_normalization=True,
        global_percentiles=None,
        file_paths=("/VNC/data_labeled_mito.h5",),
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
        ),
    )
    predictor_semantic: Pytorch3DUnetPredictorMetaConfig = (
        Pytorch3DUnetPredictorMetaConfig(
            name="PatchWisePredictor",
            save_segmentation=False,
            min_size=None,
            layer_id=None,
            zero_largest_instance=False,
            no_adjust_background=False,
        )
    )
    predictor_instance: None = None
    eval_dataloader_semantic: EvalDataloaderMetaConfig = EvalDataloaderMetaConfig(
        name="StandardEvalDataset",
        gt_path=("/VNC/data_labeled_mito.h5",),
        pred_key="predictions",
        gt_key="label",
        patch_key="patch_index",
        roi=None,
        ignore_index=None,
        ignore_path=None,
        ignore_key=None,
        convert_to_boundary_label=False,
        convert_to_binary_label=True,
        min_object_size=None,
        relabel_background=False,
        instance_zero_background=False,
        zero_largest_instance=False,
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
        convert_to_binary_label=True,
        min_object_size=None,
        relabel_background=False,
        instance_zero_background=False,
        zero_largest_instance=False,
        batch_size=32,
        num_workers=8,
    )
    consis_dataloader_instance: None = None
    filter_results: ForegroundFilterConfig = ForegroundFilterConfig(
        name="ForegroundFilter",
        foreground_threshold=0.02,
        gt_dir_path="/VNC/",
        gt_key="labels",
        roi=None,
        save_selection=True,
        overwrite=False,
    )


class SummaryResultsMetaConfig(BaseModel):
    overwrite_scores: bool


class MetaConfig(BaseModel):
    target_datasets: Sequence[
        Annotated[
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
            ],
            Discriminator("name"),
        ]
    ]
    source_models: Sequence[
        Annotated[
            Union[Model3LayerSourceConfig, Model4LayerSourceConfig],
            Discriminator("source_name"),
        ]
    ]
    segmentation_mode: Literal["instance", "semantic"]
    run_mode: Literal["full", "evaluation", "consistency"]
    summary_results: SummaryResultsMetaConfig
    overwrite_yaml: bool
    data_base_path: str
    model_dir_path: str
    feature_perturbations: FeaturePerturbationConfig
    output_settings: OutputSettingsConfig
    input_augs: Dict[str, List[Tuple[float, float]]]
    eval_settings: Optional[
        Annotated[
            Union[
                AdaptedRandErrorEvalConfig,
                MeanAvgPrecisionConfig,
                MultiClassF1Config,
                BinaryF1Config,
                SoftF1Config,
            ],
            Discriminator("name"),
        ]
    ]
    consistency_settings: Optional[
        Annotated[
            Union[
                DifferenceImageConfig,
                EffectiveInvarianceConfig,
                EntropyConfig,
                KLDivergenceConfig,
                CrossEntropyConfig,
                HammingDistanceConfig,
                AdaptedRandErrorConsisConfig,
            ],
            Discriminator("name"),
        ]
    ]


class SummaryResultsConfig(BaseModel):
    filter_patches: Optional[ForegroundFilterConfig]
    output_path: str
    eval_key: Optional[str]
    consis_key: Optional[str]
    overwrite_scores: bool


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
