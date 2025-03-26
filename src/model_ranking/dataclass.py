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


class ConsistencyMetricConfig(BaseModel):
    metric: str
    threshold: Optional[List[float]]
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
    # dataset_name: Literal["Standard_TIF_Dataset", "HelaNuc_Dataset", "Hoechst_Dataset"]
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
        "TIF_txt_Dataset", "Standard_TIF_Dataset", "HelaNuc_Dataset", "Hoechst_Dataset"
    ]
    eval: Union[TIFPhaseConfig, TIFtxtPhaseConfig]
    expand_dims: bool
    global_norm: bool
    percentiles: Optional[Sequence[Union[float, int]]]
    image_key: Optional[str]
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
    mask_paths: Sequence[str]
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
    gt_path: Sequence[str]
    pred_key: str
    gt_key: str
    patch_key: str
    roi: Optional[Sequence[Sequence[int]]]
    ignore_index: Optional[int]
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
        for i in range(len(self.gt_path)):
            gt_path.append(data_base_path + self.gt_path[i])
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
    instance_zero_background: bool
    batch_size: int
    num_workers: int

    def create_config(self, img_paths: Sequence[str], data_base_path: str):
        mask_paths: List[str] = []
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
                instance_zero_background=self.instance_zero_background,
            ),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )


class EvalMetricConfig(BaseModel, frozen=True):
    name: Literal[
        "BinaryF1",
        "MultiClassF1",
        "SoftF1",
        "RandError",
        "AdaptedRandError",
        "MeanAvgPrecision",
    ]
    threshold: Optional[float]
    eval_parameters: Optional[Dict[str, Any]]


class EvaluateConfig(BaseModel, frozen=True):
    eval_dataloader: EvalDataloaderConfig
    eval_metric: EvalMetricConfig
    eval_save_key: str
    consistency: ConsistencyMetricConfig


class WandbConfig(BaseModel):
    project: str
    name: str
    mode: Literal["disabled", "online", "offline"]


class DropOutPerturbationConfig(BaseModel):
    name: Literal["DropOutPerturbation"]
    random_seed: int
    layers: Sequence[int]
    drop_rate: float
    spatial_dropout: bool


class FeatureDropPerturbationConfig(BaseModel):
    name: Literal["FeatureDropPerturbation"]
    random_seed: int
    layers: Sequence[int]
    lower_th: float
    upper_th: float


class FeatureNoisePerturbationConfig(BaseModel):
    name: Literal["FeatureNoisePerturbation"]
    random_seed: int
    layers: Sequence[int]
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
    dataset: Literal["StandardHDF5Dataset", "S_BIAD1410_Dataset"]
    output_dir: str
    batch_size: int
    num_workers: int
    raw_internal_path: str
    label_internal_path: str
    global_normalization: bool
    global_percentiles: Optional[Sequence[float]]
    test: Pytorch3DUnetDatasetConfig


class Pytorch3DUnetLoaderMetaConfig(BaseModel, frozen=True):
    dataset: Literal["StandardHDF5Dataset", "S_BIAD1410_Dataset"]
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
        "Standard_TIF_Dataset", "TIF_txt_Dataset", "HelaNuc_Dataset", "Hoechst_Dataset"
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
    dataset: Literal["Standard_TIF_Dataset", "HelaNuc_Dataset", "Hoechst_Dataset"]

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
    mask_dir: Sequence[str]
    filenames_path: str
    batch_size: int
    num_workers: int
    transformer: Mapping[
        str, List[Mapping[str, Optional[Union[str, bool, int, Sequence[int]]]]]
    ]

    def create_config(self, image_dir: Sequence[str], data_base_path: str):
        mask_dir: List[str] = []
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
                min_object_size=self.min_object_size,
                instance_zero_background=self.instance_zero_background,
            ),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )


class Eval_TIF_DataloaderMetaConfig(BaseModel, frozen=True):
    name: Literal["Standard_TIF_Dataset", "HelaNuc_Dataset", "Hoechst_Dataset"]
    expand_dims: bool
    global_norm: bool
    percentiles: Optional[Sequence[float]]
    image_key: Optional[str]
    min_object_size: Optional[int]
    instance_zero_background: bool
    mask_dir: Sequence[str]
    batch_size: int
    num_workers: int
    transformer: Mapping[
        str, List[Mapping[str, Optional[Union[str, bool, int, Sequence[int]]]]]
    ]

    def create_config(self, image_dir: Sequence[str], data_base_path: str):
        mask_dir: List[str] = []
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


class TargetDatasetConfigBase(BaseModel, frozen=True):
    name: Literal["BBBC039", "DSB2018", "Go-Nuclear"]
    loader: Annotated[
        Union[
            Pytorch3DUnetLoaderMetaConfig, TIFLoaderMetaConfig, TIFtxtLoaderMetaConfig
        ],
        Discriminator("dataset"),
    ]
    predictor_semantic: Annotated[
        Union[
            Pytorch3DUnetPredictorMetaConfig,
            TIFNucleiSemanticPredictorConfig,
        ],
        Discriminator("name"),
    ]
    predictor_instance: Annotated[
        Union[
            Pytorch3DUnetPredictorMetaConfig,
            TIFNucleiInstancePredictorConfig,
        ],
        Discriminator("name"),
    ]
    eval_dataloader_instance: Annotated[
        Union[
            Eval_TIF_TxtDataloaderMetaConfig,
            Eval_TIF_DataloaderMetaConfig,
            EvalDataloaderMetaConfig,
            EvalSB1410DataloaderMetaConfig,
        ],
        Discriminator("name"),
    ]
    eval_dataloader_semantic: Annotated[
        Union[
            Eval_TIF_TxtDataloaderMetaConfig,
            Eval_TIF_DataloaderMetaConfig,
            EvalDataloaderMetaConfig,
            EvalSB1410DataloaderMetaConfig,
        ],
        Discriminator("name"),
    ]
    consistency: ConsistencyMetricMetaConfig

    class Config:
        arbitrary_types_allowed = True


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


class BBBC039TargetConfig(TargetDatasetConfigBase, frozen=True):
    name: Literal["BBBC039"] = "BBBC039"
    # name = "BBBC039"
    loader: TIFtxtLoaderMetaConfig = TIFtxtLoaderMetaConfig(
        batch_size=32,
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
            image_key="segmentation",
            min_object_size=None,
            instance_zero_background=False,
            mask_dir=("/BBBC039/instance_annotations/instance_labels",),
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
    consistency: ConsistencyMetricMetaConfig = ConsistencyMetricMetaConfig(
        save_mask=True,
        ignore_path=None,
        ignore_key=None,
        remove_background=False,
        zero_largest_instance=False,
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
            percentiles=(5, 98),
            image_key="segmentation",
            min_object_size=None,
            instance_zero_background=False,
            mask_dir=("/dsb2018_fluorescence/test/masks",),
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
            percentiles=(5, 98),
            image_key="segmentation",
            min_object_size=None,
            instance_zero_background=False,
            mask_dir=("/dsb2018_fluorescence/test/masks",),
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
    consistency: ConsistencyMetricMetaConfig = ConsistencyMetricMetaConfig(
        save_mask=True,
        ignore_path=None,
        ignore_key=None,
        remove_background=False,
        zero_largest_instance=False,
    )


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
        pred_key="segmentation",
        gt_key="label/gold",
        patch_key="patch_index",
        roi=[[50, 170]],
        ignore_index=None,
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
        convert_to_boundary_label=False,
        convert_to_binary_label=False,
        min_object_size=50,
        relabel_background=False,
        instance_zero_background=False,
        zero_largest_instance=False,
        batch_size=1,
        num_workers=8,
    )
    consistency: ConsistencyMetricMetaConfig = ConsistencyMetricMetaConfig(
        save_mask=True,
        ignore_path=None,
        ignore_key=None,
        remove_background=False,
        zero_largest_instance=False,
    )


class ConsistencyMetaConfig(BaseModel):
    metric: str
    threshold: Optional[List[float]]
    save_key: str
    pred_key: str


class AdaptedRandErrorConfig(EvalMetricConfig, frozen=True):
    name: Literal["AdaptedRandError"] = "AdaptedRandError"
    threshold: Optional[float] = None
    eval_parameters: Dict[str, Any] = {
        "num_dilations": 1,
        "num_erosions": 1,
    }


class MeanAvgPrecisionConfig(EvalMetricConfig, frozen=True):
    name: Literal["MeanAvgPrecision"] = "MeanAvgPrecision"
    threshold: Optional[float] = None
    eval_parameters: Dict[str, Any] = {
        "iou_range": [0.5, 0.95, 10],
        "min_instance_size": None,
    }


class MetaConfig(BaseModel):
    target_datasets: Sequence[
        Annotated[
            Union[BBBC039TargetConfig, DSB2018TargetConfig, GoNuclearTargetConfig],
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
    overwrite_yaml: bool
    data_base_path: str
    model_names: Dict[str, str]
    feature_perturbations: FeaturePerturbationConfig
    output_settings: OutputSettingsConfig
    # percentile_ranges: Dict[str, Optional[List[float]]]
    input_augs: Dict[str, List[Tuple[float, float]]]
    eval_settings: Annotated[
        Union[AdaptedRandErrorConfig, MeanAvgPrecisionConfig], Discriminator("name")
    ]
    eval_save_key: str
    consistency_settings: ConsistencyMetricConfig
