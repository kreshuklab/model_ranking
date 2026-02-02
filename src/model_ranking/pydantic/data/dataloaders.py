from pydantic import BaseModel, Discriminator
from typing import Annotated, assert_never, List, Literal, Optional, Sequence, Union

from .datasets import (
    Pytorch3DUnetDatasetConfig,
    SBIAD1410PhaseConfig,
    TIF_dataset_names,
    TIFPhaseConfig,
    TIFtxtPhaseConfig,
    transforms_type,
)
from .slice_builders import slice_builder_type


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


class Pytorch3DUnetValLoaderConfig(Pytorch3DUnetLoaderConfig, frozen=True):
    val: Pytorch3DUnetDatasetConfig


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
        slice_builder_type,
        Discriminator("name"),
    ]
    transformer: transforms_type
    roi: Optional[Union[Sequence[Sequence[int]], Sequence[int]]]

    def create_config(
        self,
        output_dir: Optional[str],
        data_base_path: str,
        phase: Literal["train", "test", "val"] = "test",
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

        elif phase == "val":
            loader = Pytorch3DUnetValLoaderConfig(
                dataset=self.dataset,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                raw_internal_path=self.raw_internal_path,
                label_internal_path=self.label_internal_path,
                global_normalization=self.global_normalization,
                global_percentiles=self.global_percentiles,
                val=Pytorch3DUnetDatasetConfig(
                    file_paths=file_paths,
                    slice_builder=self.slice_builder,
                    transformer=self.transformer,
                    roi=self.roi,
                ),
            )
            return loader
        else:
            assert_never(phase)


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


class SBIAD1410LoaderValConfig(SBIAD1410LoaderConfig, frozen=True):
    val: SBIAD1410PhaseConfig


class SBIAD1410LoaderMetaConfig(BaseModel, frozen=True):
    dataset: Literal["S_BIAD1410_Dataset"]
    batch_size: int
    num_workers: int
    global_normalization: bool
    global_percentiles: Optional[Sequence[Union[float, int]]]
    img_paths: Sequence[str]
    mask_paths: Sequence[str]
    roi: Optional[Sequence[Sequence[int]]]
    transformer: transforms_type
    slice_builder: Optional[slice_builder_type]

    def create_config(
        self,
        output_dir: Optional[str],
        data_base_path: str,
        phase: Literal["train", "test", "val"] = "test",
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
        elif phase == "val":
            loader = SBIAD1410LoaderValConfig(
                dataset=self.dataset,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                global_normalization=self.global_normalization,
                global_percentiles=self.global_percentiles,
                val=SBIAD1410PhaseConfig(
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


class TIFLoadersConfig(BaseModel, frozen=True):
    dataset: TIF_dataset_names
    batch_size: int
    num_workers: int
    global_norm: bool
    percentiles: Optional[Sequence[Union[float, int]]]


class TIFPredictionLoadersConfig(TIFLoadersConfig, frozen=True):
    output_dir: str
    test: Union[TIFPhaseConfig, TIFtxtPhaseConfig]


class TIFTrainLoadersConfig(TIFLoadersConfig, frozen=True):
    train: Union[TIFPhaseConfig, TIFtxtPhaseConfig]


class TIFValLoadersConfig(TIFLoadersConfig, frozen=True):
    val: Union[TIFPhaseConfig, TIFtxtPhaseConfig]


class LoaderMetaConfig(BaseModel):
    batch_size: int
    num_workers: int
    global_norm: bool
    percentiles: Optional[Sequence[float]]
    image_dir: Sequence[str]
    mask_dir: Sequence[str]
    transformer: transforms_type


class TIFLoaderMetaConfig(LoaderMetaConfig):
    dataset: Literal["Standard_TIF_Dataset", "HeLaNuc_Dataset", "Hoechst_Dataset"]

    def create_config(
        self,
        output_dir: Optional[str],
        data_base_path: str,
        phase: Literal["train", "test", "val"] = "test",
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
        elif phase == "val":
            loader = TIFValLoadersConfig(
                dataset=self.dataset,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                global_norm=self.global_norm,
                percentiles=self.percentiles,
                val=TIFPhaseConfig(
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
        phase: Literal["test", "train", "val"] = "test",
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
        elif phase == "val":
            loader = TIFValLoadersConfig(
                dataset=self.dataset,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                global_norm=self.global_norm,
                percentiles=self.percentiles,
                val=TIFtxtPhaseConfig(
                    image_dir=image_dir,
                    mask_dir=mask_dir,
                    filenames_path=data_base_path + self.filenames_path,
                    transformer=self.transformer,
                ),
            )
        else:
            assert_never(phase)
        return loader


loader_type = Annotated[
    Union[
        Pytorch3DUnetLoaderMetaConfig,
        TIFLoaderMetaConfig,
        TIFtxtLoaderMetaConfig,
        SBIAD1410LoaderMetaConfig,
    ],
    Discriminator("dataset"),
]


semantic_loaders_type = Annotated[
    Union[
        Pytorch3DUnetTrainLoaderConfig,
        TIFTrainLoadersConfig,
        SBIAD1410LoaderTrainConfig,
    ],
    Discriminator("name"),
]
