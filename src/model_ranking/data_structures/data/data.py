from typing import Annotated, Literal, Optional, Union

from pydantic import BaseModel, Discriminator


from .dataloaders import (
    Pytorch3DUnetLoaderMetaConfig,
    SBIAD1410LoaderMetaConfig,
    TIFLoaderMetaConfig,
    TIFtxtLoaderMetaConfig,
    loader_type,
)
from .datasets import SBIAD1410PhaseMetaConfig
from .eval_dataloaders import (
    Eval_TIF_DataloaderMetaConfig,
    Eval_TIF_TxtDataloaderMetaConfig,
    EvalDataloaderMetaConfig,
    EvalSB1410DataloaderMetaConfig,
    eval_dataloader_type,
)
from .slice_builders import (
    Pytorch3DUnetFilterSliceBuilderConfig,
    Pytorch3DUnetSliceBuilderConfig,
)
from model_ranking.data_structures.common_types import (
    dataset_names,
    ForegroundFilterConfig,
)
from model_ranking.data_structures.general.predictor import (
    Pytorch3DUnetPredictorMetaConfig,
    TIFNucleiInstancePredictorConfig,
    TIFNucleiSemanticPredictorConfig,
    predictor_semantic_type,
    predictor_instance_type,
)


class TargetDatasetConfigBase(BaseModel, frozen=True):
    name: dataset_names
    loader: loader_type
    predictor_semantic: Optional[predictor_semantic_type]
    predictor_instance: Optional[predictor_instance_type]
    eval_dataloader_instance: Optional[eval_dataloader_type]
    eval_dataloader_semantic: Optional[eval_dataloader_type]
    consis_dataloader_instance: Optional[eval_dataloader_type]
    consis_dataloader_semantic: Optional[eval_dataloader_type]
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
    feature_loader: TIFtxtLoaderMetaConfig = TIFtxtLoaderMetaConfig(
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
            "label": [
                {"name": "CropToFixed", "size": (512, 512), "centered": True},
                {"name": "Relabel"},
                {"name": "BlobsToMask", "append_label": False},
                {"name": "ToTensor", "expand_dims": True},
            ],
        },
    )
    feature_indices_path: Optional[str] = None
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
    feature_loader: TIFLoaderMetaConfig = TIFLoaderMetaConfig(
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
            ],
            "label": [
                {"name": "Relabel"},
                {"name": "BlobsToMask", "append_label": False},
                {"name": "ToTensor", "expand_dims": True},
            ],
        },
    )
    feature_indices_path: Optional[str] = None
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
    feature_loader: TIFLoaderMetaConfig = TIFLoaderMetaConfig(
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
            "label": [
                {"name": "Relabel"},
                {"name": "BlobsToMask", "append_label": False},
                {"name": "ToTensor", "expand_dims": True},
            ],
        },
    )
    feature_indices_path: Optional[str] = None
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
    feature_loader: TIFtxtLoaderMetaConfig = TIFtxtLoaderMetaConfig(
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
            "label": [
                {"name": "Relabel"},
                {"name": "BlobsToMask", "append_label": False},
                {"name": "ToTensor", "expand_dims": True},
            ],
        },
    )
    feature_indices_path: Optional[str] = None
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
    feature_loader: TIFLoaderMetaConfig = TIFLoaderMetaConfig(
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
            "label": [
                {"name": "Relabel"},
                {"name": "BlobsToMask", "append_label": False},
                {"name": "ToTensor", "expand_dims": True},
            ],
        },
    )
    feature_indices_path: Optional[str] = None
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
    feature_loader: Pytorch3DUnetLoaderMetaConfig = Pytorch3DUnetLoaderMetaConfig(
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
            "label": [
                {"name": "ToTensor", "expand_dims": True},
            ],
        },
        slice_builder=Pytorch3DUnetSliceBuilderConfig(
            name="SliceBuilder",
            patch_shape=(1, 200, 200),
            stride_shape=(1, 200, 200),
            halo_shape=(0, 0, 0),
        ),
    )
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
        gt_set_id_to_zero=None,
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
        gt_set_id_to_zero=None,
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
        gt_set_id_to_zero=None,
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
        gt_set_id_to_zero=None,
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
    feature_loader: SBIAD1410LoaderMetaConfig = SBIAD1410LoaderMetaConfig(
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
            halo_shape=(0, 0, 0),
        ),
    )
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
                {"name": "CropToFixed", "size": (256, 256), "centered": True},
                {"name": "ToTensor", "expand_dims": True},
            ],
        },
    )
    feature_loader: TIFLoaderMetaConfig = TIFLoaderMetaConfig(
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
                {"name": "CropToFixed", "size": (256, 256), "centered": True},
                {"name": "ToTensor", "expand_dims": True},
            ],
            "label": [
                {"name": "CropToFixed", "size": (256, 256), "centered": True},
                {"name": "Relabel"},
                {"name": "BlobsToMask", "append_label": False},
                {"name": "ToTensor", "expand_dims": True},
            ],
        },
    )
    feature_indices_path: Optional[str] = None
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
                    {"name": "CropToFixed", "size": (256, 256), "centered": True},
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
    feature_loader: Pytorch3DUnetLoaderMetaConfig = Pytorch3DUnetLoaderMetaConfig(
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
            ],
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
            halo_shape=(0, 0, 0),
        ),
    )
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
        gt_set_id_to_zero=None,
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
        gt_set_id_to_zero=None,
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
        gt_set_id_to_zero=None,
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
        gt_set_id_to_zero=None,
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
        # batch_size=5,
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
            # patch_shape=(1, 639, 765),
            # stride_shape=(1, 639, 765),
            # halo_shape=(0, 96, 96),
            patch_shape=(1, 256, 256),
            stride_shape=(1, 256, 256),
            # halo_shape=(0, 32, 32),
            halo_shape=(0, 0, 0),
        ),
        # slice_builder=Pytorch3DUnetSingleZSliceBuilderConfig(
        #     name="SingleZSliceBuilder",
        #     patch_shape=(1, 640, 640),
        #     stride_shape=(1, 640, 640),
        #     halo_shape=(0, 0, 0),
        # ),
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
            zero_large_instances=True,
            large_instance_multiplier=1.5,
            max_obj_size=3387,
            threshold=0.5,
        )
    )
    eval_dataloader_semantic: None = None
    eval_dataloader_instance: EvalDataloaderMetaConfig = EvalDataloaderMetaConfig(
        name="StandardEvalDataset",
        gt_path=("/FlyWing/GT/test/per03.h5",),
        pred_key="segmentation",
        # pred_key="seg_04th_05b",
        gt_key="volumes/labels/expanded_cells_with_ignore",
        patch_key="patch_index",
        roi=None,
        ignore_index=-1,
        ignore_path=None,
        ignore_key=None,
        convert_to_boundary_label=False,
        convert_to_binary_label=False,
        min_object_size=None,
        gt_zero_largest_instance=False,
        gt_zero_large_instances=False,
        gt_set_id_to_zero=None,
        zero_large_instances=False,
        zero_largest_instance=False,
        largest_obj_multiplier=1.5,
        max_obj_size=3387,
        batch_size=32,
        num_workers=8,
    )
    consis_dataloader_semantic: None = None
    consis_dataloader_instance: EvalDataloaderMetaConfig = EvalDataloaderMetaConfig(
        name="StandardEvalDataset",
        gt_path=None,
        pred_key="segmentation",
        gt_key="segmentation",
        # pred_key="seg_04th_05b",
        # gt_key="seg_04th_05b",
        patch_key="patch_index",
        roi=None,
        ignore_index=None,
        ignore_path=None,
        ignore_key=None,
        # ignore_path="/FlyWing/GT/test/per03.h5",
        # ignore_key="volumes/labels/ignore",
        convert_to_boundary_label=False,
        convert_to_binary_label=False,
        min_object_size=None,
        gt_zero_largest_instance=False,
        gt_zero_large_instances=False,
        gt_set_id_to_zero=None,
        zero_large_instances=False,
        zero_largest_instance=False,
        largest_obj_multiplier=1.5,
        max_obj_size=3387,
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
        # batch_size=5,
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
            # patch_shape=(1, 960, 1000),
            # stride_shape=(1, 960, 1000),
            # halo_shape=(0, 96, 96),
            patch_shape=(1, 256, 256),
            stride_shape=(1, 256, 256),
            # halo_shape=(0, 32, 32),
            halo_shape=(0, 0, 0),
        ),
        # slice_builder=Pytorch3DUnetSingleZSliceBuilderConfig(
        #     name="SingleZSliceBuilder",
        #     patch_shape=(1, 640, 640),
        #     stride_shape=(1, 640, 640),
        #     halo_shape=(0, 0, 0),
        # ),
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
            zero_large_instances=True,
            large_instance_multiplier=1.7,
            max_obj_size=5867,
            threshold=0.5,
        )
    )
    eval_dataloader_semantic: None = None
    eval_dataloader_instance: EvalDataloaderMetaConfig = EvalDataloaderMetaConfig(
        name="StandardEvalDataset",
        gt_path=("/Ovules/GT2x/test/N_294_final_crop_ds2.h5",),
        pred_key="segmentation",
        # pred_key="seg_04th_05b",
        gt_key="label_with_ignore",
        patch_key="patch_index",
        roi=None,
        ignore_index=-1,
        ignore_path=None,
        ignore_key=None,
        convert_to_boundary_label=False,
        convert_to_binary_label=False,
        min_object_size=None,
        gt_zero_largest_instance=False,
        gt_zero_large_instances=False,
        gt_set_id_to_zero=None,
        zero_large_instances=False,
        zero_largest_instance=False,
        largest_obj_multiplier=1.7,
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
        # pred_key="seg_04th_05b",
        # gt_key="seg_04th_05b",
        patch_key="patch_index",
        roi=None,
        ignore_index=None,
        ignore_path="/Ovules/GT2x/test/N_294_final_crop_ds2.h5",
        ignore_key="ignore_mask",
        convert_to_boundary_label=False,
        convert_to_binary_label=False,
        min_object_size=None,
        gt_zero_large_instances=False,
        gt_zero_largest_instance=False,
        gt_set_id_to_zero=None,
        zero_large_instances=False,
        zero_largest_instance=False,
        largest_obj_multiplier=1.7,
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
        batch_size=5,
        # batch_size=32,
        num_workers=8,
        raw_internal_path="raw",
        label_internal_path="label",
        global_normalization=True,
        global_percentiles=(5, 95),
        file_paths=(
            # "/PNAS/test/12hrs_plant18_trim-acylYFP.h5",
            "/PNAS/test/24hrs_plant18_trim-acylYFP.h5",
            # "/PNAS/test/36hrs_plant18_trim-acylYFP.h5",
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
            # halo_shape=(0, 32, 32),
            # patch_shape=(1, 512, 512),
            # stride_shape=(1, 512, 512),
            # halo_shape=(0, 64, 64),
            halo_shape=(0, 0, 0),
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
            zero_large_instances=True,
            large_instance_multiplier=1.5,
            max_obj_size=2173,
        )
    )
    eval_dataloader_semantic: None = None
    eval_dataloader_instance: EvalDataloaderMetaConfig = EvalDataloaderMetaConfig(
        name="StandardEvalDataset",
        gt_path=(
            # "/PNAS/test/12hrs_plant18_trim-acylYFP.h5",
            "/PNAS/test/24hrs_plant18_trim-acylYFP.h5",
            # "/PNAS/test/hrs_plant18_trim-acylYFP.h5",
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
        gt_zero_largest_instance=False,
        gt_zero_large_instances=False,
        gt_set_id_to_zero=1,
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
        gt_set_id_to_zero=None,
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
        # batch_size=32,
        batch_size=5,
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
            # patch_shape=(1, 480, 640),
            # stride_shape=(1, 480, 640),
            # halo_shape=(0, 64, 64),
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
        batch_size=10,
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
        # slice_builder=Pytorch3DUnetSliceBuilderConfig(
        #     name="SliceBuilder",
        #     patch_shape=(1, 256, 256),
        #     stride_shape=(1, 256, 256),
        #     halo_shape=(0, 0, 0),
        # ),
        slice_builder=Pytorch3DUnetFilterSliceBuilderConfig(
            name="FilterSliceBuilder",
            patch_shape=(1, 256, 256),
            stride_shape=(1, 256, 256),
            halo_shape=(0, 0, 0),
            threshold=0.02,
            slack_acceptance=0,
            ignore_index=None,
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
        gt_set_id_to_zero=None,
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
        gt_set_id_to_zero=None,
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
        # batch_size=32,
        batch_size=2,
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
            patch_shape=(1, 1280, 1280),
            stride_shape=(1, 1280, 1280),
            halo_shape=(0, 96, 96),
            # patch_shape=(1, 256, 256),
            # stride_shape=(1, 256, 256),
            # halo_shape=(0, 32, 32),
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
        batch_size=10,
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
        # slice_builder=Pytorch3DUnetSliceBuilderConfig(
        #     name="SliceBuilder",
        #     patch_shape=(1, 256, 256),
        #     stride_shape=(1, 256, 256),
        #     halo_shape=(0, 0, 0),
        # ),
        slice_builder=Pytorch3DUnetFilterSliceBuilderConfig(
            name="FilterSliceBuilder",
            patch_shape=(1, 256, 256),
            stride_shape=(1, 256, 256),
            halo_shape=(0, 0, 0),
            threshold=0.1,
            slack_acceptance=0,
            ignore_index=None,
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
        gt_set_id_to_zero=None,
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
        gt_set_id_to_zero=None,
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
        # batch_size=32,
        batch_size=2,
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
            patch_shape=(1, 1280, 1280),
            stride_shape=(1, 1280, 1280),
            halo_shape=(0, 96, 96),
            # patch_shape=(1, 256, 256),
            # stride_shape=(1, 256, 256),
            # # halo_shape=(0, 32, 32),
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
        batch_size=10,
        num_workers=8,
        raw_internal_path="raw",
        label_internal_path="labels",
        global_normalization=True,
        # global_normalization=False,
        global_percentiles=None,
        file_paths=("/Rmito/test_converted.h5",),
        roi=[[0, 80], [0, 1280], [0, 1280]],
        transformer={
            "raw": [
                {"name": "Normalize"},
                {"name": "ToTensor", "expand_dims": True},
            ],
            "label": [
                {"name": "ToTensor", "expand_dims": True},
            ],
        },
        # slice_builder=Pytorch3DUnetSliceBuilderConfig(
        #     name="SliceBuilder",
        #     patch_shape=(1, 256, 256),
        #     stride_shape=(1, 256, 256),
        #     halo_shape=(0, 0, 0),
        # ),
        slice_builder=Pytorch3DUnetFilterSliceBuilderConfig(
            name="FilterSliceBuilder",
            patch_shape=(1, 256, 256),
            stride_shape=(1, 256, 256),
            halo_shape=(0, 0, 0),
            threshold=0.1,
            slack_acceptance=0,
            ignore_index=None,
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
        gt_set_id_to_zero=None,
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
        gt_set_id_to_zero=None,
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
        # batch_size=32,
        batch_size=5,
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
            patch_shape=(1, 589, 589),
            stride_shape=(1, 589, 589),
            halo_shape=(0, 64, 64),
            # patch_shape=(1, 256, 256),
            # stride_shape=(1, 256, 256),
            # # halo_shape=(0, 32, 32),
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
        batch_size=10,
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
        # slice_builder=Pytorch3DUnetSliceBuilderConfig(
        #     name="SliceBuilder",
        #     patch_shape=(1, 256, 256),
        #     stride_shape=(1, 256, 256),
        #     halo_shape=(0, 0, 0),
        # ),
        slice_builder=Pytorch3DUnetFilterSliceBuilderConfig(
            name="FilterSliceBuilder",
            patch_shape=(1, 256, 256),
            stride_shape=(1, 256, 256),
            halo_shape=(0, 0, 0),
            threshold=0.02,
            slack_acceptance=0,
            ignore_index=None,
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
        gt_set_id_to_zero=None,
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
        gt_set_id_to_zero=None,
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
            # image_key="prediction",
            image_key="segmentation",
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
            # image_key="prediction",
            image_key="segmentation",
            min_object_size=0,
            zero_large_instances=False,
            mask_dir=None,
            # mask_key="prediction",
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

semantic_dataset_type = Annotated[
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
        EPFLTargetConfig,
        HmitoTargetConfig,
        RmitoTargetConfig,
        VNCTargetConfig,
    ],
    Discriminator("name"),
]
