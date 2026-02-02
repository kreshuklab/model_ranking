from pydantic import BaseModel
from typing import Annotated, List, Literal, Optional, Sequence, Union
from pydantic import Discriminator

from .datasets import SBIAD1410PhaseConfig, SBIAD1410PhaseMetaConfig

from .eval_datasets import (
    EvalDatasetConfig,
    TIFEvalDatasetConfig,
    SBIAD1410EvalDatasetConfig,
)


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
    gt_set_id_to_zero: Optional[int]
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
                gt_set_id_to_zero=self.gt_set_id_to_zero,
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
                gt_set_id_to_zero=self.gt_set_id_to_zero,
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
