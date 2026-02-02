from pydantic import BaseModel, Discriminator
from typing import Annotated, List, Literal, Mapping, Optional, Sequence, Union


from .slice_builders import slice_builder_type

transforms_type = Mapping[
    str,
    List[Mapping[str, Optional[Union[str, bool, int, Sequence[int], Sequence[float]]]]],
]


TIF_dataset_names = Literal[
    "TIF_txt_Dataset", "Standard_TIF_Dataset", "HeLaNuc_Dataset", "Hoechst_Dataset"
]


class TIFPhaseConfig(BaseModel):
    image_dir: Sequence[str]
    mask_dir: Sequence[str]
    transformer: transforms_type


class TIFtxtPhaseConfig(BaseModel, frozen=True):
    image_dir: Sequence[str]
    mask_dir: Sequence[str]
    filenames_path: str
    transformer: transforms_type


class SBIAD1410PhaseConfig(BaseModel):
    img_paths: Sequence[str]
    mask_paths: Sequence[str]
    roi: Optional[Sequence[Sequence[int]]]
    transformer: transforms_type
    slice_builder: Optional[slice_builder_type]


class SBIAD1410PhaseMetaConfig(BaseModel):
    mask_paths: Optional[Sequence[str]]
    roi: Optional[Sequence[Sequence[int]]]
    transformer: transforms_type
    slice_builder: Optional[
        Annotated[
            slice_builder_type,
            Discriminator("name"),
        ]
    ]


class Pytorch3DUnetDatasetConfig(BaseModel, frozen=True):
    file_paths: Sequence[str]
    slice_builder: Annotated[
        slice_builder_type,
        Discriminator("name"),
    ]
    transformer: transforms_type
    roi: Optional[Union[Sequence[Sequence[int]], Sequence[int]]]
