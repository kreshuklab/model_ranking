from pydantic import BaseModel
from typing import Literal, Optional, Sequence


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
