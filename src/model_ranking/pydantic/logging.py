from pydantic import BaseModel
from typing import Literal, Optional, Union


class WandbConfig(BaseModel):
    project: str
    name: str
    mode: Literal["disabled", "online", "offline"]
    run_id: Optional[str] = None
    resume: Union[bool, None, Literal["allow", "never", "must", "auto"]] = None
