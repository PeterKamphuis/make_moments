# -*- coding: future_fstrings -*-

from dataclasses import dataclass,field
import omegaconf
from omegaconf import MISSING
from typing import List,Optional
import os


@dataclass
class defaults:
    filename: str = MISSING
    mask: Optional[str] = None
    log: Optional[str] = None
    output_name: Optional[str] = None
    output_directory: str = f'{os.getcwd()}'
    level: Optional[float] = None
    moments: List = field(default_factory=lambda: [0,1,2])
    threshold: float = 3.
    debug: bool = False
    velocity_unit: Optional[str] = None
    overwrite: bool=False
