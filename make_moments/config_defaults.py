# -*- coding: future_fstrings -*-

from dataclasses import dataclass,field
from omegaconf import MISSING,OmegaConf,MissingMandatoryValue
from typing import List,Optional
import os

def get_config(moments_default=True,PV_default=False):

    @dataclass
    class defaults:
        filename: str = MISSING
        mask: Optional[str] = None
        log: Optional[str] = None
        output_name: Optional[str] = None
        output_directory: str = f'{os.getcwd()}'
        debug: bool = False
        velocity_unit: Optional[str] = None
        overwrite: bool=False
        if moments_default:
            level: Optional[float] = None
            moments: List = field(default_factory=lambda: [0,1,2])
            threshold: float = 3.
        if PV_default:
            PA: float = 16
            center: List = field(default_factory=lambda: [0.,0.,0.])
            finalsize: List= field(default_factory=lambda: [-1,-1,-1])
            convert: float = -1.
            carta: bool = False #Carta will only accept stupid fequency axis
            restfreq: float = 1.420405751767E+09 #hz
            spectral_frame: str = 'BARYCENT'

    cfg = OmegaConf.structured(defaults)
    return cfg
