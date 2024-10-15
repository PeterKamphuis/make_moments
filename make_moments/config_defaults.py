# -*- coding: future_fstrings -*-

from dataclasses import dataclass,field
from omegaconf import MISSING,OmegaConf
from typing import List,Optional
import os

def get_config(moments_default=True,PV_default=False):

    @dataclass
    class defaults:
        filename: str = MISSING   #Line Cube to extract the moment maps or PV from !!!No default
        mask: Optional[str] = None  #Mask to use for masking moments
        log: Optional[str] = None   #possible log file for printing the output in
        output_name: Optional[str] = None   #string to use for out put, if set the output will be {output_name}_mom0.fits where the end is modified for the proper output
        output_directory: str = f'{os.getcwd()}' # directory where to put the output
        debug: bool = False # Trigger to print additional info
        cube_velocity_unit: Optional[str] = None #Velocity units of the input cube 
        map_velocity_unit: Optional[str] = None #Requiested velocity units of the output
        overwrite: bool=False #Overwrite existing files?
        if moments_default:
            level: Optional[float] = None #level below which emission in the cube is not added
            moments: List = field(default_factory=lambda: [0,1,2]) #which moments to produce
            threshold: float = 3. #Same as level but calculates level as threshold * cube_rms
        if PV_default:
            PA: float = 16 #Position angle where to extract PV
            center: List = field(default_factory=lambda: [None,None,None]) #Central position of extraction in wcs
            finalsize: List= field(default_factory=lambda: [-1,-1,-1]) #final size of output in pixels
            convert: float = -1. #conversion factor for velocity axis
            carta: bool = False #Carta will only accept stupid fequency axis
            restfreq: float = 1.420405751767E+09 #hz
            spectral_frame: str = 'BARYCENT' #Spectral frame to set
            velocity_type: Optional[str] = None #Type of velocity axis

    cfg = OmegaConf.structured(defaults)
    return cfg
