# -*- coding: future_fstrings -*-

# This is the stand alone version of the pyFAT moments to create moment maps

#from optparse import OptionParser
from omegaconf import OmegaConf,MissingMandatoryValue
from make_moments.config_defaults import process_input
from make_moments.functions import extract_pv,moments
from astropy.io import fits

import numpy as np
import make_moments

import sys
import os
import traceback
import warnings
from astropy.wcs import WCS



def warn_with_traceback(message, category, filename, lineno, file=None, line=None):
    log = file if hasattr(file,'write') else sys.stderr
    traceback.print_stack(file=log)
    log.write(warnings.formatwarning(message, category, filename, lineno, line))

def main_trace_moments():
    from viztracer import VizTracer
    with VizTracer(output_file="Moments_Viztracer.json",min_duration=1000) as tracer:
        main_moments()


def main_PV():
    main(sys.argv[1:],create_PV = True,makemoments=False)

def main_moments():
    main(sys.argv[1:])


def main(argv,makemoments=True,create_PV = False):
  

    cfg = process_input(argv,moments_default=makemoments,PV_default=create_PV)

    if makemoments:       
        moments(filename = cfg.cube_name, mask = cfg.mask, moments = cfg.moments,\
                     overwrite = cfg.overwrite, level= cfg.level,\
                     cube_velocity_unit= cfg.cube_velocity_unit, threshold = cfg.threshold,\
                     debug = cfg.debug, log=cfg.log,map_velocity_unit = cfg.map_velocity_unit,\
                     output_directory = cfg.output_directory,\
                     output_name = cfg.output_name)
   
    if create_PV:
        extract_pv(filename = cfg.cube_name,overwrite = cfg.overwrite,\
                    cube_velocity_unit= cfg.cube_velocity_unit,PA=cfg.PA,\
                    center= cfg.center,finalsize=cfg.finalsize,\
                    convert= cfg.convert,log = cfg.log,\
                    map_velocity_unit = cfg.map_velocity_unit,\
                    output_directory = cfg.output_directory,
                    restfreq=cfg.restfreq,carta=cfg.carta,
                    velocity_type = cfg.velocity_type,
                    spectral_frame=cfg.spectral_frame,
                    output_name =cfg.output_name ,debug =cfg.debug)


if __name__ =="__main__":
    main()
