# -*- coding: future_fstrings -*-

# This is the stand alone version of the pyFAT moments to create moment maps

#from optparse import OptionParser
from omegaconf import OmegaConf,MissingMandatoryValue
from make_moments.config_defaults import get_config
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





def main(argv,makemoments=True,create_PV = False):
    if '-v' in argv or '--version' in argv:
        print(f"This is version {make_moments.__version__} of the program.")
        sys.exit()

    if '-h' in argv or '--help' in argv:
        print('''
Use make_moments in this way:
make_moments -c inputfile.yml   where inputfile is a yaml config file with the desired input settings.
make_moments -h print this message
make_moments -e prints a yaml file (defaults.yml) with the default setting in the current working directory.
or in the same way for create_PV_diagram instead of make_moments
in this file values designated ??? indicated values without defaults.



All config parameters can be set directly from the command line by setting the correct parameters, e.g:
make_moments filename=cube.fits mask=mask.fits to make moment maps of the file cube.fits where the maps are masked with mask.fits
''')
        sys.exit()



    cfg = get_config(moments_default=makemoments,PV_default=create_PV)

    if '-e' in argv:
        with open('default.yml','w') as default_write:
            default_write.write(OmegaConf.to_yaml(cfg))
        print(f'''We have printed the file default.yml in {os.getcwd()}.
Exiting moments.''')
        sys.exit()
        
    if '-c' in argv:
        configfile = argv[argv.index('-c')+1]
        inputconf = OmegaConf.load(configfile)
        #merge yml file with defaults
        cfg = OmegaConf.merge(cfg,inputconf)
        argv.remove('-c')
        argv.remove(configfile)
    # read command line arguments anything list input should be set in '' e.g. pyROTMOD 'rotmass.MD=[1.4,True,True]'
    inputconf = OmegaConf.from_cli(argv)
    cfg = OmegaConf.merge(cfg,inputconf)
    
    if makemoments:
        if not cfg.mask and not cfg.level and not cfg.threshold:
            print(f'''You have to specify a mask, cutoff level (in cube units), or threshold (in sigma) to mask the cube with''')
            sys.exit(1)
        moments(filename = cfg.filename, mask = cfg.mask, moments = cfg.moments,\
                     overwrite = cfg.overwrite, level= cfg.level,\
                     velocity_unit= cfg.velocity_unit, threshold = cfg.threshold,\
                     debug = cfg.debug, log=cfg.log,\
                     output_directory = cfg.output_directory,\
                     output_name = cfg.output_name)
   
    if create_PV:
        extract_pv(filename = cfg.filename,overwrite = cfg.overwrite,\
                    velocity_unit= cfg.velocity_unit,PA=cfg.PA,\
                    center= cfg.center,finalsize=cfg.finalsize,\
                    convert= cfg.convert,log = cfg.log,\
                    output_directory = cfg.output_directory,
                    restfreq=cfg.restfreq,carta=cfg.carta,
                    spectral_frame=cfg.spectral_frame,
                    output_name =cfg.output_name ,debug =cfg.debug)


if __name__ =="__main__":
    main()
