# -*- coding: future_fstrings -*-

from dataclasses import dataclass,field
from omegaconf import MISSING,OmegaConf
from typing import List,Optional

import psutil
import os
import sys
import make_moments

class InputError(Exception):
    pass


def get_default_center():
    return ['RA ("hh:mm:ss" or deg)','DEC ("hh:mm:ss" or deg)','v_sys (km/s)']

def get_config(moments_default=True,PV_default=False):

    @dataclass
    class defaults:
        cube_name: str = MISSING   #Line Cube to extract the moment maps or PV from !!!No default
        mask: Optional[str] = None  #Mask to use for masking moments
        log: Optional[str] = None   #possible log file for printing the output in
        output_name: Optional[str] = None   #string to use for out put, if set the output will be {output_name}_mom0.fits where the end is modified for the proper output
        output_directory: str = f'{os.getcwd()}' # directory where to put the output
        debug: bool = False # Trigger to print additional info
        cube_velocity_unit: Optional[str] = None #Velocity units of the input cube 
        map_velocity_unit: Optional[str] = None #Requiested velocity units of the output
        overwrite: bool=False #Overwrite existing files?
        print_examples: bool =False
        configuration_file: Optional[str] = None
        try:
            ncpu: int = len(psutil.Process().cpu_affinity())
        except AttributeError:
            ncpu: int = psutil.cpu_count()

        if moments_default:
            level: Optional[float] = None #level below which emission in the cube is not added
            moments: List = field(default_factory=lambda: [0,1,2]) #which moments to produce
            threshold: float = 3. #Same as level but calculates level as threshold * cube_rms
        if PV_default:
            PA: float = 16 #Position angle where to extract PV
            center: List = field(default_factory=lambda: get_default_center()) #Central position of extraction in wcs
            finalsize: List= field(default_factory=lambda: [-1,-1,-1]) #final size of output in pixels
            convert: float = -1. #conversion factor for velocity axis
            carta: bool = False #Carta will only accept stupid fequency axis
            restfreq: float = 1.420405751767E+09 #hz
            spectral_frame: str = 'BARYCENT' #Spectral frame to set
            velocity_type: Optional[str] = None #Type of velocity axis

    cfg = OmegaConf.structured(defaults)
    return cfg

def process_input(argv,moments_default=True,PV_default=False):
    if '-v' in argv or '--version' in argv:
        print(f"This is version {make_moments.__version__} of the program.")
        sys.exit()
    file_name = 'moments_defaults.yml'
    program = 'make_moments'
    if PV_default and moments_default:
        file_name = 'PV_and_moments_defaults.yml'
        program='make_moments or create_PV_diagram'
    elif PV_default:
        file_name = 'PV_defaults.yml'
        program='create_PV_diagram'
    if '-h' in argv or '--help' in argv:      
        print(return_help_message(program,file_name))
        sys.exit()
    #First import the defaults 
    cfg = get_config(moments_default=moments_default,PV_default=PV_default)
    #if ncpu is the total available remove 1
    if cfg.ncpu == psutil.cpu_count():
        cfg.ncpu -= 1

    # read command line arguments anything list input should be set in brackets '' e.g. pyROTMOD 'rotmass.MD=[1.4,True,True]'
    inputconf = OmegaConf.from_cli(argv)
    cfg_input = OmegaConf.merge(cfg,inputconf)

    # Print examples if requested
    if cfg_input.print_examples:
        with open(file_name,'w') as default_write:
            default_write.write(OmegaConf.to_yaml(cfg))
        print(f'''We have printed the file {file_name} in {os.getcwd()}.
''')
        sys.exit()

    #if a configuration file is provided read it
    if not cfg_input.configuration_file is None:
        succes = False
        while not succes:
            try:
                yaml_config = OmegaConf.load(cfg_input.configuration_file)
        #merge yml file with defaults
                cfg = OmegaConf.merge(cfg,yaml_config)
                succes = True
            except FileNotFoundError:
                cfg_input.configuration_file = input(f'''
You have provided a config file ({cfg_input.configuration_file}) but it can't be found.
If you want to provide a config file please give the correct name.
Else press CTRL-C to abort.
configuration_file = ''')
    # make sure the command line overwrite the file
    cfg = OmegaConf.merge(cfg,inputconf)
    if PV_default:
        if cfg.center == get_default_center():
            cfg.center= [None for x in cfg.center]

    if moments_default:
        if not cfg.mask and not cfg.level and not cfg.threshold:
            print(f'''You have to specify a mask, cutoff level (in cube units), or threshold (in sigma) to mask the cube with''')
            sys.exit(1)
   
    #set an output base name
    if cfg.output_name is None:
        cfg.output_name= f'{os.path.splitext(os.path.split(cfg.cube_name)[1])[0]}'
    
    return cfg

def return_help_message(program, file_name):
    help_message = f'''
Use {program} in this way:
{program} configuration_file=inputfile.yml   where inputfile is a yaml config file with the desired input settings.
{program} -h print this message
{program} print_examples=true print a yaml file ({file_name}) with the default setting in the current working directory.
in this file values designated ??? indicated values without defaults.

All config parameters can be set directly from the command line by setting the correct parameters, e.g:
{program} cube_name=cube.fits mask=mask.fits to make moment maps of the file cube.fits where the maps are masked with mask.fits
'''
    return help_message   
