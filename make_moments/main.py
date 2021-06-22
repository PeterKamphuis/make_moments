# -*- coding: future_fstrings -*-

# This is the stand alone version of the pyFAT moments to create moment maps

#from optparse import OptionParser
from omegaconf import OmegaConf,MissingMandatoryValue
import make_moments
from make_moments.config_defaults import defaults
from astropy.io import fits
import numpy as np
import copy
import sys
import os

class InputError(Exception):
    pass


def warn_with_traceback(message, category, filename, lineno, file=None, line=None):
    log = file if hasattr(file,'write') else sys.stderr
    traceback.print_stack(file=log)
    log.write(warnings.formatwarning(message, category, filename, lineno, line))

def print_log(log_statement,log=None, screen = False,debug = False):
    log_statement = f"{log_statement}"
    if screen or not log:
        print(log_statement)
    if log:
        with open(log,'a') as log_file:
            log_file.write(log_statement)

print_log.__doc__ =f'''
 NAME:
    print_log
 PURPOSE:
    Print statements to log if existent and screen if Requested
 CATEGORY:
    support_functions

 INPUTS:
    log_statement = statement to be printed
    log = log to print to, can be None

 OPTIONAL INPUTS:
    debug = False

    screen = False
    also print the statement to the screen

 OUTPUTS:
    line in the log or on the screen

 OPTIONAL OUTPUTS:

 PROCEDURES CALLED:
    linenumber, .write

 NOTE:
    If the log is None messages are printed to the screen.
    This is useful for testing functions.
'''

def moments(filename = None, mask = None, moments = [0,1,2],
                 overwrite = False, level=None,velocity_unit= None, threshold = 3.,
                  debug = False, log=None,output_directory = None,output_name =None):

    if not filename:
        InputError("There is no default for filename, it needs be sets to the input cube fits file.")

    if not mask and not level and not threshold:
        InputError("Moments requires a threshold (sigma), level (units of input cube) or a mask. ")


    if not output_directory:
        output_directory= f'{os.getcwd()}'
    if not output_name:
        output_name= f'{os.path.splitext(os.path.split(filename)[1])[0]}'
    cube = fits.open(filename)
    if velocity_unit:
        cube[0].header['CUNIT3'] = velocity_unit
    if mask:
        mask_cube = fits.open(mask)
        if len(np.where(mask_cube[0].data > 0.5)[0]) < 1:
           raise InputError(f'We expect mask values to start at 1 your mask has no values above 0.5')

        if mask_cube[0].header['NAXIS1'] != cube[0].header['NAXIS1'] or \
           mask_cube[0].header['NAXIS2'] != cube[0].header['NAXIS2'] or \
           mask_cube[0].header['NAXIS3'] != cube[0].header['NAXIS3']:
           raise InputError(f'Your mask {mask_cube} and cube {filename} do not have the same dimensions')
        with np.errstate(invalid='ignore', divide='ignore'):
            cube[0].data[mask_cube[0].data < 0.5] = float('NaN')
        mask_cube.close()
    else:
        if not level:
            level = threshold*np.mean([np.nanstd(cube[0].data[0:2,:,:]),np.nanstd(cube[0].data[-3:-1,:,:])])
        with np.errstate(invalid='ignore', divide='ignore'):
            cube[0].data[cube[0].data < level] = float('NaN')
    try:
        if cube[0].header['CUNIT3'].lower().strip() == 'm/s':
            print_log(f"We convert your m/s to km/s")
            cube[0].header['CUNIT3'] = 'km/s'
            cube[0].header['CDELT3'] = cube[0].header['CDELT3']/1000.
            cube[0].header['CRVAL3'] = cube[0].header['CRVAL3']/1000.
        elif cube[0].header['CUNIT3'].lower().strip() == 'km/s':
            pass
        else:
            print_log(f"Your Velocity unit {cube[0].header['CUNIT3']} is weird. Your units could be off", log)
    except KeyError:
        print_log(f"Your CUNIT3 is missing, that is bad practice. We'll add a blank one but we're not guessing the value", log)
        cube[0].header['CUNIT3'] = 'Unknown'
    #Make a 2D header to use
    hdr2D = copy.deepcopy(cube[0].header)
    hdr2D.remove('NAXIS3')
    hdr2D['NAXIS'] = 2
    # removing the third axis means we cannot correct for varying platescale, Sofia does so this is and issue so let's not do this
    hdr2D.remove('CDELT3')
    hdr2D.remove('CTYPE3')
    hdr2D.remove('CUNIT3')
    hdr2D.remove('CRPIX3')
    hdr2D.remove('CRVAL3')

    # we need a moment 0 for the moment 2 as well
    if 0 in moments:
        hdr2D['BUNIT'] = f"{cube[0].header['BUNIT']}*{cube[0].header['CUNIT3']}"
        moment0 = np.nansum(cube[0].data, axis=0) * cube[0].header['CDELT3']
        moment0[np.invert(np.isfinite(moment0))] = float('NaN')
        hdr2D['DATAMAX'] = np.nanmax(moment0)
        hdr2D['DATAMIN'] = np.nanmin(moment0)
        fits.writeto(f"{output_directory}/{output_name}_mom0.fits",moment0,hdr2D,overwrite = overwrite)
    if 1 in moments or 2 in moments:
        zaxis = cube[0].header['CRVAL3'] + (np.arange(cube[0].header['NAXIS3'])+1 \
              - cube[0].header['CRPIX3']) * cube[0].header['CDELT3']
        c=np.transpose(np.resize(zaxis,[cube[0].header['NAXIS1'],cube[0].header['NAXIS2'],len(zaxis)]),(2,1,0))
        hdr2D['BUNIT'] = f"{cube[0].header['CUNIT3']}"
        # Remember Python is stupid so z,y,x
        with np.errstate(invalid='ignore', divide='ignore'):
            moment1 = np.nansum(cube[0].data*c, axis=0)/ np.nansum(cube[0].data, axis=0)
        moment1[np.invert(np.isfinite(moment1))] = float('NaN')
        hdr2D['DATAMAX'] = np.nanmax(moment1)
        hdr2D['DATAMIN'] = np.nanmin(moment1)
        if 1 in moments:
            fits.writeto(f"{output_directory}/{output_name}_mom1.fits",moment1,hdr2D,overwrite = overwrite)
        if 2 in moments:
            d = c - np.resize(moment1,[len(zaxis),cube[0].header['NAXIS2'],cube[0].header['NAXIS1']])
            with np.errstate(invalid='ignore', divide='ignore'):
                moment2 = np.sqrt(np.nansum(cube[0].data*d**2, axis=0)/ np.nansum(cube[0].data, axis=0))
            moment2[np.invert(np.isfinite(moment1))] = float('NaN')
            hdr2D['DATAMAX'] = np.nanmax(moment2)
            hdr2D['DATAMIN'] = np.nanmin(moment2)
            fits.writeto(f"{output_directory}/{output_name}_mom2.fits",moment2,hdr2D,overwrite = overwrite)
    cube.close()

moments.__doc__ =f'''
 NAME:
    make_moments

 PURPOSE:
    Make the moment maps

 CATEGORY:
    Spectral line cube manipulations.

 INPUTS:
    filename = input file name


 OPTIONAL INPUTS:
    mask = name of the cube to be used as a mask

    debug = False

    moments = [0,1,2]
    moment maps to create

    overwrite = False
    overwrite existing maps

    level=None
    cutoff level to use, if set the mask will not be used

    velocity_unit= none
    velocity unit of the input cube

    threshold = 3.
    Cutoff level in terms of sigma, if used the std in in the first two and last channels in the cube is measured and multiplied.

    log = None
    Name for a logging file

    output_directory = None
    Name of the directory where to put the created maps. If none the current working directory is used.

    output_name = None
    Base name for output maps i.e. maps are output_name_mom#.fits with # number of the moments
    default is filename -.fits

 OUTPUTS:

 OPTIONAL OUTPUTS:

 PROCEDURES CALLED:
    Unspecified

 NOTE:
'''


def main(argv):
    if '-v' in argv or '--version' in argv:
        print(f"This is version {make_moments.__version__} of the program.")
        sys.exit()

    if '-h' in argv or '--help' in argv:
        print('''
Use make_moments in this way:
make_moments -c inputfile.yml   where inputfile is a yaml config file with the desired input settings.
make_moments -h print this message
make_moments -e prints a yaml file (defaults.yml) with the default setting in the current working directory.
in this file values designated ??? indicated values without defaults.



All config parameters can be set directly from the command line by setting the correct parameters, e.g:
make_moments filename=cube.fits mask=mask.fits to make moment maps of the file cube.fits where the maps are masked with mask.fits
''')
        sys.exit()

    cfg = OmegaConf.structured(defaults)
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

    if not cfg.mask and not cfg.level and not cfg.threshold:
        print_log(f'''You have to specify a mask, cutoff level (in cube units), or threshold (in sigma) to mask the cube with''')
        sys.exit(1)

    moments(filename = cfg.filename, mask = cfg.mask, moments = cfg.moments,
                     overwrite = cfg.overwrite, level= cfg.level,velocity_unit= cfg.velocity_unit, threshold = cfg.threshold,
                      debug = cfg.debug, log=cfg.log,output_directory = cfg.output_directory,output_name = cfg.output_name)

if __name__ =="__main__":
    main()
