# moments
A stand alone version of the pyFAT make_moments function. The function can be imported to your python code or you can run it by itself from command line after a pip install:

  pip install make_moments

use as:
  make_moments filename=Cub.fits mask=mask.fits

If the maps already exist add overwrite=True

For an overview of the possible inout type make_moments -e to print the default yaml file.

to configure setting from a yaml file 

  make_moments -c my_input_file.yml
