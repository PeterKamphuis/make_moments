# make_moments
A stand alone version of the pyFAT make_moments and create xv_diagram function. The function can be imported to your python code or you can run it by itself from command line after a pip install:

  pip install make-moments


For moment maps:

use as:
  make_moments filename=Cub.fits mask=mask.fits

If the maps already exist add overwrite=True

For an overview of the possible input type 'make_moments -e' to print the default yaml file.

to configure setting from a yaml file

  make_moments -c my_input_file.yml

To use in python script:

from make_moments.functions import moments

And then use the moments() function

For PV-diagrams:

create_PV_diagram filename=Cub.fits PA=16.

to configure setting from a yaml file

  create_PV_diagram -c my_input_file.yml

To use in python script:

from make_moments.functions import pv_diagram

And then use the pv_diagram() function
