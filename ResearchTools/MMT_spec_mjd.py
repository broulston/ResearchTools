import numpy as np
from astropy.io import fits
import sys
import matplotlib.pyplot as plt
from subprocess import *
import os

filename = sys.argv[1]
data = fits.open(filename)

mjd = data[0].header['mjd']

new_filename = os.path.splitext(filename)[0]+"_"+str(mjd)+os.path.splitext(filename)[1]
save_filename = os.path.splitext(filename)[0]+"_"+str(mjd)
check_output(["mv", filename, new_filename])