# coding=UTF-8

import numpy as np
#import astropy.units as u
from astropy.io import fits
import sys
import matplotlib.pyplot as plt
#import astropy.coordinates as coord
#from astropy.coordinates import SkyCoord, Distance
from subprocess import *
import os

def smoothFlux(flux):
    # Define our own nancumsum method since numpy's nancumsum was only
    # added recently and not everyone will have the latest numpy version
    def nancumsum(x):
        return np.ma.masked_array(x, mask = (np.isnan(x)|np.isinf(x))).cumsum().filled(np.nan)
    
    # Simply the flux, convolved with a boxcar function to smooth it out.
    # A potential failing of this method is the case where there are a
    # small number of flux values (in the hundreds) but that seems so
    # unlikely, it isn't going to be handled.
    N = max(int(len(flux)/600), 50)  # Smoothing factor, Higher value = more smoothing
    cumsum = nancumsum(np.insert(flux,0,0))
    smoothFlux = (cumsum[N:] - cumsum[:-N]) / N
    smoothFlux = np.append(flux[:int(np.floor((N-1)/2))], smoothFlux)
    smoothFlux = np.append(smoothFlux, flux[-int(np.floor(N/2)):])
    return smoothFlux

filename = sys.argv[1]
print(filename)
data = fits.open(filename)

flux = data[0].data[0, :]
err = data[0].data[1, :]

mjd = data[0].header['mjd']
CRVAL1 = data[0].header['CRVAL1']
CDELT1 = data[0].header['CDELT1']
NAXIS1 = data[0].header['NAXIS1']
flux_units = data[0].header['BUNIT']

wavelength = [CRVAL1]
for ii in range(NAXIS1):
    if ii==0:
        pass
    else:
        wavelength.append(wavelength[ii-1] + CDELT1)

wavelength = np.array(wavelength)

flux_non_nan_index = np.isfinite(flux)

spec = np.stack((wavelength[flux_non_nan_index], flux[flux_non_nan_index], err[flux_non_nan_index]), axis=1)
header = "wavelength[AA], flux"+flux_units+", err"+flux_units

if "-add_mjd" in sys.argv:
    new_filename = os.path.splitext(filename)[0]+"_"+str(mjd)+os.path.splitext(filename)[1]
    save_filename = os.path.splitext(filename)[0]+"_"+str(mjd)
    check_output(["mv", filename, new_filename])
else:
    save_filename = os.path.splitext(filename)[0]

np.savetxt(save_filename+".csv", spec, delimiter=",", header=header, fmt="%f, %f, %f")


if "-plot" in sys.argv:
    plt.plot(wavelength, flux, c='k', lw=1)
    plt.plot(wavelength, err, c='k', lw=1, alpha=0.5)

    smoothedFlux = smoothFlux(flux)
    plt.plot(wavelength, smoothedFlux, c='r', lw=1)
    plt.xlabel("Wavelength [\AA]")
    plt.ylabel("f$_{\\lambda}$ [$"+flux_units+"$]")
    plt.tight_layout()
    plt.savefig(save_filename+".pdf", dpi=600)
    plt.show()
    plt.clf()
    plt.close()
