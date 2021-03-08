import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import numpy as np
# from subprocess import *
import bisect

# from astropy import constants as const
# from astropy import units as u
from astropy.io import fits
from astropy.table import Table
from astropy.modeling import models
from astropy import units as u
import astropy.constants as const
from astropy.nddata import StdDevUncertainty
from astropy.visualization import quantity_support
quantity_support()

from specutils.analysis import snr, correlation, equivalent_width, template_logwl_resample
from specutils.manipulation import box_smooth, gaussian_smooth, trapezoid_smooth, extract_region
from specutils.spectra import Spectrum1D, SpectralRegion, SpectrumCollection
from specutils.fitting import fit_generic_continuum, find_lines_derivative, fit_lines

def plot_SDSSspec(wavelength, flux, specType="", title="", xmin=3800, xmax=10000, spec_box_size=10, kind="object"):
    line_list_dir = "/Users/benjaminroulston/Dropbox/Research/TDSS/General/SpecLineLists/"

    fig = plt.figure(figsize=(15, 5))

    major_tick_space = 1000
    minor_tick_space = 100

    flux = removeSdssStitchSpike(wavelength, flux)

    if specType != "":
        line_lis_all = np.genfromtxt(f"{line_list_dir}spec_types/{specType}star_lines.list", comments='#', dtype="S")

        lineList_wavelength = np.float64(line_lis_all[:, 0])
        lineList_labels = np.empty(lineList_wavelength.size, dtype="U60")
        for ii in range(lineList_wavelength.size):
            lineList_labels[ii] = line_lis_all[ii, 1].decode(encoding="utf-8", errors="strict")

    trim_spectrum_left = 10  # number of pixels to trim from left side
    smooth_flux = smooth(flux[trim_spectrum_left:], spec_box_size)
    smooth_wavelength = smooth(wavelength[trim_spectrum_left:], spec_box_size)

    plotted_region = np.where((smooth_wavelength >= xmin) & (smooth_wavelength <= xmax))[0]
    ymin = smooth_flux[plotted_region].min()
    ymax = smooth_flux[plotted_region].max()

    if kind == "object":
        plt.plot(smooth_wavelength, smooth_flux, color='black', linewidth=0.5)
    elif kind == "model":
        plt.plot(smooth_wavelength, smooth_flux, color='red', linewidth=0.5)

    plt_ax = plt.gca()
    plt_ax.set_xlabel("Wavelength [\AA]")  # , fontdict=font)
    plt_ax.set_ylabel("Flux Density [10$^{-17}$ erg s$^{-1}$ cm$^{-2}$ \AA$^{-1}$]")  # , fontdict=font)

    if title != "":
        plt_ax.set_title(title)

    plt_ax.set_xlim([xmin, xmax])
    plt_ax.set_ylim([ymin, ymax])

    plt_ax.xaxis.set_major_locator(ticker.MultipleLocator(major_tick_space))
    plt_ax.xaxis.set_minor_locator(ticker.MultipleLocator(minor_tick_space))
    if specType != "":
        for ll in range(lineList_wavelength.size):
            plt_ax.axvline(x=lineList_wavelength[ll], ls='dashed', c='k', alpha=0.1)
            plt_ax.text(lineList_wavelength[ll] + 20.0, plt_ax.get_ylim()[0] + 0.50, lineList_labels[ll], rotation=90, color='k', alpha=0.2)

    plt.tight_layout()


def removeSdssStitchSpike(wavelength, flux):
    """
    All SDSS spectrum have a spike in the spectra between 5569 and 5588 angstroms where
    the two detectors meet. This method will remove the spike at that point by linearly
    interpolating across that gap.
    """
    # Make a copy so as to not alter the original, passed in flux
    flux = flux.copy()
    # Search for the indices of the bounding wavelengths on the spike. Use the
    # fact that the wavelength is an array in ascending order to search quickly
    # via the searchsorted method.
    lower = np.searchsorted(wavelength, 5569)
    upper = np.searchsorted(wavelength, 5588)
    # Define the flux in the stitch region to be linearly interpolated values between
    # the lower and upper bounds of the region.
    flux[lower:upper] = np.interp(wavelength[lower:upper],
                                  [wavelength[lower], wavelength[upper]],
                                  [flux[lower], flux[upper]])
    return flux


def smooth(y, box_pts):
    box = np.ones(box_pts) / box_pts
    y_smooth = np.convolve(y, box, mode='valid')
    return y_smooth


def interpOntoGrid(wavelength, flux):
    """
    Description:
        A method to put the spectrum flux and variance onto the same
        wavelength grid as the templates (5 km/s equally spaced bins)
    """
    # Interpolate flux and variance onto the wavelength grid

    waveStart = 3_550
    waveEnd = 10_500
    waveNum = 65_000
    # 65,000 wavelengths gives 5km/s resolution across this region
    # dv = 2.9979e5 * (np.diff(waveGrid) / waveGrid[1:])
    waveGrid = np.logspace(np.log10(waveStart), np.log10(waveEnd), num=waveNum)

    interpFlux = np.interp(waveGrid, wavelength, flux, right=np.nan, left=np.nan)

    # cut the grids off at 3650 and 10200 like the templates
    startIndex = bisect.bisect_right(waveGrid, 3650)
    stopIndex = bisect.bisect_right(waveGrid, 10200)

    wavelength = waveGrid[startIndex:stopIndex]
    flux = interpFlux[startIndex:stopIndex]

    return wavelength, flux


def SDSSfitsSpec_to_csv(specFilename):
    spec_Table = SDSSfitsSpec_to_Table(specFilename)
    spec_Table.write(f"{specFilename}.csv", format='ascii.csv')


def SDSSfitsSpec_to_Table(specFilename):
    spec = fits.open(f"{specFilename}.fits")
    wavelength = 10.0**spec[1].data['loglam']
    flux = spec[1].data['flux']
    error = np.sqrt(1.0 / spec[1].data['ivar'])

    spec_Table = Table()
    spec_Table.add_column(wavelength, name="wavelength")
    spec_Table.add_column(flux, name="flux")
    spec_Table.add_column(error, name="error")

    return spec_Table


def line_vel(lam, lam0):
    dlam = lam - lam0
    return ((dlam / lam0) * const.c).to(u.km / u.s)


def plot_dynamical_spectrum(spec_collection, phase, lineregion, linecenter, contregion=None, normalize=False, interp=False):
    phase = phase[np.argsort(phase)]
    sub_spectrum_size = extract_region(spec_collection[0], lineregion).flux.size
    if normalize:
        raw_data = np.zeros((sub_spectrum_size, len(spec_collection))) * np.nan * u.dimensionless_unscaled
    else:
        raw_data = np.zeros((sub_spectrum_size, len(spec_collection))) * np.nan * spec_collection[0].flux.unit

    for ii, spec in enumerate(spec_collection):
        if normalize:
            cont_spectrum = extract_region(spec, contregion)
            g1_fit = fit_generic_continuum(cont_spectrum)
            continuum_fitted = g1_fit(cont_spectrum.wavelength)
            norm_spec = cont_spectrum / continuum_fitted
            sub_spectrum = extract_region(norm_spec, lineregion)
        else:
            sub_spectrum = extract_region(spec, lineregion)

        raw_data[:, ii] = sub_spectrum.flux

    if interp:
        interp_flux = np.zeros((sub_spectrum_size, 1000)) * np.nan * raw_data.unit
        interp_phase = np.linspace(0, 1, 1000)
        for ii in range(sub_spectrum_size):
            interp_flux[ii, :] = np.interp(interp_phase, phase, raw_data[ii, :])
    else:
        interp_phase = np.linspace(0, 1, 50)
        interp_flux = np.zeros((sub_spectrum_size, interp_phase.size)) * np.nan * raw_data.unit
        for ii in range(phase.size):
            phase_index = np.argmin((interp_phase - phase[ii])**2)
            interp_flux[:, phase_index] = raw_data[:, ii]

    fig = plt.figure(figsize=(5, 8), dpi=600)
    ax = plt.gca()

    wave_xlim0 = sub_spectrum.wavelength.min().value
    wave_xlim1 = sub_spectrum.wavelength.max().value
    flux_ylim0 = np.nanmin(interp_flux.value)
    flux_ylim1 = np.nanmax(interp_flux.value)

    IM = ax.imshow(interp_flux.value.T, aspect='auto', origin='lower', extent=[wave_xlim0, wave_xlim1, 0, 1], vmin=flux_ylim0, vmax=flux_ylim1)
    if linecenter is not None:
        ax.axvline(x=linecenter.value, c='r', alpha=0.25, ls='dashed')
    ax.set_xlabel("Wavelength \AA")
    ax.set_ylabel(f"Phase")
    # plt.ylabel(f"Flux Density [{sub_spectrum.flux.unit.to_string('latex_inline')}]")

    cbar = fig.colorbar(IM)
    if normalize:
        cbar.ax.set_ylabel(f"Continuum Normalized Flux Density")
    else:
        cbar.ax.set_ylabel(f"Flux Density [{sub_spectrum.flux.unit.to_string('latex_inline')}]")

    vel_xlim0 = line_vel(ax.get_xlim()[0] * u.Angstrom, linecenter).value
    vel_xlim1 = line_vel(ax.get_xlim()[1] * u.Angstrom, linecenter).value

    ax2 = ax.twiny()
    ax2.imshow(interp_flux.value.T, aspect='auto', origin='lower', extent=[vel_xlim0, vel_xlim1, 0, 1], vmin=flux_ylim0, vmax=flux_ylim1)
    if linecenter is not None:
        ax2.axvline(x=0., c='r', alpha=0.25, ls='dashed')
    ax2.set_xlabel('RV [km s$^{-1}$]')

    # ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
    # ax.xaxis.set_minor_locator(ticker.MultipleLocator(.5))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.05))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.01))

    # ax2.xaxis.set_major_locator(ticker.MultipleLocator(100))
    # ax2.xaxis.set_minor_locator(ticker.MultipleLocator(10))

    plt.tight_layout()
    return fig
