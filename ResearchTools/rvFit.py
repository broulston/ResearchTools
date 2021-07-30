import numpy as np

import matplotlib.pyplot as plt

from scipy.optimize import newton
from scipy.optimize import least_squares
from scipy.optimize import dual_annealing

import astropy.units as u
import astropy.constants as const

from multiprocessing import Pool
import emcee
import corner

from pdb import set_trace as bp


def mean_anomaly(t, t0, P):
    return ((2 * np.pi) / P) * (t - t0)


def f(E, e, M):
     #E = E.value
    return E - e * np.sin(E) - M


# def f_prime(E, e, M):
#     # E = E.value
#     return 1 + e * np.cos(E)


# def f_prime2(E, e, M):
#     # E = E.value
#     return e * np.sin(E)


def true_anomaly(t, P, e, t0):
    M = mean_anomaly(t, t0, P)
    x0 = ((t - t0) / P) % 1
    E = newton(f, x0=x0, args=(e, M))  # .value
    # E = newton(f, x0=0.5, fprime=f_prime, args=(e, M),
    #            maxiter=5000, fprime2=f_prime2)
    e_ratio = ((1 + e) / (1 - e))**-0.5
    tan_true_anomaly = (1 / e_ratio) * np.tan(E / 2)
    return 2 * np.arctan(tan_true_anomaly)


def vr(t, P, e, t0, omega, gamma, asini, K=None):
    theta = true_anomaly(t, P, e, t0) * u.rad
    if K is None:
        K = ((2 * np.pi * asini) / (P * (1 - e**2)**0.5)).to(u.km / u.s)
    return K * (np.cos(theta + omega) + e * np.cos(omega)) + gamma


def vr_m(t, P, m1, m2, e, t0, omega, gamma, sini, K=None):
    theta = true_anomaly(t, P, e, t0) * u.rad
    if K is None:
        K = ((2 * np.pi * const.G)**(1/3) * m2 * sini) / (P**(1/3) * (m1 + m2)**(2/3) * np.sqrt(1 - e**2))
    return K * (np.cos(theta + omega) + e * np.cos(omega)) + gamma


e = 0.7235
P = 0.30235 * u.d
t0 = 57364.32 * u.d
omega = np.radians(65.24542) * u.rad
gamma = 175.34 * u.km / u.s
asini = 1.55 * u.solRad

theta_true = [P.to(u.d).value, e, t0.to(u.d).value, omega.to(u.rad).value,
          gamma.to(u.km/u.s).value, asini.to(u.solRad).value]

t = np.linspace(t0, t0 + 3 * P, 2500)  # * u.d

# K = 100 * u.km / u.s
vrs = vr(t, P, e, t0, omega, gamma, asini)
phase = ((t - t0) / P) % 1

vrs_noise = np.random.normal(loc=vrs, scale=15.0) * u.km / u.s
vrs_error = np.abs(vrs - vrs_noise)
random_index = np.random.randint(0, vrs.size + 1, 25, int)

test_mjd = t[random_index]
test_phase = phase[random_index]
test_Vr = vrs[random_index]
test_Vrerr = vrs_error[random_index]

plt.plot(t, vrs, c='k', lw=1)
plt.errorbar(test_mjd.value, test_Vr.value, test_Vrerr.value, ms=1, c='r', ls='')
plt.xlabel("MJD")
plt.ylabel("RV [km s$^{-1}$]")
plt.show()

plt.plot(phase, vrs, c='k', lw=1)
plt.errorbar(test_phase.value, test_Vr.value, test_Vrerr.value, ms=1, c='r', ls='')
plt.xlabel("Phase")
plt.ylabel("RV [km s$^{-1}$]")
plt.show()

# **************************************************
# **************************************************
# **************************************************


def func(theta):
    RV = test_Vr
    RVerr = test_Vrerr
    P, e, t0, omega, gamma, asini = theta
    P = P * u.d
    e = e
    t0 = t0 * u.d
    omega = omega * u.rad
    gamma = gamma * u.km / u.s
    asini = asini * u.solRad
    model = vr(test_mjd, P, e, t0, omega, gamma, asini)
    diff = RV - model
    chisSq = np.sum((diff / RVerr)**2)  # * (1 / (t.size - 1))
    return chisSq.value

# theta0 = [P0.to(u.d).value, e0, t00.to(u.d).value, omega0.to(u.rad).value,
#           gamma0.to(u.km/u.s).value, asini0.to(u.solRad).value]
#     [P, e, t0, omega, gamma, asini]
lw = [0.302, 0, 57360, 0, 150, 0.5]
up = [0.303, 1, 57370, 2*np.pi, 200, 1000]

# bounds1 = (lw, up)
# res_1 = least_squares(func, theta0, bounds=bounds1)

bounds2 = list(zip(lw, up))
res_2 = dual_annealing(func, bounds2)
# want to fit: P, e, t0, omega, gamma, asini
# theta = [P, e, t0, omega, gamma, asini]
theta_fit = res_2.x

print("True | Fit | Diff")
for ii in range(theta_fit.size):
    print(theta_true[ii], theta_fit[ii], theta_fit[ii] - theta_true[ii])

P_fit, e_fit, t0_fit, omega_fit, gamma_fit, asini_fit = theta_fit
P_fit = P_fit * u.d
e_fit = e_fit
t0_fit = t0_fit * u.d
omega_fit = omega_fit * u.rad
gamma_fit = gamma_fit * u.km / u.s
asini_fit = asini_fit * u.solRad
model_fit = vr(t, P_fit, e_fit, t0_fit, omega_fit, gamma_fit, asini_fit)

model_resid = model_fit - vrs

plt.figure(1)
plt.plot(t, model_fit, c='k', lw=1, label='Model')
plt.plot(t, vrs, c='r', lw=1, alpha=0.5, label='True')
plt.errorbar(test_mjd.value, test_Vr.value, test_Vrerr.value, ms=1, c='r', ls='')
plt.xlabel("MJD")
plt.ylabel("RV [km s$^{-1}$]")
# plt.show()

plt.figure(2)
plt.plot(phase, model_fit, c='k', lw=1)
plt.errorbar(test_phase.value, test_Vr.value, test_Vrerr.value, ms=1, c='r', ls='')
plt.xlabel("Phase")
plt.ylabel("RV [km s$^{-1}$]")
# plt.show()

plt.figure(3)
plt.plot(t, model_resid, c='k', lw=1, label='Model')
plt.xlabel("MJD")
plt.ylabel("RV [km s$^{-1}$]")
plt.show()

# **************************************************
# **************************************************
# **************************************************
# **************************************************
# **************************************************
# **************************************************


def log_likelihood(theta, t, RV, RVerr):
    # RV = RV * u.km / u.s
    # RVerr = RVerr * u.km / u.s
    P, e, t0, omega, gamma, asini = theta
    P = P * u.d
    e = e
    t0 = t0 * u.d
    omega = omega * u.rad
    gamma = gamma * u.km / u.s
    asini = asini * u.solRad
    model = vr(t, P, e, t0, omega, gamma, asini)
    diff = RV - model
    chisSq = np.sum((diff / RVerr)**2)  # * (1 / (t.size - 1))
    #return chisSq
    # return -0.5 * chisSq + np.log(np.sqrt(2 * np.pi) * 15**2)
    return -0.5 * chisSq.value + np.log(np.sqrt(2 * np.pi) * RVerr.value**2)
    # return -0.5 * chisSq - 0.5 * np.log(2 * np.pi) - np.log(RVerr.value)


def log_prior(theta):
    P, e, t0, omega, gamma, asini = theta
    P = P * u.d
    e = e
    t0 = t0 * u.d
    omega = omega * u.rad
    gamma = gamma * u.km / u.s
    asini = asini * u.solRad
    if (0.302*u.d < P < 0.303*u.d) and (0.0 <= e <= 1.0) and (573760*u.d < t0 < 57370*u.d) and (0.0*u.rad < omega < 2*np.pi*u.rad)  and (150*u.km/u.s <= gamma <= 200*u.km/u.s)  and (0*u.km <= asini):
        return 0.0
    return -np.inf


# prior = 1.0/(initial_params_range_high - initial_params_range_low)
# np.log(prior.prod())

def log_probability(theta, t, RV, RVerr):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, t, RV, RVerr)


# theta0 = [0.5*u.d, 0.1, 57362*u.d, np.radians(45)*u.rad, 50*u.km/u.s, 1.0*u.solRad]
# e0 = 0.1
# P0 = 0.30325 * u.d
# t00 = 57362.0 * u.d
# omega0 = np.radians(45) * u.rad
# gamma0 = -50 * u.km / u.s
# asini0 = 1.0 * u.solRad

# theta0 = [P0.to(u.d).value, e0, t00.to(u.d).value, omega0.to(u.rad).value,
#           gamma0.to(u.km/u.s).value, asini0.to(u.solRad).value]
# theta = [P, e, t0, omega, gamma, asini]
theta0 = theta_fit
scale = [0.01, 0.01, 0.1, np.radians(1), 5, 0.1]
nwalkers = 32
ndim = len(theta0)
#initial_state = theta0 + 1e-4 * np.random.randn(nwalkers, ndim)

cov = np.diagflat([scale])
initial_state = np.abs(np.random.multivariate_normal(theta0, cov, size=(nwalkers)))


# sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability,
#                                 args=(test_mjd, test_Vr, test_Vrerr))
# sampler.run_mcmc(initial_state, nsteps=5000, progress=True)

with Pool() as pool:
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability,
                                    args=(test_mjd, test_Vr, test_Vrerr),
                                    pool=pool)
    sampler.run_mcmc(initial_state, nsteps=10000, progress=True)

flat_samples = sampler.get_chain(discard=100, flat=True)

labels = ["P", "e", "t$_0$", "$\omega$", "$\gamma$", "a$\sin{i}$"]
fig = corner.corner(flat_samples, labels=labels,
                    quantiles=[0.16, 0.5, 0.84],
                    show_titles=True)
# plt.savefig("RV_fit.pdf", dpi=600)
plt.show()
# truths=[m_true, b_true, np.log(f_true)])

# **************************************************
# **************************************************
# **************************************************
P0 = 0.30325 * u.d
e0 = 0.1
t00 = 57362.0 * u.d
omega0 = np.radians(45) * u.rad
gamma0 = -50 * u.km / u.s
asini0 = 1.0 * u.solRad

e0 = 0.3
P0 = 0.30235 * u.d
t00 = 57364.32 * u.d
omega0 = np.radians(90) * u.rad
gamma0 = 0.0 * u.km / u.s
asini0 = 1.55 * u.solRad

initial_guess = np.array([P0.to(u.d).value, e0, t00.to(u.d).value,
                          omega0.to(u.rad).value, gamma0.to(u.km / u.s).value,
                          asini0.to(u.solRad).value])
width_to_sample_around = np.array([0.5, 0.1, 100, np.pi, 50, 5])
initial_params_low =  lw  # initial_guess - width_to_sample_around
initial_params_high = up  # initial_guess + width_to_sample_around

initial_state = np.empty((nwalkers, ndim), dtype='float64')
for ii in range(ndim):
    initial_state[:, ii] = np.random.uniform(low=initial_params_low[ii],
                                             high=initial_params_high[ii],
                                             size=nwalkers)
initial_state = np.abs(initial_state)

with Pool() as pool:
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability,
                                    args=(test_mjd, test_Vr.value, test_Vrerr.value),
                                    pool=pool)
    sampler.run_mcmc(initial_state, nsteps=10000, progress=True)

flat_samples = sampler.get_chain(discard=100, flat=True)

labels = ["P", "e", "t$_0$", "$\omega$", "$\gamma$", "a$\sin{i}$"]
fig = corner.corner(flat_samples, labels=labels,
                    quantiles=[0.16, 0.5, 0.84],
                    show_titles=True)
# plt.savefig("RV_fit.pdf", dpi=600)
plt.show()
# truths=[m_true, b_true, np.log(f_true)])
