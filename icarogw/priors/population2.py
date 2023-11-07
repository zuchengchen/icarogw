import numpy as np
from numpy import log, exp, sqrt, pi
import numpy as _np
from . import custom_math_priors as _cmp
import copy as _copy
from . import redshift as rs


def Pm_log(m, mc, σc):
    """
    lognormal mass function
    """
    exp_value = -log(m / mc) ** 2 / (2 * σc**2)
    return 1 / (sqrt(2 * pi) * σc * m) * exp(exp_value)


def mergerRateDensity_log(mc, σc, log_fpbh, m1, m2):
    """See eq.(2) of https://arxiv.org/pdf/2108.11740v2.pdf"""
    fpbh = 10**log_fpbh
    σeq = 5e-3
    p10 = Pm_log(m1, mc, σc) / m1
    p20 = Pm_log(m2, mc, σc) / m2

    return (
        2.8e6
        * fpbh**2
        * (0.7 * fpbh**2 + σeq**2) ** (-21 / 74)
        * np.minimum(p10, p20)
        * (p10 + p20)
        * (m1 * m2) ** (3 / 37)
        * (m1 + m2) ** (36 / 37)
    )


class population_prior(object):
    def __init__(self, cosmo, name, hyper_params_dict):
        self.name = name
        self.hyper_params_dict = _copy.deepcopy(hyper_params_dict)
        self.cosmo = _copy.deepcopy(cosmo)
        dist = {}

    def log_prob(self, ms1, ms2, zs):
        hyper_params_dict = self.hyper_params_dict

        if self.name == "PBH-lognormal_BBH-mass_powerlaw_gaussian-z_madau":
            mc = hyper_params_dict["mc"]
            σc = hyper_params_dict["σc"]
            log_fpbh = hyper_params_dict["log_fpbh"]

            alpha = hyper_params_dict["alpha"]
            beta = hyper_params_dict["beta"]
            mmin = hyper_params_dict["mmin"]
            mmax = hyper_params_dict["mmax"]
            mu_g = hyper_params_dict["mu_g"]
            sigma_g = hyper_params_dict["sigma_g"]
            lambda_peak = hyper_params_dict["lambda_peak"]
            delta_m = hyper_params_dict["delta_m"]
            gamma = hyper_params_dict["gamma"]
            zp = hyper_params_dict["zp"]
            kappa = hyper_params_dict["kappa"]
            R0_abh = hyper_params_dict["R0_abh"]

            _zs = self.cosmo.t_at_z(zs) / self.cosmo.t_at_z(1e-4)
            to_ret = _np.log(self.cosmo.dVc_by_dz(zs)) - _np.log1p(zs)

            # pbh
            logRpbh = (-34.0 / 37.0) * _np.log(_zs) + _np.log(
                mergerRateDensity_log(mc, σc, log_fpbh, ms1, ms2)
            )

            # abh
            m1pr = _cmp.PowerLawGaussian_math(
                alpha=-alpha,
                min_pl=mmin,
                max_pl=mmax,
                lambda_g=lambda_peak,
                mean_g=mu_g,
                sigma_g=sigma_g,
                min_g=mmin,
                max_g=mu_g + 5 * sigma_g,
            )
            m2pr = _cmp.PowerLaw_math(alpha=beta, min_pl=mmin, max_pl=m1pr.maximum)

            log_rate_eval = rs.log_madau_rate(gamma, kappa, zp)

            # Smooth the lower end of these distributions
            dist = {
                "mass_1": _cmp.SmoothedProb(
                    origin_prob=m1pr, bottom=mmin, bottom_smooth=delta_m
                ),
                "mass_2": _cmp.SmoothedProb(
                    origin_prob=m2pr, bottom=mmin, bottom_smooth=delta_m
                ),
            }
            logRabh = (
                log_rate_eval(zs)
                + _np.log(R0_abh)
                + dist["mass_1"].log_prob(ms1)
                + dist["mass_2"].log_conditioned_prob(
                    ms2, mmin * _np.ones_like(ms1), ms1
                )
            )

            to_ret += _np.logaddexp(logRpbh, logRabh)

        return to_ret


"""
    Draws a random number from given probability density function.

    Parameters
    ----------
        pdf       -- the function pointer to a probability density function of form P = pdf(x)
        interval  -- the resulting random number is restricted to this interval
        pdfmax    -- the maximum of the probability density function
        integers  -- boolean, indicating if the result is desired as integer
        max_iterations -- maximum number of 'tries' to find a combination of random numbers (rand_x, rand_y) located below the function value calc_y = pdf(rand_x).

    returns a single random number according the pdf distribution.
"""


def draw_random_number_from_pdf(
    pdf, interval, pdfmax=1, integers=False, max_iterations=1000000
):
    for i in range(max_iterations):
        if integers == True:
            rand_x = np.random.randint(interval[0], interval[1])
        else:
            # (b - a) * random_sample() + a
            rand_x = (interval[1] - interval[0]) * np.random.random(1) + interval[0]

        rand_y = pdfmax * np.random.random(1)
        calc_y = pdf(rand_x)

        if rand_y <= calc_y:
            return rand_x

    raise Exception(
        "Could not find a matching random number within pdf in "
        + max_iterations
        + " iterations."
    )


def pdf_v0(v0, v):
    return v**2 * np.exp(-(v**2) / v0**2) / (0.885837118847261 * v0**3)


def get_rand_v(v0):
    interval = [-3 * v0, 3 * v0]

    def pdf_v(v):
        return pdf_v0(v0, v)

    return draw_random_number_from_pdf(pdf_v, interval, 0.84 / v0)


def get_z_doppler(v0, n_ev, n_min):
    """
    n_ev events
    n_min samples for each event

    return 1 + z_doppler
    """
    vs = np.zeros([n_ev, n_min])
    for i in range(n_ev):
        v1 = get_rand_v(v0)[0]
        v1s = np.random.normal(v1, 0.05 * np.abs(v1), n_min)
        vs[i, :] = v1s

    return (1 - vs**2) ** (-1 / 2) * (1 + vs)
