import numpy as _np
from . import custom_math_priors as _cmp
import copy as _copy

# PBH merger rate densigy ==========================================
from julia.api import Julia
jl = Julia(compiled_modules=False)
jl.eval('include("/home/czc/projects/working/pbh/merger_history_gw190521/code/merger_rate.jl")')

mergerRateDensity1st_log = jl.eval('mergerRateDensity1st_log')
mergerRateDensity2nd_log = jl.eval('mergerRateDensity2nd_log')
mergerRateDensity1st_power = jl.eval('mergerRateDensity1st_power')
mergerRateDensity2nd_power = jl.eval('mergerRateDensity2nd_power')
mergerRateDensity1st_CC = jl.eval('mergerRateDensity1st_CC')
mergerRateDensity2nd_CC = jl.eval('mergerRateDensity2nd_CC')
mergerRateDensity1st_bpower = jl.eval('mergerRateDensity1st_bpower')
mergerRateDensity2nd_bpower = jl.eval('mergerRateDensity2nd_bpower')

class population_prior(object):

    def __init__(self, cosmo, name, hyper_params_dict):

        self.name = name
        self.hyper_params_dict = _copy.deepcopy(hyper_params_dict)
        self.cosmo = _copy.deepcopy(cosmo)
        dist = {}

    def log_prob(self, ms1, ms2, zs):
        hyper_params_dict = self.hyper_params_dict
        if self.name == 'BBH-mass_powerlaw-z_powerlaw':
            alpha = hyper_params_dict['alpha']
            beta = hyper_params_dict['beta']
            mmin = hyper_params_dict['mmin']
            mmax = hyper_params_dict['mmax']
            gamma = hyper_params_dict["gamma"]
            R0 = hyper_params_dict['R0']

            # Define the prior on the masses with a truncated powerlaw as in Eq.33,34,35 on the tex document
            dist = {'mass_1': _cmp.PowerLaw_math(alpha=-alpha, min_pl=mmin, max_pl=mmax),
                    'mass_2': _cmp.PowerLaw_math(alpha=beta, min_pl=mmin, max_pl=mmax)}

            to_ret = _np.log(self.cosmo.dVc_by_dz(zs)) - \
                _np.log1p(zs) + gamma * _np.log1p(zs)

            to_ret += _np.log(R0) + dist['mass_1'].log_prob(
                ms1) + dist['mass_2'].log_conditioned_prob(ms2, mmin * _np.ones_like(ms1), ms1)

        elif self.name == "PBH-lognormal-1st":
            hyper_params_dict = self.hyper_params_dict
            mc = hyper_params_dict['mc']
            σc = hyper_params_dict['σc']
            log_fpbh = hyper_params_dict['log_fpbh']

            to_ret = _np.log(self.cosmo.dVc_by_dz(zs)) - _np.log1p(zs)

            # to_ret += (-34.0 / 37.0) * (_np.log(self.cosmo.t_at_z(zs)) - _np.log(self.cosmo.t_at_z(1e-4))) + \
            #     _np.log(mergerRateDensity1st_log(mc, σc, log_fpbh, ms1, ms2))

            _zs = self.cosmo.t_at_z(zs) / self.cosmo.t_at_z(1e-4)
            R1 = _zs ** (-34.0 / 37.0) * \
                mergerRateDensity1st_log(mc, σc, log_fpbh, ms1, ms2)
            to_ret += _np.log(R1)

        elif self.name == "PBH-lognormal-2nd":
            hyper_params_dict = self.hyper_params_dict
            mc = hyper_params_dict['mc']
            σc = hyper_params_dict['σc']
            log_fpbh = hyper_params_dict['log_fpbh']

            to_ret = _np.log(self.cosmo.dVc_by_dz(zs)) - _np.log1p(zs)

            _zs = self.cosmo.t_at_z(zs) / self.cosmo.t_at_z(1e-4)

            R1 = _zs ** (-34.0 / 37.0) * \
                mergerRateDensity1st_log(mc, σc, log_fpbh, ms1, ms2)
            R2 = _zs ** (-31.0 / 37.0) * \
                mergerRateDensity2nd_log(mc, σc, log_fpbh, ms1, ms2)

            to_ret += _np.log(R1 + R2)

        elif self.name == "PBH-power-1st":
            hyper_params_dict = self.hyper_params_dict
            α = hyper_params_dict['α']
            M = hyper_params_dict['M']
            log_fpbh = hyper_params_dict['log_fpbh']

            to_ret = _np.log(self.cosmo.dVc_by_dz(zs)) - _np.log1p(zs)

            _zs = self.cosmo.t_at_z(zs) / self.cosmo.t_at_z(1e-4)
            R1 = _zs ** (-34.0 / 37.0) * \
                mergerRateDensity1st_power(α, M, log_fpbh, ms1, ms2)
            to_ret += _np.log(R1)

        elif self.name == "PBH-power-2nd":
            hyper_params_dict = self.hyper_params_dict
            α = hyper_params_dict['α']
            M = hyper_params_dict['M']
            log_fpbh = hyper_params_dict['log_fpbh']

            to_ret = _np.log(self.cosmo.dVc_by_dz(zs)) - _np.log1p(zs)

            _zs = self.cosmo.t_at_z(zs) / self.cosmo.t_at_z(1e-4)

            R1 = _zs ** (-34.0 / 37.0) * \
                mergerRateDensity1st_power(α, M, log_fpbh, ms1, ms2)
            R2 = _zs ** (-31.0 / 37.0) * \
                mergerRateDensity2nd_power(α, M, log_fpbh, ms1, ms2)

            to_ret += _np.log(R1 + R2)

        elif self.name == "PBH-CC-1st":
            hyper_params_dict = self.hyper_params_dict
            α = hyper_params_dict['α']
            Mf = hyper_params_dict['Mf']
            log_fpbh = hyper_params_dict['log_fpbh']

            to_ret = _np.log(self.cosmo.dVc_by_dz(zs)) - _np.log1p(zs)

            _zs = self.cosmo.t_at_z(zs) / self.cosmo.t_at_z(1e-4)
            R1 = _zs ** (-34.0 / 37.0) * \
                mergerRateDensity1st_CC(α, Mf, log_fpbh, ms1, ms2)
            to_ret += _np.log(R1)

        elif self.name == "PBH-CC-2nd":
            hyper_params_dict = self.hyper_params_dict
            α = hyper_params_dict['α']
            Mf = hyper_params_dict['Mf']
            log_fpbh = hyper_params_dict['log_fpbh']

            to_ret = _np.log(self.cosmo.dVc_by_dz(zs)) - _np.log1p(zs)

            _zs = self.cosmo.t_at_z(zs) / self.cosmo.t_at_z(1e-4)

            R1 = _zs ** (-34.0 / 37.0) * \
                mergerRateDensity1st_CC(α, Mf, log_fpbh, ms1, ms2)
            R2 = _zs ** (-31.0 / 37.0) * \
                mergerRateDensity2nd_CC(α, Mf, log_fpbh, ms1, ms2)

            to_ret += _np.log(R1 + R2)

        elif self.name == "PBH-bpower-1st":
            hyper_params_dict = self.hyper_params_dict
            ms = hyper_params_dict['ms']
            α1 = hyper_params_dict['α1']
            α2 = hyper_params_dict['α2']
            log_fpbh = hyper_params_dict['log_fpbh']

            to_ret = _np.log(self.cosmo.dVc_by_dz(zs)) - _np.log1p(zs)

            _zs = self.cosmo.t_at_z(zs) / self.cosmo.t_at_z(1e-4)
            R1 = _zs ** (-34.0 / 37.0) * \
                mergerRateDensity1st_bpower(ms, α1, α2, log_fpbh, ms1, ms2)
            to_ret += _np.log(R1)

        elif self.name == "PBH-bpower-2nd":
            hyper_params_dict = self.hyper_params_dict
            ms = hyper_params_dict['ms']
            α1 = hyper_params_dict['α1']
            α2 = hyper_params_dict['α2']
            log_fpbh = hyper_params_dict['log_fpbh']

            to_ret = _np.log(self.cosmo.dVc_by_dz(zs)) - _np.log1p(zs)

            _zs = self.cosmo.t_at_z(zs) / self.cosmo.t_at_z(1e-4)

            R1 = _zs ** (-34.0 / 37.0) * \
                mergerRateDensity1st_bpower(ms, α1, α2, log_fpbh, ms1, ms2)
            R2 = _zs ** (-31.0 / 37.0) * \
                mergerRateDensity2nd_bpower(ms, α1, α2, log_fpbh, ms1, ms2)

            to_ret += _np.log(R1 + R2)

        to_ret[_np.isnan(to_ret)] = -_np.inf

        return to_ret
