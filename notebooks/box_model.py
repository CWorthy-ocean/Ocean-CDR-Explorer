import time

import numpy as np
import PyCO2SYS as csys

import xarray as xr

from scipy.integrate import solve_ivp
from scipy.optimize import minimize

import co2calc

# gas exchange coefficient
xkw_coef_cm_per_hr = 0.251

# (cm/hr s^2/m^2) --> (m/s s^2/m^2)
xkw_coef = xkw_coef_cm_per_hr * 3.6e-5

# reference density
rho_ref = 1026.0  # kg/m^3

# CO2SYS parameter types
par_type = dict(ALK=1, DIC=2, pH=3, pCO2=5)  # using fugacity in place of pCO2


def gas_transfer_velocity(u10, temp):
    """
    Compute gas transfer velocity.

    Parameters
    ----------

    u10 : numeric
      Wind speed [m/s]

    temp : numeric
      Sea surface Temperature [°C]

    Returns
    -------

    k : numeric
      Gas transfer velocity [m/s]
    """
    sc = schmidt_co2(temp)
    u10sq = u10 * u10
    return xkw_coef * u10sq * (np.sqrt(660.0 / sc))


def schmidt_co2(sst):
    """
    Compute Schmidt number of CO2 in seawater as function of SST.

    Range of validity of fit is -2:40
    Reference:
        Wanninkhof 2014, Relationship between wind speed
          and gas exchange over the ocean revisited,
          Limnol. Oceanogr.: Methods, 12,
          doi:10.4319/lom.2014.12.351

    Check value at 20°C = 668.344

    Parameters
    ----------

    sst : numeric
      Temperature

    Returns
    -------

    sc : numeric
      Schmidt number
    """
    a = 2116.8
    b = -136.25
    c = 4.7353
    d = -0.092307
    e = 0.0007555

    # enforce bounds
    sst_loc = np.where(sst < -2.0, -2.0, np.where(sst > 40.0, 40.0, sst))

    return a + sst_loc * (b + sst_loc * (c + sst_loc * (d + sst_loc * e)))


def mmolm3_to_µmolkg(value):
    """Convert from volumetric to gravimetric units"""
    return value / rho_ref * 1e3


def µmolkg_to_mmolm3(value):
    """Convert from gravimetric to volumetric units"""
    return value * rho_ref * 1e-3


def molkg_to_mmolm3(value):
    """Convert from gravimetric to volumetric units"""
    return value * rho_ref * 1e3


def calc_csys(dic, alk, salt, temp, sio3=0.0, po4=0.0, flavor="ocmip"):
    """Return a CO2 Solver Object"""
    csys_solver_avail = dict(
        pyco2sys=_calc_csys_pyco2sys,
        ocmip=_calc_csys_ocmip_ish,
    )
    assert flavor in csys_solver_avail
    return csys_solver_avail[flavor](dic, alk, salt, temp, sio3, po4)


class _calc_csys_ocmip_ish(object):
    """
    Solve carbonate system chemistry using OCMIP-derived code.
    (Volumetric units)
    """

    def __init__(self, dic, alk, salt, temp, sio3=0.0, po4=0.0):
        self.dic = dic
        self.alk = alk

        self.salt = salt
        self.temp = temp
        self.sio3 = sio3
        self.po4 = po4

        self.equil_constants = co2calc.co2_eq_const(salt, temp)
        self.co2sol = co2calc.co2sol(salt, temp)  # mmol/m^3/atm
        self._pH0 = 8.1
        self.solve_co2(dic, alk)

    def solve_co2(self, dic, alk):
        # solve carbonate system

        self.dic = dic
        self.alk = alk

        self.co2aq, self.pH = co2calc.calc_csys_iter(
            dic,
            alk,
            self.salt,
            self.temp,
            self.po4,
            self.sio3,
            pH0=self._pH0,
            thermodyn=self.equil_constants,
        )
        self._pH0 = self.pH

    @property
    def pco2(self):
        return 1.0e6 * self.co2aq / self.co2sol

    def calc_new_dic_w_oae(self, new_alk):
        """Compute the new DIC concentration in mmol/m^3
        after alkalinity addition assuming pCO2 has not changed.
        """
        return co2calc.calc_dic(
            ALK=new_alk,
            pCO2=self.pco2,
            S=self.salt,
            T=self.temp,
            PO4=self.po4,
            SiO3=self.sio3,
            input_in_gravimetric_units=False,
            pH0=self._pH0,
            thermodyn=self.equil_constants,
        )


class _calc_csys_pyco2sys(object):
    """
    Solve carbonate system chemistry using PyCO2SYS, but in volumetric units.
    """

    def __init__(self, dic, alk, salt, temp, sio3=0.0, po4=0.0, equil_constants={}):
        self.salt = salt
        self.temp = temp
        self.sio3 = sio3
        self.po4 = po4

        if not equil_constants:
            result = csys.sys(
                temperature=self.temp,
                salinity=self.salt,
            )
            self.equil_constants = {k: v for k, v in result.items() if k[:2] == "k_"}
            self.equil_constants["total_borate"] = result["total_borate"]
        else:
            self.equil_constants = equil_constants

        self.solve_co2(dic, alk)

    def solve_co2(self, dic, alk):
        """solve the C system chemistry"""
        self.co2sys = csys.sys(
            par1=mmolm3_to_µmolkg(dic),
            par2=mmolm3_to_µmolkg(alk),
            par1_type=par_type["DIC"],
            par2_type=par_type["ALK"],
            temperature=self.temp,
            salinity=self.salt,
            total_silicate=mmolm3_to_µmolkg(self.sio3),
            total_phosphate=mmolm3_to_µmolkg(self.po4),
            opt_buffers_mode=0,
            **self.equil_constants,
        )

    @property
    def co2sol(self):
        """return solubility in mmol/m^3/atm"""
        return molkg_to_mmolm3(self.co2sys["k_CO2"])

    @property
    def co2aq(self):
        """return CO2aq in mmol/m^3"""
        return µmolkg_to_mmolm3(self.co2sys["CO2"])

    @property
    def pco2(self):
        """Return pCO2 in µatm (using fugacity)"""
        return self.co2sys["fCO2"]

    @property
    def dic(self):
        """Return dic in mmol/m^3"""
        return µmolkg_to_mmolm3(self.co2sys["dic"])

    @property
    def alk(self):
        """Return alk in mmol/m^3"""
        return µmolkg_to_mmolm3(self.co2sys["alkalinity"])

    @property
    def pH(self):
        return self.co2sys["pH_total"]  # what's the right choice here?

    def calc_new_dic_w_oae(self, new_alk):
        """Compute the new DIC concentration in mmol/m^3
        after alkalinity addition assuming pCO2 has not changed.
        """
        new_co2sys = csys.sys(
            par1=self.pco2,
            par2=mmolm3_to_µmolkg(new_alk),
            par1_type=par_type["pCO2"],
            par2_type=par_type["ALK"],
            temperature=self.temp,
            salinity=self.salt,
            total_silicate=mmolm3_to_µmolkg(self.sio3),
            total_phosphate=mmolm3_to_µmolkg(self.po4),
            **self.equil_constants,
        )
        return µmolkg_to_mmolm3(new_co2sys["dic"])

    def calc_dic_removal_equiv_to_oae(self, new_alk):
        """Compute the amount of DIC removal needed to generate an
        anomaly in pCO2 that is equivalent to that achieved by adding
        alkalinity.
        """
        new_co2sys = csys.sys(
            par1=self.co2sys["dic"],
            par2=mmolm3_to_µmolkg(new_alk),
            par1_type=par_type["DIC"],
            par2_type=par_type["ALK"],
            temperature=self.temp,
            salinity=self.salt,
            total_silicate=mmolm3_to_µmolkg(self.sio3),
            total_phosphate=mmolm3_to_µmolkg(self.po4),
            **self.equil_constants,
        )
        pco2_w_oae_t0 = new_co2sys["fCO2"]

        new_co2sys = csys.sys(
            par1=pco2_w_oae_t0,
            par2=self.co2sys["alkalinity"],
            par1_type=par_type["pCO2"],
            par2_type=par_type["ALK"],
            temperature=self.temp,
            salinity=self.salt,
            total_silicate=mmolm3_to_µmolkg(self.sio3),
            total_phosphate=mmolm3_to_µmolkg(self.po4),
            **self.equil_constants,
        )
        return self.dic - µmolkg_to_mmolm3(new_co2sys["dic"])

    def ddicdco2(self, new_alk):
        """
        Compute the partial derivative of DIC wrt CO2
        """

        k1 = self.equil_constants["k_carbonic_1"]
        k2 = self.equil_constants["k_carbonic_2"]
        kb = self.equil_constants["k_borate"]
        bt = µmolkg_to_mmolm3(self.equil_constants["total_borate"])
        kw = self.equil_constants["k_water"]

        new_co2sys = csys.sys(
            par1=mmolm3_to_µmolkg(self.dic),
            par2=mmolm3_to_µmolkg(new_alk),
            par1_type=par_type["DIC"],
            par2_type=par_type["ALK"],
            temperature=self.temp,
            salinity=self.salt,
            total_silicate=mmolm3_to_µmolkg(self.sio3),
            total_phosphate=mmolm3_to_µmolkg(self.po4),
            **self.equil_constants,
        )
        co2 = µmolkg_to_mmolm3(new_co2sys["CO2"])
        pH = new_co2sys["pH_total"]

        # preliminaries
        h = 10 ** (-pH)
        h2 = h * h
        h3 = h * h * h
        k1k2 = k1 * k2
        kb_p_h_sq = (kb + h) ** 2

        # dDIC/d[CO2], pH = constant
        Ds = 1 + k1 / h + k1k2 / h2

        # dDIC/d[H+], [CO2] = constant
        Dh = -co2 * (k1 / h2 + 2 * k1k2 / h3)

        # dAlk/d[CO2], pH = constant
        As = k1 / h + 2 * k1k2 / h2

        # dAlk/d[H+], [CO2] = constant
        Ah = -co2 * (k1 / h2 + 4 * k1k2 / h3) - kb * bt / kb_p_h_sq - kw / h2 - 1

        # the result
        return Ds - Dh * As / Ah


class mixed_layer(object):
    """
    A simple mixed layer model where
    mixed layer depth, temperature, salinity, etc. are held constant
    and only DIC varies (and eventually maybe Alk too).
    """

    def __init__(
        self,
        dic,
        alk,
        salt,
        temp,
        sio3,
        po4,
        h,
        u10,
        Xco2atm,
        tau_dilution=None,
        dic_bc=None,
        alk_bc=None,
        csys_solver="ocmip",
    ):
        self.state = [dic, alk]
        self.h = h
        self.u10 = u10
        self.Xco2atm = Xco2atm
        self.salt = salt
        self.temp = temp
        self.sio3 = sio3
        self.po4 = po4

        self.tau_dilution = tau_dilution
        if self.tau_dilution is not None:
            if tau_dilution == 0.0:
                self.tau_dilution = None
            else:
                self.tau_dilution = tau_dilution * 86400.0

        self.dic_bc = dic_bc
        self.alk_bc = alk_bc

        self.dt = 86400.0

        # set the CO2 system solver
        self.csys_solver = calc_csys(dic, alk, salt, temp, sio3, po4, csys_solver)

        self._allocations()

    def _allocations(self):
        """allocate memory for working arrays"""
        self.tend = dict(
            dilution_dic=0.0,
            dilution_alk=0.0,
            gasex_co2=0.0,
            total=np.array([0.0, 0.0]),
        )
        self.diags = dict(
            pco2=0.0,
            fgco2=0.0,
            dilution_dic=0.0,
            dilution_alk=0.0,
        )

    def _compute_tendency(self, t, state):
        """
        Compute the tendency equation for box model
        h (dC/dt) = xkw * (co2atm - co2aq)

        t is not used at present
        """

        dic, alk = state

        # compute gas exchange
        self.csys_solver.solve_co2(dic, alk)

        xkw = gas_transfer_velocity(self.u10, self.temp)  # m/s
        co2atm = (
            self.Xco2atm * self.csys_solver.co2sol * 1e-6
        )  # mmol/m^3; implicit multiplication by 1 atm

        self.tend["gasex_co2"] = (
            xkw * (co2atm - self.csys_solver.co2aq) / self.h
        )  # mmol/m^3/s

        # compute dilution
        if self.tau_dilution is not None:
            self.tend["dilution_dic"] = (self.dic_bc - dic) / self.tau_dilution
            self.tend["dilution_alk"] = (self.alk_bc - alk) / self.tau_dilution

        # assemble tendendency
        self.tend["total"][0] = self.tend["dilution_dic"] + self.tend["gasex_co2"]
        self.tend["total"][1] = self.tend["dilution_alk"]

        # store diagnostics
        self.diags["pco2"] = self.csys_solver.pco2
        self.diags["fgco2"] = self.tend["gasex_co2"]
        self.diags["dilution_dic"] = self.tend["dilution_dic"]
        self.diags["dilution_alk"] = self.tend["dilution_alk"]

        return self.tend["total"], self.diags

    def run(self, nday, ddic=0.0, dalk=0.0, spinup=False):
        """
        Integrate the box model forward in time for nday
        """

        # spin up the model to steady-state
        if spinup:
            self._spinup()

        # add alkalinity or DIC anomalies to initial conditions
        self.state += np.array([ddic, dalk])

        # construct time axis
        time = xr.DataArray(
            np.arange(0, nday + 1, 1),
            dims=("time"),
            attrs=dict(units="days"),
        )
        nt = len(time)

        y = np.empty((2, nt))
        diags = {k: np.empty((nt)) for k in self.diags.keys()}
        for l in range(0, nt):
            self.state, diag_t = forward_euler(
                self._compute_tendency, self.dt, time[l], self.state
            )

            y[:, l] = self.state
            for k in self.diags.keys():
                diags[k][l] = diag_t[k]

        # SciPy ODE solver is faster, but I haven't figured out
        # how to get diagnostics back out
        # t_sec = time * 86400.0
        # soln = solve_ivp(
        #     self._compute_tendency,
        #     t_span=[t_sec[0], t_sec[-1]],
        #     t_eval=t_sec,
        #     y0=self.state,
        #     dense_output=False,
        #     method='LSODA',
        # )

        # construct output Dataset
        data_vars = dict(
            dic=xr.DataArray(
                y[0, :],
                coords=dict(time=time),
                attrs=dict(long_name="DIC", units="mmol/m^3"),
            ),
            alk=xr.DataArray(
                y[1, :],
                coords=dict(time=time),
                attrs=dict(long_name="Alkalinity", units="mmol/m^3"),
            ),
            pco2=xr.DataArray(
                diags["pco2"],
                coords=dict(time=time),
                attrs=dict(long_name="pCO2", units="µatm"),
            ),
            fgco2=xr.DataArray(
                diags["fgco2"] * 86400.0,
                coords=dict(time=time),
                attrs=dict(long_name="fgco2", units="mmol/m^3/d"),
            ),
            dilution_dic=xr.DataArray(
                diags["dilution_dic"] * 86400.0,
                coords=dict(time=time),
                attrs=dict(long_name="dilution_dic", units="mmol/m^3/d"),
            ),
            fgco2_cumulative=xr.DataArray(
                np.cumsum(diags["fgco2"]) * 86400.0 * self.h * 1e-3,
                coords=dict(time=time),
                attrs=dict(long_name="Total CO2 Uptake", units="mol/m^2"),
            ),
        )

        return xr.Dataset(
            data_vars=data_vars,
            coords=dict(time=time),
        )

    def _spinup(self):
        """generate a spun-up DIC state"""

        def wrapper(state_in):
            dcdt, _ = self._compute_tendency(0.0, state_in)
            state_out = state_in + dcdt * self.dt
            return np.sum((state_in - state_out) ** 2)

        opt_result = minimize(wrapper, self.state, bounds=[(1.0, 6000.0)], tol=1e-12)
        self.state = opt_result.x


def rk4(dfdt, dt, t, y):
    """
    4th order Runge-Kutta
    """
    dydt1, diag1 = dfdt(t, y)
    dydt2, diag2 = dfdt(t + dt / 2, y + dt * dydt1 / 2)
    dydt3, diag3 = dfdt(t + dt / 2, y + dt * dydt2 / 2)
    dydt4, diag4 = dfdt(t + dt, y + dt * dydt3)

    y_next_t = y + dt * (dydt1 + 2.0 * dydt2 + 2.0 * dydt3 + dydt4) / 6.0

    diag_t = dict(**diag1)
    for key in diag1.keys():
        diag_t[key] = (
            diag1[key] + 2.0 * diag2[key] + 2.0 * diag3[key] + diag4[key]
        ) / 6.0

    return y_next_t, diag_t


def forward_euler(dfdt, dt, t, y):
    """
    Forward Euler
    """
    dydt1, diag1 = dfdt(t, y)

    y_next_t = y + dt * dydt1

    diag_t = dict(**diag1)
    for key in diag1.keys():
        diag_t[key] = diag1[key]

    return y_next_t, diag_t


class timer(object):
    """timer object"""

    def __init__(self, name=None):
        self.name = name

    def __enter__(self):
        self.tstart = time.time()
        if self.name:
            print(f"[{self.name}]")

    def __exit__(self, type, value, traceback):
        print(f"Elapsed: {time.time() - self.tstart}")
