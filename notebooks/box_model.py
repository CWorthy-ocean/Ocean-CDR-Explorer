import numpy as np
import PyCO2SYS as csys
import xarray as xr

from scipy.integrate import solve_ivp

# gas exchange coefficient
xkw_coef_cm_per_hr = 0.251

# (cm/hr s^2/m^2) --> (m/s s^2/m^2)
xkw_coef = xkw_coef_cm_per_hr * 3.6e-5

# reference density
rho_ref = 1026.0

# CO2SYS parameter types
par_type = dict(ALK=1, DIC=2, pH=3, pCO2=5) # using fugacity in place of pCO2


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
    return xkw_coef * u10sq * (np.sqrt(sc / 660.0))


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


class calc_csys(object):
    """
    Solve carbonate system chemistry using PyCO2SYS, but in volumetric units.
    """

    def __init__(self, dic, alk, salt, temp, sio3, po4, equil_constants={}):
        self.salt = salt
        self.temp = temp
        self.sio3 = sio3
        self.po4 = po4

        if not equil_constants:
            result = csys.sys(
                temperature=self.temp,
                salinity=self.salt,
            )
            self.equil_constants = {k: v for k, v in result.items() if k[:2] == 'k_'}
        else:
            self.equil_constants = equil_constants

        self.solve_co2(dic, alk)

    def solve_co2(self, dic, alk):
        """solve the C system chemistry"""
        self.co2sys = csys.sys(
            par1=mmolm3_to_µmolkg(dic),
            par2=mmolm3_to_µmolkg(alk),
            par1_type=par_type['DIC'],
            par2_type=par_type['ALK'],
            temperature=self.temp,
            salinity=self.salt,
            total_silicate=self.sio3,
            total_phosphate=self.po4,
            **self.equil_constants,
        )

    @property
    def co2sol(self):
        """return solubility in mmol/m^3/µatm"""
        return µmolkg_to_mmolm3(self.co2sys['k_CO2'])

    @property
    def co2aq(self):
        """return CO2aq in mmol/m^3"""
        return µmolkg_to_mmolm3(self.co2sys['CO2'])

    @property
    def pco2(self):
        """Return pCO2 in µatm (using fugacity)"""
        return self.co2sys['fCO2']

    @property
    def dic(self):
        """Return dic in mmol/m^3"""
        return µmolkg_to_mmolm3(self.co2sys['dic'])

    @property
    def alk(self):
        """Return alk in mmol/m^3"""
        return µmolkg_to_mmolm3(self.co2sys['alkalinity'])
        
    def calc_new_dic_w_oae(self, new_alk):
        """Compute the new DIC concentration in mmol/m^3
           after alkalinity addition assuming pCO2 has not changed.
        """
        new_co2sys = csys.sys(
            par1=self.pco2,
            par2=mmolm3_to_µmolkg(new_alk),
            par1_type=par_type['pCO2'],
            par2_type=par_type['ALK'],
            temperature=self.temp,
            salinity=self.salt,
            total_silicate=self.sio3,
            total_phosphate=self.po4,
            **self.equil_constants,
        )
        return µmolkg_to_mmolm3(new_co2sys['dic'])


class mixed_layer(object):
    """
    A simple mixed layer model where 
    mixed layer depth, temperature, salinity, etc. are held constant
    and only DIC and Alk vary.
    """

    def __init__(self, dic, alk, h, u10, Xco2atm, salt, temp, sio3, po4):
        self.state = [dic, alk]
        self.h = h
        self.u10 = u10
        self.Xco2atm = Xco2atm
        self.salt = salt
        self.temp = temp
        self.sio3 = sio3
        self.po4 = po4

        self.csys_solver = calc_csys(dic, alk, salt, temp, sio3, po4)

    def compute_tendency(self, t, state):
        """
        Compute the tendency equation for box model
        h (dC/dt) = xkw * (co2atm - co2aq)
        """

        dic, alk = state
        self.csys_solver.solve_co2(dic, alk)

        xkw = gas_transfer_velocity(self.u10, self.temp)  # m/s
        co2atm = self.Xco2atm * self.csys_solver.co2sol  # mmol/m^3; implicit multiplication by 1 atm
        gasex_co2 = xkw * (co2atm - self.csys_solver.co2aq)  # mmol/m^2/s

       
        return gasex_co2 / self.h, 0.0  # mmol/m^3/s

    def run(self, nday):
        """
        Integrate the box model forward in time for nday
        """
        time = xr.DataArray(
            np.arange(0, nday, 1),
            dims=('time'),
            attrs=dict(units='days'),
        )

        t_sec = time * 86400.0
        soln = solve_ivp(
            self.compute_tendency,
            t_span=[t_sec[0], t_sec[-1]],
            t_eval=t_sec,
            y0=self.state,
            method='Radau',
        )

        # compute diagnostics
        self.csys_solver.solve_co2(soln.y[0, :], soln.y[1, :])

        # set up output Dataset
        data_vars = dict(
            dic=xr.DataArray(
                soln.y[0, :], 
                coords=dict(time=time), 
                attrs=dict(long_name='DIC', units='mmol/m^3')),
            alk=xr.DataArray(
                soln.y[1, :], 
                coords=dict(time=time), 
                attrs=dict(long_name='Alkalinity', units='mmol/m^3')),
            pco2=xr.DataArray(
                self.csys_solver.pco2, 
                coords=dict(time=time), 
                attrs=dict(long_name='pCO2', units='µatm')),
        )

        return xr.Dataset(
            data_vars=data_vars,
            coords=dict(time=time),
        )
