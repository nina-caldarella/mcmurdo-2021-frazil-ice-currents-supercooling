import xarray as xr
import numpy as np
# ----------------------------
# EOS-80 freezing point (ITS-90)
# t_f(°C, ITS-90) = (a0*S + a1*S*sqrt(S) + a2*S^2 + b*p) × 0.99976
# a0..b are UNESCO / Millero coefficients (originally on IPTS-68).
# Multiply by 0.99976 to convert to ITS-90.  (1/1.00024 ≈ 0.99976)
# Refs: UNESCO 1983; Millero & Leung (1976); CSIRO sw_fp.m implementation.
# ----------------------------
from seawater import fp

# ----------------------------
# Practical Salinity from conductivity (PSS-78)
# Try python-seawater (EOS-80) first; fall back to TEOS-10 gsw.SP_from_C.
# Both implement the PSS-78 practical-salinity algorithm from UNESCO (1983).
# ----------------------------
from gsw import SP_from_C

## density
from seawater import pden

# ----------------------------
# Monte-Carlo uncertainty propagation for SP and EOS-80 freezing point
# ----------------------------
def propagate_uncertainty_eos80(
    C_Sm, T90_C, p_dbar,
    uC_Sm, uT_C, uP_dbar,
    n_samples=20000, seed=0, clip_C_positive=True, return_fd_check=True
):
    """
    Inputs may be scalars or arrays (they will be broadcast to common shape).
    Instrument 1-sigma uncertainties: uC (mS/cm), uT (°C ITS-90), uP (dbar).
    Returns dict with central values and 1-sigma uncertainties from Monte Carlo.
    """
    rng = np.random.default_rng(seed)

    # Broadcast to common shape
    C0 = np.asarray(C_Sm, float)
    T0 = np.asarray(T90_C, float)
    P0 = np.asarray(p_dbar, float)
    shape = np.broadcast_shapes(C0.shape, T0.shape, P0.shape)
    C0 = np.broadcast_to(C0, shape)
    T0 = np.broadcast_to(T0, shape)
    P0 = np.broadcast_to(P0, shape)

    # Draw samples
    C_s = rng.normal(C0, uC_Sm, size=(n_samples,) + shape)
    T_s = rng.normal(T0, uT_C,     size=(n_samples,) + shape)
    P_s = rng.normal(P0, uP_dbar,  size=(n_samples,) + shape)

    if clip_C_positive:
        C_s = np.maximum(C_s, np.finfo(float).tiny)  # avoid non-positive C on log/ratio ops

    # Convert C to units mS/cm for GSW toolbox
    C_s_mScm = 10 * C_s

    # Evaluate SP and freezing point for each draw
    # (loop by chunk to keep memory in check if your arrays are large)
    SP_s = np.empty_like(C_s)
    rho_s = np.empty_like(C_s)
    for k in range(n_samples):
        SP_s[k] = SP_from_C(C_s_mScm[k], T_s[k], P_s[k]) # replaced with original GSW package
        rho_s[k] = pden(SP_s[k], T_s[k]/ 0.99976, P_s[k])
        
    Tf_s = fp(SP_s, P_s) # replaced with original EOS80 package

    # Central values at nominal inputs
    SP0 = SP_from_C(10*C0, T0, P0)
    Tf0 = fp(SP0, P0)
    rho0 = pden(SP0, T0/ 0.99976, P0)

    # Monte-Carlo 1-sigma
    u_SP = SP_s.std(axis=0, ddof=1)
    u_rho = rho_s.std(axis=0, ddof=1)
    u_Tf = Tf_s.std(axis=0, ddof=1)
    
    depth_dummy = np.arange(len(C_Sm))

    ds = xr.Dataset({
        "SP": (("depth_dummy"), SP0),        # Practical Salinity (PSS-78)
        "rho": (("depth_dummy"), rho0),      # Potential density
        "Tf": (("depth_dummy"), Tf0),        # Freezing point (°C, ITS-90)
        "u_SP": (("depth_dummy"), u_SP),     # 1-sigma uncertainty in SP (PSS-78 units)
        "u_Tf": (("depth_dummy"), u_Tf),     # 1-sigma uncertainty in Tf (°C)
        "u_rho": (("depth_dummy"), u_rho),      # Potential density
         },
         coords = {"depth_dummy": depth_dummy,
         "n_samples": n_samples})

    return ds
    
# ----------------------------
# Monte-Carlo uncertainty propagation for TEOS-10
# ----------------------------
import gsw

def pressure_uncertainty_teos10(SA, CT, p, uP, dp=0.01):
    """
    Depth-dependent absolute uncertainty from pressure only.
    """

    # Reference
    Tf0  = gsw.CT_freezing(SA, p, 0.0)
    rho0 = gsw.rho(SA, CT, p)

    # Numerical sensitivities
    dTf_dp = (
        gsw.CT_freezing(SA, p + dp, 0.0)
        - gsw.CT_freezing(SA, p - dp, 0.0)
    ) / (2 * dp)

    drho_dp = (
        gsw.rho(SA, CT, p + dp)
        - gsw.rho(SA, CT, p - dp)
    ) / (2 * dp)

    # Absolute uncertainties
    uTf_p  = np.abs(dTf_dp)  * uP
    urho_p = np.abs(drho_dp) * uP

    return uTf_p, urho_p

def uncertainty_SA_from_conductivity(
    C_Sm, CT, p, uC_Sm,
    lon, lat,
    n_samples=5000, seed=0, clip_C_positive=True
):
    """
    Scalar SA uncertainty propagated from conductivity uncertainty.
    """

    rng = np.random.default_rng(seed)

    # Representative state
    C0  = float(np.nanmedian(C_Sm))
    CT0 = float(np.nanmedian(CT))
    P0  = float(np.nanmedian(p))

    uC  = float(uC_Sm)

    # Monte Carlo on conductivity
    C_s = rng.normal(C0, uC, size=n_samples)
    if clip_C_positive:
        C_s = np.maximum(C_s, np.finfo(float).tiny)

    # C → SP → SA
    SP_s = gsw.SP_from_C(10.0 * C_s, CT0, P0)
    SA_s = gsw.SA_from_SP(SP_s, P0, lon, lat)

    SA0 = gsw.SA_from_SP(
        gsw.SP_from_C(10.0 * C0, CT0, P0),
        P0, lon, lat
    )

    uSA = SA_s.std(ddof=1)

    return SA0, uSA

def uncertainty_Tf_rho_SA(
    SA, CT, C_Sm, p, uCT, uC_Sm, lon, lat,
    n_samples=5000, seed=0, clip_C_positive=True
):
    """
    Scalar relative uncertainty from CT and SA.
    """

    rng = np.random.default_rng(seed)

    CT0 = float(np.nanmedian(CT))
    P0  = float(np.nanmedian(p))
    C0  = float(np.nanmedian(C_Sm))
    SA0 = float(np.nanmedian(SA))

    uCT = float(uCT)
        
    uC  = float(uC_Sm)

    # Monte Carlo on conductivity
    C_s = rng.normal(C0, uC, size=n_samples)
    if clip_C_positive:
        C_s = np.maximum(C_s, np.finfo(float).tiny)

    # C → SP → SA
    SP_s = gsw.SP_from_C(10.0 * C_s, CT0, P0)
    SA_s = gsw.SA_from_SP(SP_s, P0, lon, lat)

    Tf_s  = gsw.CT_freezing(SA_s, P0, 0.0)
    rho_s = gsw.rho(SA_s, CT0, P0)

    Tf0  = gsw.CT_freezing(SA0, P0, 0.0)
    rho0 = gsw.rho(SA0, CT0, P0)

    rTf_CTSA  = Tf_s.std(ddof=1)  / abs(Tf0)
    rrho_CTSA = rho_s.std(ddof=1) / abs(rho0)
    ruSA = SA_s.std(ddof=1) / abs(SA0)

    return rTf_CTSA, rrho_CTSA, ruSA
    
def total_uncertainty_TEOS10(
    SA, CT, p, uP,
    C_Sm, uC_Sm,
    uCT,
    lon, lat
):

    # --- Scalar relative CT+SA uncertainty ---
    rTf_CTSA, rrho_CTSA, ruSA = uncertainty_Tf_rho_SA(SA, CT, C_Sm, p, uCT, uC_Sm, lon, lat,
    )

    # --- Pressure contribution ---
    uTf_p, urho_p = pressure_uncertainty_teos10(
        SA, CT, p, uP
    )

    # Reference values
    Tf0  = gsw.CT_freezing(SA, p, 0.0)
    rho0 = gsw.rho(SA, CT, p)

    # Convert scalar relative → absolute
    uTf_CTSA  = rTf_CTSA  * np.abs(Tf0)
    urho_CTSA = rrho_CTSA * np.abs(rho0)

    # Final absolute uncertainty vs depth
    uTf_total  = np.sqrt(uTf_p**2  + uTf_CTSA**2)
    urho_total = np.sqrt(urho_p**2 + urho_CTSA**2)

    return {
        "uTf": uTf_total,            # absolute, depth-dependent
        "urho": urho_total,          # absolute, depth-dependent
        "ruSA": ruSA                 # scalar absolute
    }
