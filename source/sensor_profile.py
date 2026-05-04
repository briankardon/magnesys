"""Sensor profiles for magnetometer noise modeling.

A SensorProfile captures the two parameters that actually couple the
sensor to inversion accuracy in a multi-frequency lock-in setup:

* **noise spectral density** (nT/√Hz) — sets the noise floor that lock-in
  averaging integrates against
* **max 3-axis output data rate** (Hz) — caps how many samples per
  demodulation window the system actually receives

Per-sample RMS noise depends on both: for a sensor with one-sided PSD
N0 (nT/√Hz) read out at sample rate fs, the per-sample standard deviation
is

    σ_sample = N0 · √(fs / 2)

(equivalent noise bandwidth = fs/2 for an ideally anti-aliased signal).

The registry below lists representative profiles compiled from public
datasheets; values are approximate and should be refined against the
specific operating mode (OSR/cycle count/drive mode) you intend to use.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np


@dataclass(frozen=True)
class SensorProfile:
    """Describes a 3-axis magnetometer for noise modeling.

    Parameters
    ----------
    name : str
        Display name (used as the registry key).
    noise_density_nT_sqrtHz : float
        Noise spectral density per axis in nT/√Hz.  Use 0.0 for an ideal
        noiseless sensor.
    max_odr_hz : float
        Maximum sustainable 3-axis output data rate in Hz.  Use np.inf
        for an unconstrained analog frontend.
    technology : str
        Sense element type (Hall, AMR, TMR, fluxgate, magneto-inductive).
    mass_mg : Optional[float]
        Approximate sensor die/package mass in milligrams.
    power_uW : Optional[float]
        Typical active-mode power draw in microwatts.
    notes : str
        Free-form remarks (operating mode assumed, datasheet caveats).
    """

    name: str
    noise_density_nT_sqrtHz: float
    max_odr_hz: float
    technology: str = "unspecified"
    mass_mg: Optional[float] = None
    power_uW: Optional[float] = None
    notes: str = ""

    def per_sample_sigma_T(self, sample_rate_hz: float) -> float:
        """RMS per-sample noise in Tesla at the requested sample rate.

        Assumes white noise across the Nyquist band (ENBW = fs/2).
        If the requested rate exceeds the sensor's max ODR, the noise
        is computed at the max ODR (silent clamp; callers should warn).
        """
        if self.noise_density_nT_sqrtHz <= 0.0:
            return 0.0
        fs = min(sample_rate_hz, self.max_odr_hz)
        sigma_nT = self.noise_density_nT_sqrtHz * np.sqrt(fs / 2.0)
        return sigma_nT * 1e-9

    def per_sample_sigma_uT(self, sample_rate_hz: float) -> float:
        """RMS per-sample noise in µT (convenience wrapper)."""
        return self.per_sample_sigma_T(sample_rate_hz) * 1e6


_REGISTRY: Dict[str, SensorProfile] = {}


def register(profile: SensorProfile) -> SensorProfile:
    """Add a profile to the registry, returning it for chaining."""
    _REGISTRY[profile.name] = profile
    return profile


def get(name: str) -> SensorProfile:
    """Look up a profile by name (raises KeyError if unknown)."""
    return _REGISTRY[name]


def names() -> list:
    """Names of all registered profiles, in registration order."""
    return list(_REGISTRY.keys())


def all_profiles() -> list:
    """All registered profiles, in registration order."""
    return list(_REGISTRY.values())


# ----------------------------------------------------------------------
# Built-in profiles
# ----------------------------------------------------------------------
# Numbers are best-effort estimates from public datasheets / app notes;
# they reflect the lowest-noise practical mode for each part.  Real-world
# performance will depend on OSR, drive current / cycle count, and PCB
# layout — treat these as a starting point for a sweep, not a contract.

register(SensorProfile(
    name="ideal",
    noise_density_nT_sqrtHz=0.0,
    max_odr_hz=np.inf,
    technology="ideal",
    mass_mg=0.0,
    power_uW=0.0,
    notes="Noiseless reference; useful for isolating geometric error.",
))

register(SensorProfile(
    name="MLX90393",
    noise_density_nT_sqrtHz=150.0,
    max_odr_hz=500.0,
    technology="Hall",
    mass_mg=30.0,
    power_uW=300.0,
    notes=(
        "Melexis Hall-effect, 3x3x0.9 mm QFN. Burst mode 3-axis caps near "
        "~500 Hz at minimum OSR/filtering. Cannot support 1 kHz lock-in."
    ),
))

register(SensorProfile(
    name="MLX90394",
    noise_density_nT_sqrtHz=70.0,
    max_odr_hz=1000.0,
    technology="Hall",
    mass_mg=20.0,
    power_uW=10.0,
    notes=(
        "Melexis micropower successor to MLX90393; lower noise and "
        "current at similar package size."
    ),
))

register(SensorProfile(
    name="AK09940A",
    noise_density_nT_sqrtHz=30.0,
    max_odr_hz=2500.0,
    technology="TMR",
    mass_mg=10.0,
    power_uW=500.0,
    notes=(
        "AKM TMR magnetometer, 1.2x1.2 mm WLCSP. Low-noise drive mode. "
        "Best digital integrated option for ~200 Hz lock-in."
    ),
))

register(SensorProfile(
    name="MMC5983MA",
    noise_density_nT_sqrtHz=100.0,
    max_odr_hz=1000.0,
    technology="AMR",
    mass_mg=10.0,
    power_uW=500.0,
    notes="Memsic AMR, 1.2x1.2 mm. Similar regime to MLX90393.",
))

register(SensorProfile(
    name="RM3100",
    noise_density_nT_sqrtHz=13.0,
    max_odr_hz=430.0,
    technology="magneto-inductive",
    mass_mg=250.0,
    power_uW=3000.0,
    notes=(
        "PNI magneto-inductive. Excellent noise floor but capped 3-axis "
        "rate at default cycle count = 200; heavier package."
    ),
))

register(SensorProfile(
    name="DRV425x3",
    noise_density_nT_sqrtHz=1.5,
    max_odr_hz=47000.0,
    technology="fluxgate",
    mass_mg=60.0,
    power_uW=200000.0,
    notes=(
        "Three TI DRV425 single-axis fluxgates mounted orthogonally with "
        "external ADC. Best electrical performance; high power and "
        "mechanical complexity for a wearable."
    ),
))

# Default profile name (what the GUI/CLI pick if nothing is specified).
DEFAULT_NAME = "MLX90393"


def default() -> SensorProfile:
    return get(DEFAULT_NAME)
