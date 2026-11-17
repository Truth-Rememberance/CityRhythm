"""HIDDEN ground truth for CityRhythm R2.

TRUTH FIREWALL: this module contains the hidden coupling parameters and the
generation parameters (daily curve amplitudes, noise levels, cross-system
couplings). Nothing outside backend/sim may import it. Analysis modules must
infer these from the emitted sim.tick stream, never read them.

All values describe a SYNTHETIC city; they are calibrated by feel, not from
real data.
"""

from __future__ import annotations

from typing import Any

# --- Determinism -----------------------------------------------------------
# Three independent noise streams per run (ERRATA spec: process / sensor /
# scenario events). Offsets are arbitrary but fixed.
STREAM_OFFSETS: dict[str, int] = {
    "process": 100_000,
    "sensor": 200_000,
    "events": 300_000,
}

# --- Daily-shape parameters (fractional hour h in [0, 24)) ------------------
# Each curve = base + sum of Gaussian bumps: amp * exp(-0.5*((h-mu)/sigma)**2)
# Values are utilization = load / capacity, nominal range ~0..1.3.

# Traffic: morning + evening commute peaks.
TRAFFIC_CURVE: dict[str, Any] = {
    "base": 0.16,
    "bumps": [
        {"amp": 0.62, "mu": 9.0, "sigma": 1.4},  # morning peak
        {"amp": 0.72, "mu": 18.5, "sigma": 1.7},  # evening peak
        {"amp": 0.14, "mu": 13.0, "sigma": 2.6},  # midday shoulder
    ],
}

# Power: moderate morning bump, dominant evening peak.
POWER_CURVE: dict[str, Any] = {
    "base": 0.34,
    "bumps": [
        {"amp": 0.16, "mu": 9.5, "sigma": 2.0},  # morning bump
        {"amp": 0.46, "mu": 19.5, "sigma": 2.2},  # evening peak
    ],
}

# Water: dominant morning peak, smaller evening peak.
WATER_CURVE: dict[str, Any] = {
    "base": 0.26,
    "bumps": [
        {"amp": 0.50, "mu": 7.0, "sigma": 1.5},  # morning peak
        {"amp": 0.30, "mu": 19.0, "sigma": 2.0},  # evening peak
    ],
}

CURVES: dict[str, dict[str, Any]] = {
    "traffic": TRAFFIC_CURVE,
    "power": POWER_CURVE,
    "water": WATER_CURVE,
}

# --- Weekend / holiday overrides (ERRATA item 3) ----------------------------
# Applied on top of the weekday curve when day_type != "weekday".
WEEKEND_OVERRIDE: dict[str, Any] = {
    "traffic": {"amp_scale": 0.70, "mu_shift": {"morning": 0.75}},  # +45 min
    "power": {
        "amp_scale": 1.0,
        "mu_shift": {},
        "morning_amp_scale": 0.60,
        "evening_mu_shift": 0.5,
    },  # morning bump x0.6, evening +30 min
    "water": {"amp_scale": 1.0, "mu_shift": {"morning": 1.0}},  # +60 min
}

# --- Zone profile multipliers (amplitude modulation per zone) ---------------
# Keys: base (additive), per-bump-role scale.
PROFILE_MOD: dict[str, dict[str, float]] = {
    "it_corridor": {
        "base": 0.03,
        "morning": 1.25,
        "evening": 0.85,
        "midday": 1.05,
        "power_base": 0.08,
    },
    "junction": {"base": 0.05, "morning": 1.05, "evening": 1.15, "midday": 1.0, "power_base": 0.0},
    "commercial": {
        "base": 0.02,
        "morning": 1.05,
        "evening": 1.05,
        "midday": 1.1,
        "power_base": 0.05,
    },
    "residential": {
        "base": 0.0,
        "morning": 0.70,
        "evening": 1.10,
        "midday": 0.8,
        "power_base": -0.04,
    },
    "mixed": {"base": 0.01, "morning": 1.0, "evening": 1.0, "midday": 1.0, "power_base": 0.0},
}

# --- Dynamics (AR(1) relaxation toward the curve target) --------------------
DYNAMICS: dict[str, float] = {
    "alpha": 0.5,  # mean-reversion rate per tick toward curve target
    "phi": 0.80,  # AR(1) persistence of the process-noise state
}

SIGMA_PROCESS: dict[str, float] = {  # per system, latent process noise
    "traffic": 0.030,
    "power": 0.022,
    "water": 0.025,
}
SIGMA_SENSOR: float = 0.015  # additive Gaussian sensor noise on all nodes

# --- Hidden cross-system couplings (what analysis should have to infer) -----
COUPLING: dict[str, float] = {
    # Congested traffic adds latent load to power (commercial/IT activity).
    "traffic_to_power": 0.18,
    "traffic_to_power_threshold": 0.55,  # only above this utilization
    # Hot afternoons push water demand (per degC above 24).
    "temp_to_water": 0.010,
    # Cross-system correlated shock (common latent activity factor).
    "common_factor_sigma": 0.012,
}

# --- Synthetic temperature model (degC) ------------------------------------
TEMP_MODEL: dict[str, float] = {
    "mean": 24.0,
    "diurnal_amp": 4.5,  # peak ~15:00
    "sigma": 0.35,  # slow day-to-day drift driven by the process stream
}

# --- Rounding (ERRATA item 6) ----------------------------------------------
ROUND_VALUE: int = 4
ROUND_RAW: int = 3
VALUE_MAX: float = 1.3

# --- Synthetic citizen reports ----------------------------------------------
REPORT_P: float = 0.10  # probability of one report per tick
REPORT_P_EXTRA: float = 0.25  # given one, probability of a second
REPORT_TEMPLATES: dict[str, dict[str, str]] = {
    "traffic": {
        "en": "Heavy traffic near {zone} junction, movement is very slow.",
        "kn": "ಟ್ರಾಫಿಕ್ ಜಾಸ್ತಿ, {zone} ಬಳಿ ವಾಹನಗಳು ನಿಧಾನವಾಗಿ ಚಲಿಸುತ್ತಿವೆ.",
    },
    "power": {
        "en": "Power fluctuation reported around {zone} since morning.",
        "kn": "{zone} ಸುತ್ತ ಬೆಳಿಗ್ಗೆಯಿಂದ ಕರಂಟ್ ಏರಿಳಿತ.",
    },
    "water": {
        "en": "Low water pressure in parts of {zone} today.",
        "kn": "ಇಂದು {zone} ನಲ್ಲಿ ನೀರಿನ ಒತ್ತಡ ಕಡಿಮೆ.",
    },
}
