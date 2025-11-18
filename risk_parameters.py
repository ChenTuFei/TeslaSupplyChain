# risk_parameters.py
# Centralized risk & uncertainty parameters for Tesla sourcing model

from dataclasses import dataclass
from typing import Dict
import json
import os


# -----------------------------
# 1. Basic site-level parameters
# -----------------------------

SITES = ["US", "MX", "CN"]


@dataclass
class SiteBaseParams:
    raw: float
    labor: float
    indirect: float
    logistics: float
    electricity: float
    depreciation: float
    tariff: float
    yield_rate: float
    damage_rate: float
    lead_time_days: int


BASE_PARAMS: Dict[str, SiteBaseParams] = {
    "US": SiteBaseParams(
        raw=40.0, labor=12.0, indirect=10.0, logistics=7.0,
        electricity=4.0, depreciation=5.0, tariff=0.0,
        yield_rate=0.80, damage_rate=0.002, lead_time_days=7,
    ),
    "MX": SiteBaseParams(
        raw=35.0, labor=8.0, indirect=8.0, logistics=7.0,
        electricity=3.0, depreciation=1.0, tariff=15.5,
        yield_rate=0.90, damage_rate=0.005, lead_time_days=10,
    ),
    "CN": SiteBaseParams(
        raw=30.0, labor=4.0, indirect=4.0, logistics=12.0,
        electricity=4.0, depreciation=5.0, tariff=15.0,
        yield_rate=0.95, damage_rate=0.010, lead_time_days=40,
    ),
}


# -----------------------------
# 2. Risk & delay parameters
# -----------------------------

RISK_EVENT_PROB: Dict[str, float] = {
    "US": 0.15,
    "MX": 0.10,
    "CN": 0.30,
}

RISK_EVENT_MAGNITUDE: Dict[str, float] = {
    "US": 0.05,
    "MX": 0.10,
    "CN": 0.25,
}

DELAY_COST_PER_DAY: Dict[str, float] = {
    "US": 0.055,
    "MX": 0.053,
    "CN": 0.631,
}

EXPECTED_DELAY_DAYS: Dict[str, float] = {
    "US": 1.0,
    "MX": 2.0,
    "CN": 15.0,
}

ESG_PENALTY: Dict[str, float] = {
    "US": 0.15,
    "MX": 0.20,
    "CN": 0.25,
}


# -----------------------------
# 3. Helper functions
# -----------------------------

def get_risk_premium_factor(site: str) -> float:
    p = RISK_EVENT_PROB[site]
    delta = RISK_EVENT_MAGNITUDE[site]
    return p * delta

def get_delay_penalty(site: str) -> float:
    return DELAY_COST_PER_DAY[site] * EXPECTED_DELAY_DAYS[site]

def get_esg_penalty(site: str) -> float:
    return ESG_PENALTY[site]
