# TeslaSupplyChain
# Tesla Supply Chain Case Competition - Headlamp Component Sourcing Strategy Analysis

## Overview

This project provides a comprehensive sourcing decision analysis for Tesla's automotive component manufacturing across three potential locations: **United States**, **Mexico**, and **China**. The analysis employs multiple methodologies including:

- **Two-phase sourcing strategy** (Launch/Ramp and Steady State optimization)
- **Gurobi MILP optimization** for multi-objective cost minimization
- **GARCH model + Monte Carlo simulation** for FX risk quantification
- **Yield ramp-up modeling** during production startup
- **Multi-criteria decision analysis** considering Cost, Risk, Delay, Damage, and ESG factors

### Core Thesis

**Sourcing from Mexico is the optimal choice**, as it lowers costs by eliminating tariffs and increases supply chain resilience through near-shoring, despite manageable risks.

**Key Advantages:**
-  **Tariff Advantage**: USMCA eliminates the 25% tariff, saving ~$15 per unit vs. China
-  **Near-shoring Benefits**: Reduces delivery time (<3 days vs. 4-6 weeks from China)
-  **Cost Advantage**: Lowest total landed cost with 30-40% reduction in logistics costs
-  **Manageable Risks**: Lowest overall risk score (1.0)

## Cost Breakdown

The following cost components (in USD per unit) are analyzed for each location:

| Component | US | Mexico | China |
|-----------|------|--------|-------|
| Raw material | $40 | $35 | $30 |
| Labor | $12 | $8 | $4 |
| Indirect costs | $10 | $8 | $4 |
| Packaging/transport/inventory | $9 | $7 | $12 |
| Electricity | $4 | $3 | $4 |
| Depreciation | $5 | $1 | $5 |
| Tariff costs | $0 | $15.50 | $15 |
| **Total (baseline)** | **$80** | **$77.50** | **$74** |

**Note**: Baseline costs exclude logistics, yield adjustments, risk premiums, delay penalties, and ESG factors, which are calculated separately.

## Yield Analysis

Production yield significantly impacts effective cost per good unit. During the 6-month ramp-up period:

- **US**: 80% → 100% (ease-out quadratic)
- **Mexico**: 90% → 100% (ease-out quadratic)
- **China**: 95% → 99% (exponential approach)

Effective price = Baseline cost / Yield rate

Even though China has the lowest baseline cost ($74), its effective cost advantage diminishes when accounting for yield inefficiencies. The US shows the most dramatic improvement, starting at $100/lamp (80% yield) but dropping to $80/lamp at full yield.

## Scripts

### 1. `tesla_final_model.py` - Integrated Main Model ⭐ **PRIMARY ANALYSIS**

**What it does:**
- **Phase 1 (0-6 months)**: Models 100% Mexico sourcing with yield ramp-up, applies GARCH FX scenarios, generates cost distribution statistics
- **Phase 2 (6+ months)**: Solves Gurobi MILP optimization minimizing total cost (BaseLanded + Risk + Damage + Delay + ESG)
- **Logistics**: Models ocean freight (China), trucking (Mexico/US), includes holding costs and border fees

**Output:**
- `phase1_ramp_costs.png`: Cost curves over 6 months
- `phase2_optimization_results.csv`: Optimal allocation with cost breakdown
- FX risk statistics (if `fx_scenarios_from_garch_paths.csv` exists)

**Results:**
- Phase 1: 100% Mexico with yield ramp-up (90% → 100%)
- Phase 2: Optimal allocation: 66.7% Mexico, 33.3% US, 0% China

### 2. `cost_exp.py` - Smooth Yield Ramp Model

Models continuous yield improvement over 6 months using ease-out quadratic (US/MX) and exponential (CN) functions. Plots per-lamp cost curves showing gradual cost reduction.


### 3. `risk_parameters.py` - Centralized Risk Parameters

Defines base costs, yield rates, risk event probabilities/magnitudes, delay costs, and ESG penalties for all locations. Imported by `tesla_final_model.py`.

**Key Parameters:**
- Yield rates: US 80%, MX 90%, CN 95%
- Risk probabilities: US 15%, MX 10%, CN 30%
- ESG penalties: US 15%, MX 20%, CN 25% of base landed cost

### 4. Notebooks

- **`garch.ipynb`**: GARCH modeling for FX risk analysis, generates `fx_scenarios_from_garch_paths.csv`
- **`logistics_cost.ipynb`**: Detailed transportation cost analysis (ocean freight, trucking, air contingencies)
- **`monte carlo.ipynb`**: Monte Carlo simulation for cost uncertainty (20,000+ scenarios)
- **`tesla_model_1.ipynb`**: Additional exploratory analysis

## Key Findings

### Phase 2 Optimization Results

| Site | Units | Percentage | Unit Cost | BaseLanded | Risk | Damage | Delay | ESG |
|------|-------|------------|-----------|------------|------|--------|-------|-----|
| **US** | 100,000 | 33.3% | $134.92 | $116.31 | $0.87 | $0.23 | $0.06 | $17.45 |
| **MX** | 200,000 | 66.7% | $122.57 | $100.79 | $1.01 | $0.50 | $0.11 | $20.16 |
| **CN** | 0 | 0.0% | $170.90 | $120.92 | $9.07 | $1.21 | $9.46 | $30.23 |

**Key Insights:**

1. **Mexico Dominates**: 66.7% allocation with lowest unit cost ($122.57)
   - Lowest base landed cost ($100.79), manageable risk ($1.01)

2. **US Strategic Buffer**: 33.3% share to meet 7-day fast delivery requirement
   - Higher cost ($134.92) but lowest risk ($0.87) and delay ($0.06)

3. **China Excluded**: High risk ($9.07), delay costs ($9.46), and ESG penalty ($30.23)
