#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tesla Sourcing Strategy - AI Integrated Model
整合所有模块：成本预期、风险参数、物流成本、GARCH FX场景、Monte Carlo模拟

Phase 1: Launch/Ramp (0-6 months) - 100% Mexico with yield ramp-up
Phase 2: Steady State (6+ months) - Gurobi MILP optimization
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from gurobipy import Model, GRB, quicksum
from risk_parameters import (
    BASE_PARAMS, RISK_EVENT_PROB, RISK_EVENT_MAGNITUDE,
    DELAY_COST_PER_DAY, EXPECTED_DELAY_DAYS, ESG_PENALTY,
    get_risk_premium_factor, get_delay_penalty, get_esg_penalty
)

# ==============================
# 1. 物流成本计算（来自 logistics.ipynb）
# ==============================

def calculate_logistics_costs():
    """计算各站点的运输成本"""
    weight_kg = 7.26
    unit_value_usd = 100.0
    holding_rate = 0.15
    days_in_year = 365
    
    air_rate_usd_per_kg = 7.0
    air_oversize_surcharge = 200.0
    p_air = 0.01
    
    # 中国海运
    ocean_rate_40HC_normal = 3500.0
    ocean_rate_40HC_crisis_mult = 3.0
    units_per_container = 150
    us_port_drayage_per_container = 2000.0
    cn_ocean_days = 30
    cn_extra_delay_days = 10
    cn_crisis_prob = 0.01
    
    # 墨西哥卡车
    mx_distance_miles = 1500
    truck_cost_per_mile = 2.50
    mx_units_per_truck = 400
    mx_border_fee_per_truck = 500.0
    mx_border_delay_days = 2
    mx_p_air = 0.01
    
    # 美国国内卡车
    us_distance_miles = 2000
    us_units_per_truck = 400
    us_truck_delay_days = 1
    us_p_air = 0.01
    
    def holding_cost_per_unit(extra_days, unit_value=unit_value_usd, rate=holding_rate):
        return unit_value * rate * (extra_days / days_in_year)
    
    def air_cost_per_unit(weight=weight_kg, rate_per_kg=air_rate_usd_per_kg, oversize=air_oversize_surcharge):
        return weight * rate_per_kg + oversize
    
    def expected_air_cost(p_air, air_unit_cost):
        return p_air * air_unit_cost
    
    # 中国运输成本
    ocean_normal_per_unit = ocean_rate_40HC_normal / units_per_container
    drayage_per_unit = us_port_drayage_per_container / units_per_container
    expected_multiplier = (1 - cn_crisis_prob) * 1.0 + cn_crisis_prob * ocean_rate_40HC_crisis_mult
    expected_ocean_per_unit = (ocean_rate_40HC_normal * expected_multiplier) / units_per_container + drayage_per_unit
    exp_days = cn_ocean_days + cn_extra_delay_days * cn_crisis_prob
    holding = holding_cost_per_unit(exp_days)
    air_unit = air_cost_per_unit()
    exp_air = expected_air_cost(p_air, air_unit)
    cn_logistics = expected_ocean_per_unit + holding + exp_air
    
    # 墨西哥运输成本
    truck_total = truck_cost_per_mile * mx_distance_miles
    per_truck_all_in = truck_total + mx_border_fee_per_truck
    base_per_unit = per_truck_all_in / mx_units_per_truck
    holding = holding_cost_per_unit(mx_border_delay_days)
    air_unit = air_cost_per_unit()
    exp_air = expected_air_cost(mx_p_air, air_unit)
    mx_logistics = base_per_unit + holding + exp_air
    
    # 美国运输成本
    truck_total = truck_cost_per_mile * us_distance_miles
    base_per_unit = truck_total / us_units_per_truck
    holding = holding_cost_per_unit(us_truck_delay_days)
    air_unit = air_cost_per_unit()
    exp_air = expected_air_cost(us_p_air, air_unit)
    us_logistics = base_per_unit + holding + exp_air
    
    return {
        "US": us_logistics,
        "MX": mx_logistics,
        "CN": cn_logistics
    }


# ==============================
# 2. Phase 1: Ramp Period (0-6 months)
# ==============================

def ease_to_one(t, t_end, y0):
    """平滑yield ramp函数（ease-out quadratic）"""
    x = np.clip(t / t_end, 0, 1)
    return y0 + (1 - y0) * (1 - (1 - x)**2)

def calculate_ramp_costs(months=None):
    """计算ramp期间的时变成本（来自 cost_exp.py）"""
    if months is None:
        months = np.linspace(0, 6, 61)  # 月度分辨率
    
    # 基础成本（包含包装/库存，但不含运输成本，从BASE_PARAMS）
    # 注意：BASE_PARAMS中的logistics是包装/库存成本，运输成本单独计算
    base_costs = {
        "US": BASE_PARAMS["US"].raw + BASE_PARAMS["US"].labor + BASE_PARAMS["US"].indirect + 
              BASE_PARAMS["US"].logistics + BASE_PARAMS["US"].electricity + 
              BASE_PARAMS["US"].depreciation + BASE_PARAMS["US"].tariff,
        "MX": BASE_PARAMS["MX"].raw + BASE_PARAMS["MX"].labor + BASE_PARAMS["MX"].indirect + 
              BASE_PARAMS["MX"].logistics + BASE_PARAMS["MX"].electricity + 
              BASE_PARAMS["MX"].depreciation + BASE_PARAMS["MX"].tariff,
        "CN": BASE_PARAMS["CN"].raw + BASE_PARAMS["CN"].labor + BASE_PARAMS["CN"].indirect + 
              BASE_PARAMS["CN"].logistics + BASE_PARAMS["CN"].electricity + 
              BASE_PARAMS["CN"].depreciation + BASE_PARAMS["CN"].tariff,
    }
    
    # Yield ramp
    yield_US = ease_to_one(months, 6, BASE_PARAMS["US"].yield_rate)
    yield_MX = ease_to_one(months, 6, BASE_PARAMS["MX"].yield_rate)
    y0_cn = BASE_PARAMS["CN"].yield_rate
    target_at_6 = 0.99
    k = -np.log((1 - target_at_6) / (1 - y0_cn)) / 6.0
    yield_CN = 1 - (1 - y0_cn) * np.exp(-k * months)
    
    # 转换为每单位成本
    price_US = base_costs["US"] / yield_US
    price_MX = base_costs["MX"] / yield_MX
    price_CN = base_costs["CN"] / yield_CN
    
    # 添加物流成本
    logistics = calculate_logistics_costs()
    price_US += logistics["US"]
    price_MX += logistics["MX"]
    price_CN += logistics["CN"]
    
    return {
        "months": months,
        "US": price_US,
        "MX": price_MX,
        "CN": price_CN,
        "yield_US": yield_US,
        "yield_MX": yield_MX,
        "yield_CN": yield_CN
    }


def phase1_analysis():
    """Phase 1: 100% Mexico策略分析（含FX风险）"""
    print("=" * 60)
    print("Phase 1: Launch/Ramp Period (0-6 months)")
    print("Strategy: 100% Mexico sourcing")
    print("=" * 60)
    
    # 计算ramp成本
    ramp_data = calculate_ramp_costs()
    months = ramp_data["months"]
    mx_costs = ramp_data["MX"]
    
    # 加载FX场景（来自Monte Carlo）
    try:
        fx_scenarios = pd.read_csv("fx_scenarios_from_garch_paths.csv")
        print(f"\nLoaded {len(fx_scenarios)} FX scenarios from GARCH Monte Carlo")
        
        # 应用FX乘子到墨西哥成本（使用MXN）
        mx_base_cost_avg = np.mean(mx_costs)
        fx_multipliers = fx_scenarios["cost_multiplier_mxn"].values
        
        # 计算FX调整后的成本分布
        mx_costs_with_fx = mx_base_cost_avg * fx_multipliers
        
        print(f"\nMexico base cost (avg over 6 months): ${mx_base_cost_avg:.2f}")
        print(f"FX-adjusted cost statistics:")
        print(f"  Mean: ${np.mean(mx_costs_with_fx):.2f}")
        print(f"  Std:  ${np.std(mx_costs_with_fx):.2f}")
        print(f"  Min:  ${np.min(mx_costs_with_fx):.2f}")
        print(f"  Max:  ${np.max(mx_costs_with_fx):.2f}")
        print(f"  5th percentile: ${np.percentile(mx_costs_with_fx, 5):.2f}")
        print(f"  95th percentile: ${np.percentile(mx_costs_with_fx, 95):.2f}")
        
    except FileNotFoundError:
        print("\nWarning: fx_scenarios_from_garch_paths.csv not found. Skipping FX analysis.")
        mx_costs_with_fx = None
    
    # 绘制ramp期间成本曲线
    plt.figure(figsize=(10, 6))
    plt.plot(months, ramp_data["US"], label="US", color="royalblue", linewidth=2)
    plt.plot(months, ramp_data["MX"], label="Mexico", color="green", linewidth=2)
    plt.plot(months, ramp_data["CN"], label="China", color="red", linewidth=2)
    plt.title("Per-Lamp Cost During 6-Month Production Ramp\n(Phase 1: 100% Mexico Strategy)", 
              fontsize=14, fontweight='bold')
    plt.xlabel("Time (months)", fontsize=12)
    plt.ylabel("Price per lamp (USD)", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig("phase1_ramp_costs.png", dpi=300, bbox_inches='tight')
    print("\n[SAVED] Phase 1 ramp cost curve: phase1_ramp_costs.png")
    plt.show()
    
    return ramp_data, mx_costs_with_fx


# ==============================
# 3. Phase 2: Steady State Optimization
# ==============================

def build_steady_state_model(demand=300000, service_level_7day_pct=0.10):
    """
    构建Phase 2稳态优化模型（Gurobi MILP）
    整合：成本、风险、延迟、损坏、ESG惩罚
    """
    sites = ["US", "MX", "CN"]
    
    # 从BASE_PARAMS获取基础成本（包含包装/库存，但不含运输和关税）
    base_costs = {}
    for site in sites:
        params = BASE_PARAMS[site]
        base_costs[site] = (params.raw + params.labor + params.indirect + 
                           params.logistics + params.electricity + params.depreciation)
    
    # 物流成本
    logistics = calculate_logistics_costs()
    
    # Yield rates
    Y = {site: BASE_PARAMS[site].yield_rate for site in sites}
    
    # Tariff (已包含在BASE_PARAMS中，但需要单独考虑)
    T = {site: BASE_PARAMS[site].tariff for site in sites}
    
    # 风险、延迟、损坏、ESG
    risk_premium = {site: get_risk_premium_factor(site) for site in sites}
    delay_penalty = {site: get_delay_penalty(site) for site in sites}
    damage_rate = {site: BASE_PARAMS[site].damage_rate for site in sites}
    esg_penalty = {site: get_esg_penalty(site) for site in sites}
    
    # Lead times
    L = {site: BASE_PARAMS[site].lead_time_days for site in sites}
    
    # 容量
    K = {site: 200000 for site in sites}  # 每个站点200k容量
    
    # 固定成本（假设为0）
    F = {site: 0.0 for site in sites}
    
    # 服务级别要求
    R1 = service_level_7day_pct * demand  # >= 10% within 7 days (US only)
    
    # 构建每单位成本
    BaseLanded, Risk, Damage, Delay, ESG, v = {}, {}, {}, {}, {}, {}
    for site in sites:
        # 基础落地成本 = (基础成本 + 关税 + 物流) / yield
        BaseLanded[site] = (base_costs[site] + T[site] + logistics[site]) / Y[site]
        
        # 风险溢价
        Risk[site] = risk_premium[site] * BaseLanded[site]
        
        # 损坏成本
        Damage[site] = damage_rate[site] * BaseLanded[site]
        
        # 延迟成本
        Delay[site] = delay_penalty[site]
        
        # ESG惩罚
        ESG[site] = esg_penalty[site] * BaseLanded[site]
        
        # 总单位成本
        v[site] = BaseLanded[site] + Risk[site] + Damage[site] + Delay[site] + ESG[site]
    
    # 创建模型
    m = Model("Tesla_SteadyState_Sourcing")
    m.Params.OutputFlag = 1
    
    # 决策变量
    x = {
        site: m.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name=f"x_{site}")
        for site in sites
    }
    
    y = {
        site: m.addVar(vtype=GRB.BINARY, name=f"y_{site}")
        for site in sites
    }
    
    # 目标函数：最小化总成本
    m.setObjective(
        quicksum(v[site] * x[site] for site in sites) +
        quicksum(F[site] * y[site] for site in sites),
        GRB.MINIMIZE
    )
    
    # 约束
    # 需求满足
    m.addConstr(quicksum(x[site] for site in sites) == demand, name="Demand")
    
    # 服务级别：至少10%在7天内（US only）
    m.addConstr(
        quicksum(x[site] for site in sites if L[site] <= 7) >= R1,
        name="ServiceLevel_7day"
    )
    
    # 容量约束
    for site in sites:
        m.addConstr(x[site] <= K[site] * y[site], name=f"Capacity_{site}")
    
    # 供应商数量约束（可选，这里不限制）
    # m.addConstr(quicksum(y[site] for site in sites) <= 3, name="MaxSuppliers")
    
    return m, sites, x, y, v, BaseLanded, Risk, Damage, Delay, ESG


def phase2_optimization(demand=300000, service_level_7day_pct=0.10):
    """Phase 2: 稳态优化"""
    print("\n" + "=" * 60)
    print("Phase 2: Steady State Optimization (6+ months)")
    print("=" * 60)
    
    m, sites, x, y, v, BaseLanded, Risk, Damage, Delay, ESG = build_steady_state_model(
        demand, service_level_7day_pct
    )
    
    print("\nOptimizing steady-state sourcing model...")
    m.optimize()
    
    if m.status == GRB.OPTIMAL:
        print("\n=== Optimal Sourcing Solution ===")
        total_cost = m.objVal
        
        results = []
        for site in sites:
            xi = x[site].X
            yi = int(y[site].X)
            pct = 100.0 * xi / demand if demand > 0 else 0.0
            
            results.append({
                "Site": site,
                "Units": f"{xi:,.0f}",
                "Percentage": f"{pct:.1f}%",
                "Active": yi,
                "Unit_Cost": f"${v[site]:.3f}",
                "BaseLanded": f"${BaseLanded[site]:.2f}",
                "Risk": f"${Risk[site]:.2f}",
                "Damage": f"${Damage[site]:.2f}",
                "Delay": f"${Delay[site]:.2f}",
                "ESG": f"${ESG[site]:.2f}",
            })
            
            print(f"{site}: {xi:,.0f} units ({pct:5.1f}%), active={yi}, "
                  f"unit cost = ${v[site]:.3f}")
            print(f"  Breakdown: Base=${BaseLanded[site]:.2f}, "
                  f"Risk=${Risk[site]:.2f}, Damage=${Damage[site]:.2f}, "
                  f"Delay=${Delay[site]:.2f}, ESG=${ESG[site]:.2f}")
        
        print(f"\nTotal cost = ${total_cost:,.2f}")
        
        # 保存结果
        df_results = pd.DataFrame(results)
        df_results.to_csv("phase2_optimization_results.csv", index=False)
        print("\n[SAVED] Results saved to: phase2_optimization_results.csv")
        
        return df_results, total_cost
    else:
        print(f"Model did not solve to optimality. Status code: {m.status}")
        return None, None


# ==============================
# 4. 整合分析和可视化
# ==============================

def comprehensive_analysis():
    """执行完整的策略分析"""
    print("\n" + "=" * 60)
    print("Tesla Sourcing Strategy - Comprehensive Analysis")
    print("=" * 60)
    
    # Phase 1: Ramp Period
    ramp_data, fx_costs = phase1_analysis()
    
    # Phase 2: Steady State
    results, total_cost = phase2_optimization()
    
    # 总结
    print("\n" + "=" * 60)
    print("Strategy Summary")
    print("=" * 60)
    print("\nPhase 1 (0-6 months):")
    print("  - Strategy: 100% Mexico sourcing")
    print("  - Leverages mature automotive ecosystem")
    print("  - Shorter lead-time vs. China")
    print("  - Yield ramp-up: 90% → 100%")
    
    print("\nPhase 2 (6+ months):")
    print("  - Strategy: Optimized multi-source allocation")
    print("  - Gurobi MILP optimization")
    print("  - Considers: Cost, Risk, Delay, Damage, ESG")
    print("  - Service level: ≥10% within 7 days (US)")
    
    if results is not None:
        print("\nOptimal allocation:")
        for _, row in results.iterrows():
            print(f"  {row['Site']}: {row['Units']} ({row['Percentage']})")
    
    return ramp_data, results


# ==============================
# 5. 主程序
# ==============================

if __name__ == "__main__":
    # 执行完整分析
    ramp_data, results = comprehensive_analysis()
    
    print("\n" + "=" * 60)
    print("Analysis Complete!")
    print("=" * 60)
    print("\nGenerated files:")
    print("  - phase1_ramp_costs.png: Ramp period cost curves")
    print("  - phase2_optimization_results.csv: Steady state optimization results")
    print("\nNote: Ensure fx_scenarios_from_garch_paths.csv exists for FX risk analysis")

