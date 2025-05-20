import numpy as np
import pandas as pd
import math

def alpha_from_learning_rate(learning_rate):
    """
    Convert learning rate to alpha parameter.
    
    Parameters:
    -----------
    learning_rate : float
        Learning rate (% reduction per doubling)
        
    Returns:
    --------
    float
        Alpha parameter
    """
    # Convert percentage to decimal
    lr_decimal = learning_rate / 100
    # Calculate alpha
    alpha = math.log(1 - lr_decimal, 2)
    return alpha

def generate_local_learning_data(
    costs_0, 
    capacities_0, 
    growth_rates, 
    alphas, 
    years, 
    base_year=2023
):
    """
    Generate data points for the local learning model for BoP and EPC.
    
    In local learning model:
    C_region = C_0_region * (x_region/x_0_region)^alpha
    
    Parameters:
    -----------
    costs_0 : dict
        Dictionary of current capital costs ($/kW) for each region
    capacities_0 : dict
        Dictionary of current installed capacities (MW) for each region
    growth_rates : dict
        Dictionary of annual growth rates for each region (fraction)
    alphas : dict
        Dictionary of learning parameters for each region
    years : int
        Number of years to project
    base_year : int
        Starting year for projections
        
    Returns:
    --------
    dict
        Dictionary with DataFrames for each region containing year, capacity, and cost
    """
    # Initialize result dictionary
    results = {}
    
    # Get list of regions
    regions = list(costs_0.keys())
    
    # Initialize current capacities
    current_capacities = capacities_0.copy()
    
    # Create year range
    year_range = list(range(base_year, base_year + years + 1))
    
    # Initialize data for each region
    for region in regions:
        results[region] = {
            'year': year_range,
            'capacity': [current_capacities[region]],
            'cost': [costs_0[region]]
        }
    
    # Project for each year
    for year_idx, year in enumerate(year_range[1:], 1):  # Skip the base year (already set)
        # For each region, update capacity and calculate cost
        for region in regions:
            # Update capacity based on growth rate
            current_capacities[region] *= (1 + growth_rates[region])
            results[region]['capacity'].append(current_capacities[region])
            
            # Calculate cost using local learning model
            # C_region = C_0_region * (x_region/x_0_region)^alpha
            cost = costs_0[region] * (current_capacities[region] / capacities_0[region]) ** alphas[region]
            results[region]['cost'].append(cost)
    
    # Convert to DataFrames
    for region in regions:
        results[region] = pd.DataFrame(results[region])
    
    return results

def generate_global_learning_data(
    costs_0, 
    capacities_0, 
    growth_rates, 
    alphas, 
    years, 
    base_year=2023
):
    """
    Generate data points for the global learning model for BoP and EPC.
    
    In global learning model:
    C_region = C_0_region * ((x_USA + x_EU + x_CHINA + x_ROW)/(x_0_USA + x_0_EU + x_0_CHINA + x_0_ROW))^alpha
    
    Parameters:
    -----------
    costs_0 : dict
        Dictionary of current capital costs ($/kW) for each region
    capacities_0 : dict
        Dictionary of current installed capacities (MW) for each region
    growth_rates : dict
        Dictionary of annual growth rates for each region (fraction)
    alphas : dict
        Dictionary of learning parameters for each region
    years : int
        Number of years to project
    base_year : int
        Starting year for projections
        
    Returns:
    --------
    dict
        Dictionary with DataFrames for each region containing year, capacity, and cost
    """
    # Initialize result dictionary
    results = {}
    
    # Get list of regions
    regions = list(costs_0.keys())
    
    # Calculate initial total capacity across all regions
    x_0_total = sum(capacities_0.values())
    
    # Initialize current capacities
    current_capacities = capacities_0.copy()
    
    # Create year range
    year_range = list(range(base_year, base_year + years + 1))
    
    # Initialize data for each region
    for region in regions:
        results[region] = {
            'year': year_range,
            'capacity': [current_capacities[region]],
            'cost': [costs_0[region]]
        }
    
    # Project for each year
    for year_idx, year in enumerate(year_range[1:], 1):  # Skip the base year (already set)
        # Update capacities based on growth rates
        for region in regions:
            current_capacities[region] *= (1 + growth_rates[region])
            results[region]['capacity'].append(current_capacities[region])
        
        # Calculate total capacity across all regions
        total_capacity = sum(current_capacities.values())
        
        # Calculate costs using global learning for each region
        for region in regions:
            # C_region = C_0_region * ((x_total)/(x_0_total))^alpha
            cost = costs_0[region] * (total_capacity / x_0_total) ** alphas[region]
            results[region]['cost'].append(cost)
    
    # Convert to DataFrames
    for region in regions:
        results[region] = pd.DataFrame(results[region])
    
    return results