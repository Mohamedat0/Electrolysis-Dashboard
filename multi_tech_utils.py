import numpy as np
import pandas as pd
import math

def calculate_learning_rate(alpha):
    """
    Convert the learning parameter (alpha) to a learning rate.
    
    The learning rate represents the percentage reduction in cost
    for each doubling of installed capacity.
    
    Parameters:
    -----------
    alpha : float
        Learning parameter
        
    Returns:
    --------
    float
        Learning rate (% reduction per doubling)
    """
    learning_rate = 1 - 2**alpha
    return learning_rate * 100  # Convert to percentage

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

def calculate_future_cost(c_0, x_0, x_y, alpha):
    """
    Calculate the future capital cost using the experience curve formula.
    
    Parameters:
    -----------
    c_0 : float
        Current capital cost ($/kW)
    x_0 : float
        Current installed capacity (MW)
    x_y : float
        Future installed capacity (MW)
    alpha : float
        Learning parameter
        
    Returns:
    --------
    float
        Future capital cost ($/kW)
    """
    # Apply the experience curve formula
    c_y = c_0 * (x_y / x_0)**alpha
    return c_y

def generate_shared_learning_data(
    costs_0, 
    capacities_0, 
    growth_rates, 
    alphas, 
    years, 
    base_year=2023
):
    """
    Generate data points for plotting the shared learning curve over time 
    for multiple technologies.
    
    In shared learning model:
    C_tech = C_0_tech * ((x_total)/(x_0_total))^alpha
    
    Parameters:
    -----------
    costs_0 : dict
        Dictionary of current capital costs ($/kW) for each technology
    capacities_0 : dict
        Dictionary of current installed capacities (MW) for each technology
    growth_rates : dict
        Dictionary of annual growth rates for each technology (fraction)
    alphas : dict
        Dictionary of learning parameters for each technology
    years : int
        Number of years to project
    base_year : int
        Starting year for projections
        
    Returns:
    --------
    dict
        Dictionary with DataFrames for each technology containing year, capacity, and cost
    """
    # Initialize result dictionary
    results = {}
    
    # Get list of technologies
    technologies = list(costs_0.keys())
    
    # Calculate initial total capacities by type
    x_0_pem = capacities_0['western_pem'] + capacities_0['chinese_pem']
    x_0_alk = capacities_0['western_alk'] + capacities_0['chinese_alk']
    x_0_total = x_0_pem + x_0_alk
    
    # Initialize current capacities
    current_capacities = capacities_0.copy()
    
    # Create year range
    year_range = list(range(base_year, base_year + years + 1))
    
    # Initialize data for each technology
    for tech in technologies:
        results[tech] = {
            'year': year_range,
            'capacity': [current_capacities[tech]],
            'cost': [costs_0[tech]]
        }
    
    # Project for each year
    for year_idx, year in enumerate(year_range[1:], 1):  # Skip the base year (already set)
        # Update capacities based on growth rates
        for tech in technologies:
            current_capacities[tech] *= (1 + growth_rates[tech])
            results[tech]['capacity'].append(current_capacities[tech])
        
        # Calculate total capacity across all technologies
        total_capacity = sum(current_capacities.values())
        
        # Calculate costs using shared learning for each technology
        for tech in technologies:
            if tech.endswith('pem'):
                # For PEM technologies
                cost = costs_0[tech] * (total_capacity / x_0_total) ** alphas[tech]
            else:
                # For ALK technologies
                cost = costs_0[tech] * (total_capacity / x_0_total) ** alphas[tech]
                
            results[tech]['cost'].append(cost)
    
    # Convert to DataFrames
    for tech in technologies:
        results[tech] = pd.DataFrame(results[tech])
    
    return results

def generate_first_layer_fragmented_data(
    costs_0, 
    capacities_0, 
    growth_rates, 
    alphas, 
    years, 
    base_year=2023
):
    """
    Generate data points for the first-layer fragmented learning model,
    where technologies of the same type (PEM or ALK) learn together.
    
    In first-layer fragmented model:
    C_western_PEM = C_0_western_PEM * ((x_western_PEM + x_chinese_PEM)/(x_0_PEM))^alpha
    
    Parameters:
    -----------
    costs_0 : dict
        Dictionary of current capital costs ($/kW) for each technology
    capacities_0 : dict
        Dictionary of current installed capacities (MW) for each technology
    growth_rates : dict
        Dictionary of annual growth rates for each technology (fraction)
    alphas : dict
        Dictionary of learning parameters for each technology
    years : int
        Number of years to project
    base_year : int
        Starting year for projections
        
    Returns:
    --------
    dict
        Dictionary with DataFrames for each technology containing year, capacity, and cost
    """
    # Initialize result dictionary
    results = {}
    
    # Get list of technologies
    technologies = list(costs_0.keys())
    
    # Calculate initial total capacities by type
    x_0_pem = capacities_0['western_pem'] + capacities_0['chinese_pem']
    x_0_alk = capacities_0['western_alk'] + capacities_0['chinese_alk']
    
    # Initialize current capacities
    current_capacities = capacities_0.copy()
    
    # Create year range
    year_range = list(range(base_year, base_year + years + 1))
    
    # Initialize data for each technology
    for tech in technologies:
        results[tech] = {
            'year': year_range,
            'capacity': [current_capacities[tech]],
            'cost': [costs_0[tech]]
        }
    
    # Project for each year
    for year_idx, year in enumerate(year_range[1:], 1):  # Skip the base year (already set)
        # Update capacities based on growth rates
        for tech in technologies:
            current_capacities[tech] *= (1 + growth_rates[tech])
            results[tech]['capacity'].append(current_capacities[tech])
        
        # Calculate total PEM and ALK capacities
        pem_capacity = current_capacities['western_pem'] + current_capacities['chinese_pem']
        alk_capacity = current_capacities['western_alk'] + current_capacities['chinese_alk']
        
        # Calculate costs using first-layer fragmented learning
        # PEM technologies
        results['western_pem']['cost'].append(
            costs_0['western_pem'] * (pem_capacity / x_0_pem) ** alphas['western_pem']
        )
        results['chinese_pem']['cost'].append(
            costs_0['chinese_pem'] * (pem_capacity / x_0_pem) ** alphas['chinese_pem']
        )
        
        # ALK technologies
        results['western_alk']['cost'].append(
            costs_0['western_alk'] * (alk_capacity / x_0_alk) ** alphas['western_alk']
        )
        results['chinese_alk']['cost'].append(
            costs_0['chinese_alk'] * (alk_capacity / x_0_alk) ** alphas['chinese_alk']
        )
    
    # Convert to DataFrames
    for tech in technologies:
        results[tech] = pd.DataFrame(results[tech])
    
    return results

def generate_second_layer_fragmented_data(
    costs_0, 
    capacities_0, 
    growth_rates, 
    alphas, 
    years, 
    base_year=2023
):
    """
    Generate data points for the second-layer fragmented learning model,
    where each technology learns independently.
    
    In second-layer fragmented model:
    C_tech = C_0_tech * (x_tech/x_0_tech)^alpha
    
    Parameters:
    -----------
    costs_0 : dict
        Dictionary of current capital costs ($/kW) for each technology
    capacities_0 : dict
        Dictionary of current installed capacities (MW) for each technology
    growth_rates : dict
        Dictionary of annual growth rates for each technology (fraction)
    alphas : dict
        Dictionary of learning parameters for each technology
    years : int
        Number of years to project
    base_year : int
        Starting year for projections
        
    Returns:
    --------
    dict
        Dictionary with DataFrames for each technology containing year, capacity, and cost
    """
    # Initialize result dictionary
    results = {}
    
    # Get list of technologies
    technologies = list(costs_0.keys())
    
    # Initialize current capacities
    current_capacities = capacities_0.copy()
    
    # Create year range
    year_range = list(range(base_year, base_year + years + 1))
    
    # Initialize data for each technology
    for tech in technologies:
        results[tech] = {
            'year': year_range,
            'capacity': [current_capacities[tech]],
            'cost': [costs_0[tech]]
        }
    
    # Project for each year
    for year_idx, year in enumerate(year_range[1:], 1):  # Skip the base year (already set)
        # For each technology, update capacity and calculate cost
        for tech in technologies:
            # Update capacity based on growth rate
            current_capacities[tech] *= (1 + growth_rates[tech])
            results[tech]['capacity'].append(current_capacities[tech])
            
            # Calculate cost using second-layer fragmented learning
            cost = costs_0[tech] * (current_capacities[tech] / capacities_0[tech]) ** alphas[tech]
            results[tech]['cost'].append(cost)
    
    # Convert to DataFrames
    for tech in technologies:
        results[tech] = pd.DataFrame(results[tech])
    
    return results