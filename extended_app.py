import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from multi_tech_utils import (
    alpha_from_learning_rate,
    generate_shared_learning_data,
    generate_first_layer_fragmented_data,
    generate_second_layer_fragmented_data
)
from bop_epc_utils import (
    generate_local_learning_data,
    generate_global_learning_data
)

# Configure page
st.set_page_config(
    page_title="Electrolysis Cost Projections - Extended",
    page_icon="⚡",
    layout="wide"
)

# Title and introduction
st.title("Electrolysis Cost Projections - Extended")
st.markdown("""
This application calculates future capital costs for water electrolysis technologies, including:

1. **Electrolysis Stacks**: Four technologies with three learning models:
   - Western PEM, Chinese PEM, Western ALK, Chinese ALK
   - Learning models: Shared, Technological Fragmentation, Regional Fragmentation

2. **Balance of Plant (BoP) & EPC**: Four regions with two learning models:
   - USA, EU, China, Rest of World
   - Learning models: Local and Global

Use the sidebar to adjust parameters and explore different scenarios.
""")

# Create main tabs for Stack vs BoP+EPC
main_tabs = st.tabs(["Electrolysis Stacks", "Balance of Plant & EPC"])

# ==================== SIDEBAR PARAMETERS ====================
with st.sidebar:
    st.subheader("Model Parameters")

    # Create tabs for Stack vs BoP+EPC parameters
    params_tabs = st.tabs(["Stack Parameters", "BoP & EPC Parameters"])
    
    # ========== STACK PARAMETERS ==========
    with params_tabs[0]:
        # Learning rate inputs for stacks
        st.subheader("Stack Learning Rates (%)")
        
        col1, col2 = st.columns(2)
        
        with col1:
            western_pem_lr = st.slider(
                "Western PEM",
                min_value=1.0,
                max_value=30.0,
                value=15.0,
                step=1.0,
                key="wpem_lr",
                help="Percentage reduction in cost for each doubling of capacity"
            )
            
            western_alk_lr = st.slider(
                "Western ALK",
                min_value=1.0,
                max_value=30.0,
                value=10.0,
                step=1.0,
                key="walk_lr",
                help="Percentage reduction in cost for each doubling of capacity"
            )
        
        with col2:
            chinese_pem_lr = st.slider(
                "Chinese PEM",
                min_value=1.0,
                max_value=30.0,
                value=18.0,
                step=1.0,
                key="cpem_lr",
                help="Percentage reduction in cost for each doubling of capacity"
            )
            
            chinese_alk_lr = st.slider(
                "Chinese ALK",
                min_value=1.0,
                max_value=30.0,
                value=12.0,
                step=1.0,
                key="calk_lr",
                help="Percentage reduction in cost for each doubling of capacity"
            )
        
        # Convert learning rates to alpha parameters for stacks
        stack_alphas = {
            'western_pem': alpha_from_learning_rate(western_pem_lr),
            'chinese_pem': alpha_from_learning_rate(chinese_pem_lr),
            'western_alk': alpha_from_learning_rate(western_alk_lr),
            'chinese_alk': alpha_from_learning_rate(chinese_alk_lr)
        }
        
        # Current costs for stacks
        st.subheader("Stack Current Costs ($/kW)")
        
        col1, col2 = st.columns(2)
        
        with col1:
            western_pem_cost = st.number_input(
                "Western PEM",
                min_value=100.0,
                max_value=5000.0,
                value=1200.0,
                step=50.0,
                key="wpem_cost",
                help="Current capital cost in $/kW"
            )
            
            western_alk_cost = st.number_input(
                "Western ALK",
                min_value=100.0,
                max_value=5000.0,
                value=900.0,
                step=50.0,
                key="walk_cost",
                help="Current capital cost in $/kW"
            )
        
        with col2:
            chinese_pem_cost = st.number_input(
                "Chinese PEM",
                min_value=100.0,
                max_value=5000.0,
                value=800.0,
                step=50.0,
                key="cpem_cost",
                help="Current capital cost in $/kW"
            )
            
            chinese_alk_cost = st.number_input(
                "Chinese ALK",
                min_value=100.0,
                max_value=5000.0,
                value=600.0,
                step=50.0,
                key="calk_cost",
                help="Current capital cost in $/kW"
            )
        
        # Dictionary of costs for stacks
        stack_costs_0 = {
            'western_pem': western_pem_cost,
            'chinese_pem': chinese_pem_cost,
            'western_alk': western_alk_cost,
            'chinese_alk': chinese_alk_cost
        }
        
        # Current capacities for stacks
        st.subheader("Stack Current Capacity (MW)")
        
        col1, col2 = st.columns(2)
        
        with col1:
            western_pem_capacity = st.number_input(
                "Western PEM",
                min_value=1.0,
                max_value=25000.0,
                value=400.0,
                step=50.0,
                key="wpem_cap",
                help="Current global installed capacity in MW"
            )
            
            western_alk_capacity = st.number_input(
                "Western ALK",
                min_value=1.0,
                max_value=25000.0,
                value=1500.0,
                step=50.0,
                key="walk_cap",
                help="Current global installed capacity in MW"
            )
        
        with col2:
            chinese_pem_capacity = st.number_input(
                "Chinese PEM",
                min_value=1.0,
                max_value=25000.0,
                value=100.0,
                step=50.0,
                key="cpem_cap",
                help="Current global installed capacity in MW"
            )
            
            chinese_alk_capacity = st.number_input(
                "Chinese ALK",
                min_value=1.0,
                max_value=25000.0,
                value=500.0,
                step=50.0,
                key="calk_cap",
                help="Current global installed capacity in MW"
            )
        
        # Dictionary of capacities for stacks
        stack_capacities_0 = {
            'western_pem': western_pem_capacity,
            'chinese_pem': chinese_pem_capacity,
            'western_alk': western_alk_capacity,
            'chinese_alk': chinese_alk_capacity
        }
        
        # Growth rates for stacks
        st.subheader("Stack Annual Growth Rates (%)")
        
        col1, col2 = st.columns(2)
        
        with col1:
            western_pem_growth = st.slider(
                "Western PEM",
                min_value=5.0,
                max_value=100.0,
                value=30.0,
                step=5.0,
                key="wpem_growth",
                help="Annual growth rate"
            ) / 100.0  # Convert to decimal
            
            western_alk_growth = st.slider(
                "Western ALK",
                min_value=5.0,
                max_value=100.0,
                value=20.0,
                step=5.0,
                key="walk_growth",
                help="Annual growth rate"
            ) / 100.0  # Convert to decimal
        
        with col2:
            chinese_pem_growth = st.slider(
                "Chinese PEM",
                min_value=5.0,
                max_value=100.0,
                value=45.0,
                step=5.0,
                key="cpem_growth",
                help="Annual growth rate"
            ) / 100.0  # Convert to decimal
            
            chinese_alk_growth = st.slider(
                "Chinese ALK",
                min_value=5.0,
                max_value=100.0,
                value=35.0,
                step=5.0,
                key="calk_growth",
                help="Annual growth rate"
            ) / 100.0  # Convert to decimal
        
        # Dictionary of growth rates for stacks
        stack_growth_rates = {
            'western_pem': western_pem_growth,
            'chinese_pem': chinese_pem_growth,
            'western_alk': western_alk_growth,
            'chinese_alk': chinese_alk_growth
        }
    
    # ========== BOP & EPC PARAMETERS ==========
    with params_tabs[1]:
        # Learning rate inputs for BoP & EPC
        st.subheader("BoP & EPC Learning Rates (%)")
        
        col1, col2 = st.columns(2)
        
        with col1:
            usa_lr = st.slider(
                "USA",
                min_value=1.0,
                max_value=30.0,
                value=8.0,
                step=1.0,
                key="usa_lr",
                help="Percentage reduction in cost for each doubling of capacity"
            )
            
            eu_lr = st.slider(
                "European Union",
                min_value=1.0,
                max_value=30.0,
                value=7.0,
                step=1.0,
                key="eu_lr",
                help="Percentage reduction in cost for each doubling of capacity"
            )
        
        with col2:
            china_lr = st.slider(
                "China",
                min_value=1.0,
                max_value=30.0,
                value=10.0,
                step=1.0,
                key="china_lr",
                help="Percentage reduction in cost for each doubling of capacity"
            )
            
            row_lr = st.slider(
                "Rest of World",
                min_value=1.0,
                max_value=30.0,
                value=6.0,
                step=1.0,
                key="row_lr",
                help="Percentage reduction in cost for each doubling of capacity"
            )
        
        # Convert learning rates to alpha parameters for BoP & EPC
        bop_epc_alphas = {
            'usa': alpha_from_learning_rate(usa_lr),
            'eu': alpha_from_learning_rate(eu_lr),
            'china': alpha_from_learning_rate(china_lr),
            'row': alpha_from_learning_rate(row_lr)
        }
        
        # Current costs for BoP & EPC
        st.subheader("BoP & EPC Current Costs ($/kW)")
        
        col1, col2 = st.columns(2)
        
        with col1:
            usa_cost = st.number_input(
                "USA",
                min_value=100.0,
                max_value=5000.0,
                value=600.0,
                step=50.0,
                key="usa_cost",
                help="Current BoP & EPC cost in $/kW"
            )
            
            eu_cost = st.number_input(
                "European Union",
                min_value=100.0,
                max_value=5000.0,
                value=550.0,
                step=50.0,
                key="eu_cost",
                help="Current BoP & EPC cost in $/kW"
            )
        
        with col2:
            china_cost = st.number_input(
                "China",
                min_value=100.0,
                max_value=5000.0,
                value=350.0,
                step=50.0,
                key="china_cost",
                help="Current BoP & EPC cost in $/kW"
            )
            
            row_cost = st.number_input(
                "Rest of World",
                min_value=100.0,
                max_value=5000.0,
                value=500.0,
                step=50.0,
                key="row_cost",
                help="Current BoP & EPC cost in $/kW"
            )
        
        # Dictionary of costs for BoP & EPC
        bop_epc_costs_0 = {
            'usa': usa_cost,
            'eu': eu_cost,
            'china': china_cost,
            'row': row_cost
        }
        
        # Current capacities for BoP & EPC (using the same capacity as stacks)
        st.subheader("Current Capacity by Region (MW)")
        
        col1, col2 = st.columns(2)
        
        with col1:
            usa_capacity = st.number_input(
                "USA",
                min_value=1.0,
                max_value=25000.0,
                value=300.0,
                step=50.0,
                key="usa_cap",
                help="Current installed capacity in MW"
            )
            
            eu_capacity = st.number_input(
                "European Union",
                min_value=1.0,
                max_value=25000.0,
                value=1600.0,
                step=50.0,
                key="eu_cap",
                help="Current installed capacity in MW"
            )
        
        with col2:
            china_capacity = st.number_input(
                "China",
                min_value=1.0,
                max_value=25000.0,
                value=600.0,
                step=50.0,
                key="china_cap",
                help="Current installed capacity in MW"
            )
            
            row_capacity = st.number_input(
                "Rest of World",
                min_value=1.0,
                max_value=25000.0,
                value=200.0,
                step=50.0,
                key="row_cap",
                help="Current installed capacity in MW"
            )
        
        # Dictionary of capacities for BoP & EPC
        bop_epc_capacities_0 = {
            'usa': usa_capacity,
            'eu': eu_capacity,
            'china': china_capacity,
            'row': row_capacity
        }
        
        # Growth rates for BoP & EPC regions
        st.subheader("Annual Growth Rates by Region (%)")
        
        col1, col2 = st.columns(2)
        
        with col1:
            usa_growth = st.slider(
                "USA",
                min_value=5.0,
                max_value=100.0,
                value=25.0,
                step=5.0,
                key="usa_growth",
                help="Annual growth rate"
            ) / 100.0  # Convert to decimal
            
            eu_growth = st.slider(
                "European Union",
                min_value=5.0,
                max_value=100.0,
                value=30.0,
                step=5.0,
                key="eu_growth",
                help="Annual growth rate"
            ) / 100.0  # Convert to decimal
        
        with col2:
            china_growth = st.slider(
                "China",
                min_value=5.0,
                max_value=100.0,
                value=40.0,
                step=5.0,
                key="china_growth",
                help="Annual growth rate"
            ) / 100.0  # Convert to decimal
            
            row_growth = st.slider(
                "Rest of World",
                min_value=5.0,
                max_value=100.0,
                value=20.0,
                step=5.0,
                key="row_growth",
                help="Annual growth rate"
            ) / 100.0  # Convert to decimal
        
        # Dictionary of growth rates for BoP & EPC
        bop_epc_growth_rates = {
            'usa': usa_growth,
            'eu': eu_growth,
            'china': china_growth,
            'row': row_growth
        }
    
    # ========== COMMON PARAMETERS ==========
    # Projection timeframe (common for both sections)
    st.subheader("Projection Timeframe")
    
    projection_years = st.slider(
        "Projection Years",
        min_value=5,
        max_value=30,
        value=10,
        step=1,
        help="Number of years to project into the future"
    )
    
    base_year = st.number_input(
        "Base Year",
        min_value=2020,
        max_value=2030,
        value=2023,
        step=1,
        help="Starting year for projections"
    )

# ==================== GENERATE DATA ====================
# Generate all projection data

# For Stacks
# Shared Learning Model
shared_data = generate_shared_learning_data(
    stack_costs_0,
    stack_capacities_0,
    stack_growth_rates,
    stack_alphas,
    projection_years,
    base_year
)

# Technological Fragmentation Learning Model (by technology type: PEM vs ALK)
first_layer_data = generate_first_layer_fragmented_data(
    stack_costs_0,
    stack_capacities_0,
    stack_growth_rates,
    stack_alphas,
    projection_years,
    base_year
)

# Regional Fragmentation Learning Model (each technology independently)
second_layer_data = generate_second_layer_fragmented_data(
    stack_costs_0,
    stack_capacities_0,
    stack_growth_rates,
    stack_alphas,
    projection_years,
    base_year
)

# For BoP & EPC
# Local Learning Model (each region independently)
local_data = generate_local_learning_data(
    bop_epc_costs_0,
    bop_epc_capacities_0,
    bop_epc_growth_rates,
    bop_epc_alphas,
    projection_years,
    base_year
)

# Global Learning Model (regions benefit from global capacity)
global_data = generate_global_learning_data(
    bop_epc_costs_0,
    bop_epc_capacities_0,
    bop_epc_growth_rates,
    bop_epc_alphas,
    projection_years,
    base_year
)

# ==================== HELPER FUNCTIONS ====================
# Helper function to create nice display names for stacks
def get_stack_display_name(tech):
    parts = tech.split('_')
    return f"{parts[0].capitalize()} {parts[1].upper()}"

# Helper function to create nice display names for regions
def get_region_display_name(region):
    if region == 'usa':
        return "USA"
    elif region == 'eu':
        return "European Union"
    elif region == 'china':
        return "China"
    elif region == 'row':
        return "Rest of World"
    else:
        return region.upper()

# ==================== MAIN CONTENT ====================
# Electrolysis Stacks Tab Content
with main_tabs[0]:
    st.header("Electrolysis Stack Cost Projections")
    
    # Create tabs for different views
    stack_tabs = st.tabs([
        "Cost Projections by Technology", 
        "Cost Projections by Model",
        "Capacity Growth",
        "Data Tables"
    ])
    
    # Tab 1: View cost projections grouped by technology
    with stack_tabs[0]:
        st.subheader("Cost Projections by Technology")
        
        # Create subtabs for each technology
        tech_tabs = st.tabs([
            "Western PEM", 
            "Chinese PEM", 
            "Western ALK", 
            "Chinese ALK"
        ])
        
        # Map technology keys to their positions in tech_tabs
        tech_tab_map = {
            'western_pem': 0,
            'chinese_pem': 1,
            'western_alk': 2,
            'chinese_alk': 3
        }
        
        # For each technology, show all three models side by side
        for tech, tab_idx in tech_tab_map.items():
            with tech_tabs[tab_idx]:
                st.subheader(f"{get_stack_display_name(tech)} Cost Projections")
                
                # Create a DataFrame for this technology with all three models
                plot_df = pd.DataFrame({
                    'Year': shared_data[tech]['year'],
                    'Shared Learning': shared_data[tech]['cost'],
                    'Technological Fragmentation': first_layer_data[tech]['cost'],
                    'Regional Fragmentation': second_layer_data[tech]['cost']
                })
                
                # Melt the DataFrame for plotting
                plot_melted = pd.melt(
                    plot_df,
                    id_vars=['Year'],
                    value_vars=['Shared Learning', 'Technological Fragmentation', 'Regional Fragmentation'],
                    var_name='Learning Model',
                    value_name='Cost ($/kW)'
                )
                
                # Create line chart
                fig = px.line(
                    plot_melted,
                    x='Year',
                    y='Cost ($/kW)',
                    color='Learning Model',
                    markers=True,
                    title=f"{get_stack_display_name(tech)} Stack Cost Projections"
                )
                
                # Style the figure
                fig.update_layout(
                    autosize=True,
                    height=500,
                    hovermode="x unified",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                
                # Set y-axis to start at 0
                fig.update_yaxes(rangemode="tozero")
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Display final projected costs for all models
                st.subheader(f"Projected Costs in {base_year + projection_years}")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    final_cost_shared = shared_data[tech]['cost'].iloc[-1]
                    st.metric(
                        label="Shared Learning",
                        value=f"${final_cost_shared:.0f}/kW",
                        delta=f"{((final_cost_shared/stack_costs_0[tech])-1)*100:.1f}%"
                    )
                
                with col2:
                    final_cost_first = first_layer_data[tech]['cost'].iloc[-1]
                    st.metric(
                        label="Technological Fragmentation",
                        value=f"${final_cost_first:.0f}/kW",
                        delta=f"{((final_cost_first/stack_costs_0[tech])-1)*100:.1f}%"
                    )
                
                with col3:
                    final_cost_second = second_layer_data[tech]['cost'].iloc[-1]
                    st.metric(
                        label="Regional Fragmentation",
                        value=f"${final_cost_second:.0f}/kW",
                        delta=f"{((final_cost_second/stack_costs_0[tech])-1)*100:.1f}%"
                    )
    
    # Tab 2: View cost projections grouped by learning model
    with stack_tabs[1]:
        st.subheader("Cost Projections by Learning Model")
        
        # Create subtabs for each learning model
        model_tabs = st.tabs([
            "Shared Learning", 
            "Technological Fragmentation", 
            "Regional Fragmentation"
        ])
        
        # Map model data to model tabs
        model_data_map = {
            0: shared_data,
            1: first_layer_data,
            2: second_layer_data
        }
        
        model_names = {
            0: "Shared Learning",
            1: "Technological Fragmentation",
            2: "Regional Fragmentation"
        }
        
        # For each model, show all technologies
        for model_idx, model_data in model_data_map.items():
            with model_tabs[model_idx]:
                st.subheader(f"{model_names[model_idx]} Model")
                
                # Create a DataFrame with all technologies for this model
                technologies = list(model_data.keys())
                plot_df = pd.DataFrame({
                    'Year': model_data[technologies[0]]['year']
                })
                
                # Add cost columns for each technology
                for tech in technologies:
                    plot_df[get_stack_display_name(tech)] = model_data[tech]['cost']
                
                # Melt the DataFrame for plotting
                tech_columns = [get_stack_display_name(tech) for tech in technologies]
                plot_melted = pd.melt(
                    plot_df,
                    id_vars=['Year'],
                    value_vars=tech_columns,
                    var_name='Technology',
                    value_name='Cost ($/kW)'
                )
                
                # Create line chart
                fig = px.line(
                    plot_melted,
                    x='Year',
                    y='Cost ($/kW)',
                    color='Technology',
                    markers=True,
                    title=f"{model_names[model_idx]} Model - Cost Projections"
                )
                
                # Style the figure
                fig.update_layout(
                    autosize=True,
                    height=500,
                    hovermode="x unified",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                
                # Set y-axis to start at 0
                fig.update_yaxes(rangemode="tozero")
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Display final projected costs for all technologies
                st.subheader(f"Projected Costs in {base_year + projection_years}")
                
                # Display metrics in two rows for better spacing
                row1_cols = st.columns(2)
                row2_cols = st.columns(2)
                
                for i, tech in enumerate(technologies):
                    col_idx = i % 2
                    row_idx = i // 2
                    
                    final_cost = model_data[tech]['cost'].iloc[-1]
                    
                    if row_idx == 0:
                        with row1_cols[col_idx]:
                            st.metric(
                                label=get_stack_display_name(tech),
                                value=f"${final_cost:.0f}/kW",
                                delta=f"{((final_cost/stack_costs_0[tech])-1)*100:.1f}%"
                            )
                    else:
                        with row2_cols[col_idx]:
                            st.metric(
                                label=get_stack_display_name(tech),
                                value=f"${final_cost:.0f}/kW",
                                delta=f"{((final_cost/stack_costs_0[tech])-1)*100:.1f}%"
                            )
    
    # Tab 3: Capacity growth projections
    with stack_tabs[2]:
        st.subheader("Installed Capacity Projections")
        
        # Since capacity growth is the same for all three models, use data from any model
        model_data = shared_data
        
        # Create a capacity DataFrame
        capacity_df = pd.DataFrame({
            'Year': model_data['western_pem']['year'],
            'Western PEM (GW)': model_data['western_pem']['capacity'] / 1000,  # Convert MW to GW
            'Chinese PEM (GW)': model_data['chinese_pem']['capacity'] / 1000,  # Convert MW to GW
            'Western ALK (GW)': model_data['western_alk']['capacity'] / 1000,  # Convert MW to GW
            'Chinese ALK (GW)': model_data['chinese_alk']['capacity'] / 1000,  # Convert MW to GW
        })
        
        # Add total capacities by type and region
        capacity_df['Total PEM (GW)'] = capacity_df['Western PEM (GW)'] + capacity_df['Chinese PEM (GW)']
        capacity_df['Total ALK (GW)'] = capacity_df['Western ALK (GW)'] + capacity_df['Chinese ALK (GW)']
        capacity_df['Total Western (GW)'] = capacity_df['Western PEM (GW)'] + capacity_df['Western ALK (GW)']
        capacity_df['Total Chinese (GW)'] = capacity_df['Chinese PEM (GW)'] + capacity_df['Chinese ALK (GW)']
        capacity_df['Total Capacity (GW)'] = (
            capacity_df['Western PEM (GW)'] + 
            capacity_df['Chinese PEM (GW)'] + 
            capacity_df['Western ALK (GW)'] + 
            capacity_df['Chinese ALK (GW)']
        )
        
        # Create capacity visualization tabs
        cap_tabs = st.tabs(["By Technology", "By Type", "By Region", "Total"])
        
        with cap_tabs[0]:
            # Create capacity growth chart for individual technologies
            fig_tech = px.area(
                capacity_df,
                x='Year',
                y=['Western PEM (GW)', 'Chinese PEM (GW)', 'Western ALK (GW)', 'Chinese ALK (GW)'],
                title="Installed Capacity Growth by Technology",
                labels={"value": "Installed Capacity (GW)", "variable": "Technology"}
            )
            
            # Style the figure
            fig_tech.update_layout(
                autosize=True,
                height=500,
                hovermode="x unified",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            
            st.plotly_chart(fig_tech, use_container_width=True)
        
        with cap_tabs[1]:
            # Create capacity growth chart by type (PEM vs ALK)
            fig_type = px.area(
                capacity_df,
                x='Year',
                y=['Total PEM (GW)', 'Total ALK (GW)'],
                title="Installed Capacity Growth by Type",
                labels={"value": "Installed Capacity (GW)", "variable": "Technology Type"}
            )
            
            # Style the figure
            fig_type.update_layout(
                autosize=True,
                height=500,
                hovermode="x unified",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            
            st.plotly_chart(fig_type, use_container_width=True)
        
        with cap_tabs[2]:
            # Create capacity growth chart by region (Western vs Chinese)
            fig_region = px.area(
                capacity_df,
                x='Year',
                y=['Total Western (GW)', 'Total Chinese (GW)'],
                title="Installed Capacity Growth by Region",
                labels={"value": "Installed Capacity (GW)", "variable": "Region"}
            )
            
            # Style the figure
            fig_region.update_layout(
                autosize=True,
                height=500,
                hovermode="x unified",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            
            st.plotly_chart(fig_region, use_container_width=True)
        
        with cap_tabs[3]:
            # Create total capacity growth chart
            fig_total = px.area(
                capacity_df,
                x='Year',
                y=['Total Capacity (GW)'],
                title="Total Installed Capacity Growth",
                labels={"value": "Installed Capacity (GW)", "variable": ""}
            )
            
            # Style the figure
            fig_total.update_layout(
                autosize=True,
                height=500,
                hovermode="x unified",
                showlegend=False
            )
            
            st.plotly_chart(fig_total, use_container_width=True)
        
        # Display final projected capacities
        st.subheader(f"Projected Capacity in {base_year + projection_years}")
        
        # Get final capacities
        final_year_idx = -1
        final_year_data = {
            'western_pem': model_data['western_pem']['capacity'].iloc[final_year_idx] / 1000,  # GW
            'chinese_pem': model_data['chinese_pem']['capacity'].iloc[final_year_idx] / 1000,  # GW
            'western_alk': model_data['western_alk']['capacity'].iloc[final_year_idx] / 1000,  # GW
            'chinese_alk': model_data['chinese_alk']['capacity'].iloc[final_year_idx] / 1000,  # GW
        }
        
        final_total = sum(final_year_data.values())
        final_pem = final_year_data['western_pem'] + final_year_data['chinese_pem']
        final_alk = final_year_data['western_alk'] + final_year_data['chinese_alk']
        final_western = final_year_data['western_pem'] + final_year_data['western_alk']
        final_chinese = final_year_data['chinese_pem'] + final_year_data['chinese_alk']
        
        # Display capacities in a condensed format
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="Total Capacity",
                value=f"{final_total:.1f} GW",
                delta=f"x{final_total/(sum(c/1000 for c in stack_capacities_0.values())):.1f}"
            )
        
        with col2:
            st.metric(
                label="PEM Total",
                value=f"{final_pem:.1f} GW",
                delta=f"{final_pem/final_total*100:.1f}% of total"
            )
        
        with col3:
            st.metric(
                label="ALK Total",
                value=f"{final_alk:.1f} GW",
                delta=f"{final_alk/final_total*100:.1f}% of total"
            )
        
        with col4:
            st.metric(
                label="Western : Chinese",
                value=f"{final_western:.1f} : {final_chinese:.1f} GW",
                delta=f"{final_western/final_total*100:.1f}% : {final_chinese/final_total*100:.1f}%"
            )
    
    # Tab 4: Data tables
    with stack_tabs[3]:
        st.subheader("Detailed Projection Data")
        
        # Create tabs for different models
        data_tabs = st.tabs([
            "Shared Learning", 
            "Technological Fragmentation", 
            "Regional Fragmentation"
        ])
        
        # Helper function to format the data table for a model
        def format_stack_data_table(model_data):
            # Create a combined DataFrame with all technologies
            combined_df = pd.DataFrame({'Year': model_data['western_pem']['year']})
            
            # Add capacities in GW
            combined_df['Western PEM (GW)'] = model_data['western_pem']['capacity'] / 1000
            combined_df['Chinese PEM (GW)'] = model_data['chinese_pem']['capacity'] / 1000
            combined_df['Western ALK (GW)'] = model_data['western_alk']['capacity'] / 1000
            combined_df['Chinese ALK (GW)'] = model_data['chinese_alk']['capacity'] / 1000
            combined_df['Total (GW)'] = (
                combined_df['Western PEM (GW)'] + 
                combined_df['Chinese PEM (GW)'] + 
                combined_df['Western ALK (GW)'] + 
                combined_df['Chinese ALK (GW)']
            )
            
            # Add costs
            combined_df['Western PEM ($/kW)'] = model_data['western_pem']['cost']
            combined_df['Chinese PEM ($/kW)'] = model_data['chinese_pem']['cost']
            combined_df['Western ALK ($/kW)'] = model_data['western_alk']['cost']
            combined_df['Chinese ALK ($/kW)'] = model_data['chinese_alk']['cost']
            
            # Format numeric columns
            for col in combined_df.columns[1:]:
                if '(GW)' in col:
                    combined_df[col] = combined_df[col].round(2)
                elif '($/kW)' in col:
                    combined_df[col] = combined_df[col].round(0)
            
            return combined_df
        
        # Display data tables for each model
        with data_tabs[0]:
            shared_table = format_stack_data_table(shared_data)
            st.dataframe(shared_table, use_container_width=True)
            
            # Download button for shared learning data
            csv_shared = shared_table.to_csv(index=False)
            st.download_button(
                label="Download Shared Learning Data (CSV)",
                data=csv_shared,
                file_name="electrolysis_stacks_shared_learning.csv",
                mime="text/csv"
            )
        
        with data_tabs[1]:
            first_layer_table = format_stack_data_table(first_layer_data)
            st.dataframe(first_layer_table, use_container_width=True)
            
            # Download button for first-layer data
            csv_first = first_layer_table.to_csv(index=False)
            st.download_button(
                label="Download Technological Fragmentation Data (CSV)",
                data=csv_first,
                file_name="electrolysis_stacks_technological_fragmentation.csv",
                mime="text/csv"
            )
        
        with data_tabs[2]:
            second_layer_table = format_stack_data_table(second_layer_data)
            st.dataframe(second_layer_table, use_container_width=True)
            
            # Download button for second-layer data
            csv_second = second_layer_table.to_csv(index=False)
            st.download_button(
                label="Download Regional Fragmentation Data (CSV)",
                data=csv_second,
                file_name="electrolysis_stacks_regional_fragmentation.csv",
                mime="text/csv"
            )

# Balance of Plant & EPC Tab Content
with main_tabs[1]:
    st.header("Balance of Plant & EPC Cost Projections")
    
    # Create tabs for different views
    bop_epc_tabs = st.tabs([
        "Cost Projections by Region", 
        "Cost Projections by Model",
        "Capacity Growth",
        "Data Tables"
    ])
    
    # Tab 1: View cost projections grouped by region
    with bop_epc_tabs[0]:
        st.subheader("Cost Projections by Region")
        
        # Create subtabs for each region
        region_tabs = st.tabs([
            "USA", 
            "European Union", 
            "China", 
            "Rest of World"
        ])
        
        # Map region keys to their positions in region_tabs
        region_tab_map = {
            'usa': 0,
            'eu': 1,
            'china': 2,
            'row': 3
        }
        
        # For each region, show both models side by side
        for region, tab_idx in region_tab_map.items():
            with region_tabs[tab_idx]:
                st.subheader(f"{get_region_display_name(region)} Cost Projections")
                
                # Create a DataFrame for this region with both models
                plot_df = pd.DataFrame({
                    'Year': local_data[region]['year'],
                    'Local Learning': local_data[region]['cost'],
                    'Global Learning': global_data[region]['cost']
                })
                
                # Melt the DataFrame for plotting
                plot_melted = pd.melt(
                    plot_df,
                    id_vars=['Year'],
                    value_vars=['Local Learning', 'Global Learning'],
                    var_name='Learning Model',
                    value_name='Cost ($/kW)'
                )
                
                # Create line chart
                fig = px.line(
                    plot_melted,
                    x='Year',
                    y='Cost ($/kW)',
                    color='Learning Model',
                    markers=True,
                    title=f"{get_region_display_name(region)} BoP & EPC Cost Projections"
                )
                
                # Style the figure
                fig.update_layout(
                    autosize=True,
                    height=500,
                    hovermode="x unified",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                
                # Set y-axis to start at 0
                fig.update_yaxes(rangemode="tozero")
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Display final projected costs for both models
                st.subheader(f"Projected Costs in {base_year + projection_years}")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    final_cost_local = local_data[region]['cost'].iloc[-1]
                    st.metric(
                        label="Local Learning",
                        value=f"${final_cost_local:.0f}/kW",
                        delta=f"{((final_cost_local/bop_epc_costs_0[region])-1)*100:.1f}%"
                    )
                
                with col2:
                    final_cost_global = global_data[region]['cost'].iloc[-1]
                    st.metric(
                        label="Global Learning",
                        value=f"${final_cost_global:.0f}/kW",
                        delta=f"{((final_cost_global/bop_epc_costs_0[region])-1)*100:.1f}%"
                    )
    
    # Tab 2: View cost projections grouped by learning model
    with bop_epc_tabs[1]:
        st.subheader("Cost Projections by Learning Model")
        
        # Create subtabs for each learning model
        model_tabs = st.tabs([
            "Local Learning", 
            "Global Learning"
        ])
        
        # Map model data to model tabs
        model_data_map = {
            0: local_data,
            1: global_data
        }
        
        model_names = {
            0: "Local Learning",
            1: "Global Learning"
        }
        
        # For each model, show all regions
        for model_idx, model_data in model_data_map.items():
            with model_tabs[model_idx]:
                st.subheader(f"{model_names[model_idx]} Model")
                
                # Create a DataFrame with all regions for this model
                regions = list(model_data.keys())
                plot_df = pd.DataFrame({
                    'Year': model_data[regions[0]]['year']
                })
                
                # Add cost columns for each region
                for region in regions:
                    plot_df[get_region_display_name(region)] = model_data[region]['cost']
                
                # Melt the DataFrame for plotting
                region_columns = [get_region_display_name(region) for region in regions]
                plot_melted = pd.melt(
                    plot_df,
                    id_vars=['Year'],
                    value_vars=region_columns,
                    var_name='Region',
                    value_name='Cost ($/kW)'
                )
                
                # Create line chart
                fig = px.line(
                    plot_melted,
                    x='Year',
                    y='Cost ($/kW)',
                    color='Region',
                    markers=True,
                    title=f"{model_names[model_idx]} Model - Cost Projections"
                )
                
                # Style the figure
                fig.update_layout(
                    autosize=True,
                    height=500,
                    hovermode="x unified",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                
                # Set y-axis to start at 0
                fig.update_yaxes(rangemode="tozero")
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Display final projected costs for all regions
                st.subheader(f"Projected Costs in {base_year + projection_years}")
                
                # Display metrics in a row
                cols = st.columns(4)
                
                for i, region in enumerate(regions):
                    final_cost = model_data[region]['cost'].iloc[-1]
                    
                    with cols[i]:
                        st.metric(
                            label=get_region_display_name(region),
                            value=f"${final_cost:.0f}/kW",
                            delta=f"{((final_cost/bop_epc_costs_0[region])-1)*100:.1f}%"
                        )
    
    # Tab 3: Capacity growth projections
    with bop_epc_tabs[2]:
        st.subheader("Installed Capacity Projections by Region")
        
        # Since capacity growth is the same for both models, use data from any model
        model_data = local_data
        
        # Create a capacity DataFrame
        capacity_df = pd.DataFrame({
            'Year': model_data['usa']['year'],
            'USA (GW)': model_data['usa']['capacity'] / 1000,  # Convert MW to GW
            'EU (GW)': model_data['eu']['capacity'] / 1000,  # Convert MW to GW
            'China (GW)': model_data['china']['capacity'] / 1000,  # Convert MW to GW
            'Rest of World (GW)': model_data['row']['capacity'] / 1000,  # Convert MW to GW
        })
        
        # Add total capacity
        capacity_df['Total Capacity (GW)'] = (
            capacity_df['USA (GW)'] + 
            capacity_df['EU (GW)'] + 
            capacity_df['China (GW)'] + 
            capacity_df['Rest of World (GW)']
        )
        
        # Create capacity visualization tabs
        cap_tabs = st.tabs(["By Region", "Total"])
        
        with cap_tabs[0]:
            # Create capacity growth chart for individual regions
            fig_region = px.area(
                capacity_df,
                x='Year',
                y=['USA (GW)', 'EU (GW)', 'China (GW)', 'Rest of World (GW)'],
                title="Installed Capacity Growth by Region",
                labels={"value": "Installed Capacity (GW)", "variable": "Region"}
            )
            
            # Style the figure
            fig_region.update_layout(
                autosize=True,
                height=500,
                hovermode="x unified",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            
            st.plotly_chart(fig_region, use_container_width=True)
        
        with cap_tabs[1]:
            # Create total capacity growth chart
            fig_total = px.line(
                capacity_df,
                x='Year',
                y=['Total Capacity (GW)'],
                title="Total Installed Capacity Growth",
                labels={"value": "Installed Capacity (GW)", "variable": ""}
            )
            
            # Style the figure
            fig_total.update_layout(
                autosize=True,
                height=500,
                hovermode="x unified",
                showlegend=False
            )
            
            st.plotly_chart(fig_total, use_container_width=True)
        
        # Display final projected capacities
        st.subheader(f"Projected Capacity in {base_year + projection_years}")
        
        # Get final capacities
        final_year_idx = -1
        final_year_data = {
            'usa': model_data['usa']['capacity'].iloc[final_year_idx] / 1000,  # GW
            'eu': model_data['eu']['capacity'].iloc[final_year_idx] / 1000,  # GW
            'china': model_data['china']['capacity'].iloc[final_year_idx] / 1000,  # GW
            'row': model_data['row']['capacity'].iloc[final_year_idx] / 1000,  # GW
        }
        
        final_total = sum(final_year_data.values())
        
        # Display capacities in a condensed format
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric(
                label="Total",
                value=f"{final_total:.1f} GW",
                delta=f"x{final_total/(sum(c/1000 for c in bop_epc_capacities_0.values())):.1f}"
            )
        
        with col2:
            st.metric(
                label="USA",
                value=f"{final_year_data['usa']:.1f} GW",
                delta=f"{final_year_data['usa']/final_total*100:.1f}%"
            )
        
        with col3:
            st.metric(
                label="EU",
                value=f"{final_year_data['eu']:.1f} GW",
                delta=f"{final_year_data['eu']/final_total*100:.1f}%"
            )
        
        with col4:
            st.metric(
                label="China",
                value=f"{final_year_data['china']:.1f} GW",
                delta=f"{final_year_data['china']/final_total*100:.1f}%"
            )
        
        with col5:
            st.metric(
                label="Rest of World",
                value=f"{final_year_data['row']:.1f} GW",
                delta=f"{final_year_data['row']/final_total*100:.1f}%"
            )
    
    # Tab 4: Data tables
    with bop_epc_tabs[3]:
        st.subheader("Detailed Projection Data")
        
        # Create tabs for different models
        data_tabs = st.tabs([
            "Local Learning", 
            "Global Learning"
        ])
        
        # Helper function to format the data table for a model
        def format_bop_epc_data_table(model_data):
            # Create a combined DataFrame with all regions
            combined_df = pd.DataFrame({'Year': model_data['usa']['year']})
            
            # Add capacities in GW
            combined_df['USA (GW)'] = model_data['usa']['capacity'] / 1000
            combined_df['EU (GW)'] = model_data['eu']['capacity'] / 1000
            combined_df['China (GW)'] = model_data['china']['capacity'] / 1000
            combined_df['ROW (GW)'] = model_data['row']['capacity'] / 1000
            combined_df['Total (GW)'] = (
                combined_df['USA (GW)'] + 
                combined_df['EU (GW)'] + 
                combined_df['China (GW)'] + 
                combined_df['ROW (GW)']
            )
            
            # Add costs
            combined_df['USA BoP & EPC ($/kW)'] = model_data['usa']['cost']
            combined_df['EU BoP & EPC ($/kW)'] = model_data['eu']['cost']
            combined_df['China BoP & EPC ($/kW)'] = model_data['china']['cost']
            combined_df['ROW BoP & EPC ($/kW)'] = model_data['row']['cost']
            
            # Format numeric columns
            for col in combined_df.columns[1:]:
                if '(GW)' in col:
                    combined_df[col] = combined_df[col].round(2)
                elif '($/kW)' in col:
                    combined_df[col] = combined_df[col].round(0)
            
            return combined_df
        
        # Display data tables for each model
        with data_tabs[0]:
            local_table = format_bop_epc_data_table(local_data)
            st.dataframe(local_table, use_container_width=True)
            
            # Download button for local learning data
            csv_local = local_table.to_csv(index=False)
            st.download_button(
                label="Download Local Learning Data (CSV)",
                data=csv_local,
                file_name="bop_epc_local_learning.csv",
                mime="text/csv"
            )
        
        with data_tabs[1]:
            global_table = format_bop_epc_data_table(global_data)
            st.dataframe(global_table, use_container_width=True)
            
            # Download button for global learning data
            csv_global = global_table.to_csv(index=False)
            st.download_button(
                label="Download Global Learning Data (CSV)",
                data=csv_global,
                file_name="bop_epc_global_learning.csv",
                mime="text/csv"
            )

# Note about projections
st.info("""
**Note**: These projections are based on historical learning rates and assumed growth trajectories. 
Actual future costs may differ due to technological breakthroughs, policy changes, supply chain constraints, 
or other factors not captured in the experience curve model.
""")

# Footer
st.markdown("---")
st.markdown("Electrolysis Cost Projection Tool | Created with Streamlit")