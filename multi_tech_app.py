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

# Configure page
st.set_page_config(
    page_title="Water Electrolysis Stack Cost Projections - Multiple Technologies",
    page_icon="⚡",
    layout="wide"
)

# Title and introduction
st.title("Water Electrolysis Stack Cost Projections - Multiple Technologies")
st.markdown("""
This application calculates future capital costs for four water electrolysis stack technologies:
- Western PEM
- Chinese PEM
- Western Alkaline
- Chinese Alkaline

Compare how costs evolve under three different learning scenarios:
1. **Shared Learning**: All technologies benefit from collective deployment
2. **Technological Fragmentation**: Technologies of the same type (PEM vs ALK) learn together
3. **Regional Fragmentation**: Each technology learns independently
""")

# Input parameters in sidebar
with st.sidebar:
    st.subheader("Model Parameters")
    
    # Learning rate inputs
    st.subheader("Learning Rates (%)")
    
    col1, col2 = st.columns(2)
    
    with col1:
        western_pem_lr = st.slider(
            "Western PEM",
            min_value=1.0,
            max_value=30.0,
            value=15.0,
            step=1.0,
            help="Percentage reduction in cost for each doubling of capacity"
        )
        
        western_alk_lr = st.slider(
            "Western ALK",
            min_value=1.0,
            max_value=30.0,
            value=10.0,
            step=1.0,
            help="Percentage reduction in cost for each doubling of capacity"
        )
    
    with col2:
        chinese_pem_lr = st.slider(
            "Chinese PEM",
            min_value=1.0,
            max_value=30.0,
            value=18.0,
            step=1.0,
            help="Percentage reduction in cost for each doubling of capacity"
        )
        
        chinese_alk_lr = st.slider(
            "Chinese ALK",
            min_value=1.0,
            max_value=30.0,
            value=12.0,
            step=1.0,
            help="Percentage reduction in cost for each doubling of capacity"
        )
    
    # Convert learning rates to alpha parameters
    alphas = {
        'western_pem': alpha_from_learning_rate(western_pem_lr),
        'chinese_pem': alpha_from_learning_rate(chinese_pem_lr),
        'western_alk': alpha_from_learning_rate(western_alk_lr),
        'chinese_alk': alpha_from_learning_rate(chinese_alk_lr)
    }
    
    # Current costs
    st.subheader("Current Costs ($/kW)")
    
    col1, col2 = st.columns(2)
    
    with col1:
        western_pem_cost = st.number_input(
            "Western PEM",
            min_value=100.0,
            max_value=5000.0,
            value=1200.0,
            step=50.0,
            help="Current capital cost in $/kW"
        )
        
        western_alk_cost = st.number_input(
            "Western ALK",
            min_value=100.0,
            max_value=5000.0,
            value=900.0,
            step=50.0,
            help="Current capital cost in $/kW"
        )
    
    with col2:
        chinese_pem_cost = st.number_input(
            "Chinese PEM",
            min_value=100.0,
            max_value=5000.0,
            value=800.0,
            step=50.0,
            help="Current capital cost in $/kW"
        )
        
        chinese_alk_cost = st.number_input(
            "Chinese ALK",
            min_value=100.0,
            max_value=5000.0,
            value=600.0,
            step=50.0,
            help="Current capital cost in $/kW"
        )
    
    # Dictionary of costs
    costs_0 = {
        'western_pem': western_pem_cost,
        'chinese_pem': chinese_pem_cost,
        'western_alk': western_alk_cost,
        'chinese_alk': chinese_alk_cost
    }
    
    # Current capacities
    st.subheader("Current Installed Capacity (MW)")
    
    col1, col2 = st.columns(2)
    
    with col1:
        western_pem_capacity = st.number_input(
            "Western PEM",
            min_value=1.0,
            max_value=25000.0,
            value=400.0,
            step=50.0,
            help="Current global installed capacity in MW"
        )
        
        western_alk_capacity = st.number_input(
            "Western ALK",
            min_value=1.0,
            max_value=25000.0,
            value=1500.0,
            step=50.0,
            help="Current global installed capacity in MW"
        )
    
    with col2:
        chinese_pem_capacity = st.number_input(
            "Chinese PEM",
            min_value=1.0,
            max_value=25000.0,
            value=100.0,
            step=50.0,
            help="Current global installed capacity in MW"
        )
        
        chinese_alk_capacity = st.number_input(
            "Chinese ALK",
            min_value=1.0,
            max_value=25000.0,
            value=500.0,
            step=50.0,
            help="Current global installed capacity in MW"
        )
    
    # Dictionary of capacities
    capacities_0 = {
        'western_pem': western_pem_capacity,
        'chinese_pem': chinese_pem_capacity,
        'western_alk': western_alk_capacity,
        'chinese_alk': chinese_alk_capacity
    }
    
    # Growth rates
    st.subheader("Annual Growth Rates (%)")
    
    col1, col2 = st.columns(2)
    
    with col1:
        western_pem_growth = st.slider(
            "Western PEM",
            min_value=5.0,
            max_value=100.0,
            value=30.0,
            step=5.0,
            help="Annual growth rate"
        ) / 100.0  # Convert to decimal
        
        western_alk_growth = st.slider(
            "Western ALK",
            min_value=5.0,
            max_value=100.0,
            value=20.0,
            step=5.0,
            help="Annual growth rate"
        ) / 100.0  # Convert to decimal
    
    with col2:
        chinese_pem_growth = st.slider(
            "Chinese PEM",
            min_value=5.0,
            max_value=100.0,
            value=45.0,
            step=5.0,
            help="Annual growth rate"
        ) / 100.0  # Convert to decimal
        
        chinese_alk_growth = st.slider(
            "Chinese ALK",
            min_value=5.0,
            max_value=100.0,
            value=35.0,
            step=5.0,
            help="Annual growth rate"
        ) / 100.0  # Convert to decimal
    
    # Dictionary of growth rates
    growth_rates = {
        'western_pem': western_pem_growth,
        'chinese_pem': chinese_pem_growth,
        'western_alk': western_alk_growth,
        'chinese_alk': chinese_alk_growth
    }
    
    # Projection timeframe
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

# Generate projection data for all three models
# Shared Learning Model
shared_data = generate_shared_learning_data(
    costs_0,
    capacities_0,
    growth_rates,
    alphas,
    projection_years,
    base_year
)

# First-layer Fragmented Learning Model (by technology type: PEM vs ALK)
first_layer_data = generate_first_layer_fragmented_data(
    costs_0,
    capacities_0,
    growth_rates,
    alphas,
    projection_years,
    base_year
)

# Second-layer Fragmented Learning Model (each technology independently)
second_layer_data = generate_second_layer_fragmented_data(
    costs_0,
    capacities_0,
    growth_rates,
    alphas,
    projection_years,
    base_year
)

# Main content - Create tabs for different views
tab1, tab2, tab3, tab4 = st.tabs([
    "Cost Projections by Technology", 
    "Cost Projections by Model",
    "Capacity Growth",
    "Data Tables"
])

# Helper function to create nice display names
def get_display_name(tech):
    parts = tech.split('_')
    return f"{parts[0].capitalize()} {parts[1].upper()}"

# Tab 1: View cost projections grouped by technology
with tab1:
    st.header("Cost Projections by Technology")
    
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
            st.subheader(f"{get_display_name(tech)} Cost Projections")
            
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
                title=f"{get_display_name(tech)} Stack Cost Projections"
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
                    delta=f"{((final_cost_shared/costs_0[tech])-1)*100:.1f}%"
                )
            
            with col2:
                final_cost_first = first_layer_data[tech]['cost'].iloc[-1]
                st.metric(
                    label="Technological Fragmentation",
                    value=f"${final_cost_first:.0f}/kW",
                    delta=f"{((final_cost_first/costs_0[tech])-1)*100:.1f}%"
                )
            
            with col3:
                final_cost_second = second_layer_data[tech]['cost'].iloc[-1]
                st.metric(
                    label="Regional Fragmentation",
                    value=f"${final_cost_second:.0f}/kW",
                    delta=f"{((final_cost_second/costs_0[tech])-1)*100:.1f}%"
                )

# Tab 2: View cost projections grouped by learning model
with tab2:
    st.header("Cost Projections by Learning Model")
    
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
                plot_df[get_display_name(tech)] = model_data[tech]['cost']
            
            # Melt the DataFrame for plotting
            tech_columns = [get_display_name(tech) for tech in technologies]
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
                            label=get_display_name(tech),
                            value=f"${final_cost:.0f}/kW",
                            delta=f"{((final_cost/costs_0[tech])-1)*100:.1f}%"
                        )
                else:
                    with row2_cols[col_idx]:
                        st.metric(
                            label=get_display_name(tech),
                            value=f"${final_cost:.0f}/kW",
                            delta=f"{((final_cost/costs_0[tech])-1)*100:.1f}%"
                        )

# Tab 3: Capacity growth projections
with tab3:
    st.header("Installed Capacity Projections")
    
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
            delta=f"x{final_total/(sum(c/1000 for c in capacities_0.values())):.1f}"
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
with tab4:
    st.header("Detailed Projection Data")
    
    # Create tabs for different models
    data_tabs = st.tabs([
        "Shared Learning", 
        "Technological Fragmentation", 
        "Regional Fragmentation"
    ])
    
    # Helper function to format the data table for a model
    def format_data_table(model_data):
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
        shared_table = format_data_table(shared_data)
        st.dataframe(shared_table, use_container_width=True)
        
        # Download button for shared learning data
        csv_shared = shared_table.to_csv(index=False)
        st.download_button(
            label="Download Shared Learning Data (CSV)",
            data=csv_shared,
            file_name="electrolysis_projections_shared_learning.csv",
            mime="text/csv"
        )
    
    with data_tabs[1]:
        first_layer_table = format_data_table(first_layer_data)
        st.dataframe(first_layer_table, use_container_width=True)
        
        # Download button for first-layer data
        csv_first = first_layer_table.to_csv(index=False)
        st.download_button(
            label="Download Technological Fragmentation Data (CSV)",
            data=csv_first,
            file_name="electrolysis_projections_first_layer.csv",
            mime="text/csv"
        )
    
    with data_tabs[2]:
        second_layer_table = format_data_table(second_layer_data)
        st.dataframe(second_layer_table, use_container_width=True)
        
        # Download button for second-layer data
        csv_second = second_layer_table.to_csv(index=False)
        st.download_button(
            label="Download Regional Fragmentation Data (CSV)",
            data=csv_second,
            file_name="electrolysis_projections_second_layer.csv",
            mime="text/csv"
        )

# Just add a note at the bottom
st.info("""
**Note**: These projections are based on historical learning rates and assumed growth trajectories. 
Actual future costs may differ due to technological breakthroughs, policy changes, supply chain constraints, 
or other factors not captured in the experience curve model.
""")

# Footer
st.markdown("---")
st.markdown("Water Electrolysis Stack Cost Projection Tool")