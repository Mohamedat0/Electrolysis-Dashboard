import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from utils import (
    alpha_from_learning_rate,
    calculate_fragmented_learning_cost,
    calculate_shared_learning_cost,
    generate_shared_learning_curve_data,
    generate_fragmented_learning_curve_data
)

# Configure page
st.set_page_config(
    page_title="Electrolysis Stack Cost Comparison",
    page_icon="⚡",
    layout="wide"
)

# Title and introduction
st.title("Electrolysis Stack Cost Comparison: PEM vs. Alkaline")
st.markdown("""
This application compares cost projections for two electrolysis technologies: 
Proton Exchange Membrane (PEM) and Alkaline (ALK) using both shared and fragmented learning models.

- **Shared Learning Model**: Both technologies follow the same experience curve with combined deployed capacity
- **Fragmented Learning Model**: Each technology follows its own independent experience curve

Adjust the parameters to explore different deployment scenarios and learning rates.
""")

# Create tabs for different analysis views
tab1, tab2 = st.tabs(["Technology Comparison", "Deployment Scenarios"])

# Tab 1: Technology Comparison
with tab1:
    # Create multiple columns for inputs
    col1, col2, col3 = st.columns([1, 1, 1])
    
    # PEM Inputs
    with col1:
        st.subheader("PEM Parameters")
        
        pem_cost = st.number_input(
            "Current PEM Cost ($/kW)",
            min_value=100.0,
            max_value=5000.0,
            value=1000.0,
            step=50.0,
            help="Current capital cost of PEM electrolysis stacks in $/kW"
        )
        
        pem_capacity = st.number_input(
            "Current PEM Installed Capacity (MW)",
            min_value=1.0,
            max_value=10000.0,
            value=500.0,
            step=50.0,
            help="Current global installed capacity of PEM electrolysis in MW"
        )
        
        pem_growth = st.slider(
            "PEM Annual Growth Rate (%)",
            min_value=5.0,
            max_value=100.0,
            value=25.0,
            step=5.0,
            help="Projected annual growth rate for PEM electrolysis installations"
        ) / 100.0  # Convert to decimal
    
    # ALK Inputs
    with col2:
        st.subheader("Alkaline Parameters")
        
        alk_cost = st.number_input(
            "Current Alkaline Cost ($/kW)",
            min_value=100.0,
            max_value=5000.0,
            value=800.0,
            step=50.0,
            help="Current capital cost of alkaline electrolysis stacks in $/kW"
        )
        
        alk_capacity = st.number_input(
            "Current Alkaline Installed Capacity (MW)",
            min_value=1.0,
            max_value=10000.0,
            value=2000.0,
            step=50.0,
            help="Current global installed capacity of alkaline electrolysis in MW"
        )
        
        alk_growth = st.slider(
            "Alkaline Annual Growth Rate (%)",
            min_value=5.0,
            max_value=100.0,
            value=20.0,
            step=5.0,
            help="Projected annual growth rate for alkaline electrolysis installations"
        ) / 100.0  # Convert to decimal
    
    # Learning Model Inputs
    with col3:
        st.subheader("Learning Parameters")
        
        learning_rate = st.slider(
            "Learning Rate (%)",
            min_value=1.0,
            max_value=30.0,
            value=15.0,
            step=1.0,
            help="Learning rate: the percentage reduction in cost for each doubling of capacity"
        )
        
        # Convert learning rate to alpha
        alpha = alpha_from_learning_rate(learning_rate)
        
        projection_years = st.slider(
            "Projection Period (years)",
            min_value=5,
            max_value=30,
            value=10,
            step=1,
            help="Number of years to project costs into the future"
        )
        
        base_year = st.number_input(
            "Base Year",
            min_value=2020,
            max_value=2030,
            value=2023,
            step=1,
            help="Starting year for the projection"
        )
    
    # Generate projection data
    # PEM projections
    pem_shared_data = generate_shared_learning_curve_data(
        pem_cost, pem_capacity, alk_capacity, 
        pem_growth, alk_growth, 
        alpha, projection_years, base_year
    )
    
    pem_fragmented_data = generate_fragmented_learning_curve_data(
        pem_cost, pem_capacity, pem_growth, 
        alpha, projection_years, base_year
    )
    
    # ALK projections
    alk_shared_data = generate_shared_learning_curve_data(
        alk_cost, alk_capacity, pem_capacity, 
        alk_growth, pem_growth, 
        alpha, projection_years, base_year
    )
    
    alk_fragmented_data = generate_fragmented_learning_curve_data(
        alk_cost, alk_capacity, alk_growth, 
        alpha, projection_years, base_year
    )
    
    # Display results
    st.subheader("Cost Projection Results")
    
    # Create a combined plot
    fig = go.Figure()
    
    # Add PEM data traces
    fig.add_trace(go.Scatter(
        x=pem_shared_data['year'], 
        y=pem_shared_data['cost'],
        mode='lines+markers',
        name='PEM (Shared Learning)',
        line=dict(color='blue', dash='solid')
    ))
    
    fig.add_trace(go.Scatter(
        x=pem_fragmented_data['year'], 
        y=pem_fragmented_data['cost'],
        mode='lines+markers',
        name='PEM (Fragmented Learning)',
        line=dict(color='blue', dash='dot')
    ))
    
    # Add ALK data traces
    fig.add_trace(go.Scatter(
        x=alk_shared_data['year'], 
        y=alk_shared_data['cost'],
        mode='lines+markers',
        name='Alkaline (Shared Learning)',
        line=dict(color='green', dash='solid')
    ))
    
    fig.add_trace(go.Scatter(
        x=alk_fragmented_data['year'], 
        y=alk_fragmented_data['cost'],
        mode='lines+markers',
        name='Alkaline (Fragmented Learning)',
        line=dict(color='green', dash='dot')
    ))
    
    # Update layout
    fig.update_layout(
        title="Cost Projections: PEM vs. Alkaline",
        xaxis_title="Year",
        yaxis_title="Stack Cost ($/kW)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=600,
        hovermode="x unified"
    )
    
    # Display the figure
    st.plotly_chart(fig, use_container_width=True)
    
    # Show data tables
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.subheader("PEM Cost Projections")
        
        # Combine data from both models for PEM
        pem_comparison = pd.DataFrame({
            'Year': pem_shared_data['year'],
            'Capacity (MW)': pem_shared_data['capacity_own'].round(0),
            'Shared Learning Cost ($/kW)': pem_shared_data['cost'].round(2),
            'Fragmented Learning Cost ($/kW)': pem_fragmented_data['cost'].round(2),
            'Cost Difference ($/kW)': (pem_fragmented_data['cost'] - pem_shared_data['cost']).round(2)
        })
        
        st.dataframe(pem_comparison, use_container_width=True)
    
    with col_b:
        st.subheader("Alkaline Cost Projections")
        
        # Combine data from both models for Alkaline
        alk_comparison = pd.DataFrame({
            'Year': alk_shared_data['year'],
            'Capacity (MW)': alk_shared_data['capacity_own'].round(0),
            'Shared Learning Cost ($/kW)': alk_shared_data['cost'].round(2),
            'Fragmented Learning Cost ($/kW)': alk_fragmented_data['cost'].round(2),
            'Cost Difference ($/kW)': (alk_fragmented_data['cost'] - alk_shared_data['cost']).round(2)
        })
        
        st.dataframe(alk_comparison, use_container_width=True)
    
    # Download buttons for data
    col_c, col_d = st.columns(2)
    
    with col_c:
        pem_csv = pem_comparison.to_csv(index=False)
        st.download_button(
            label="Download PEM Projection Data (CSV)",
            data=pem_csv,
            file_name="pem_cost_projections.csv",
            mime="text/csv"
        )
    
    with col_d:
        alk_csv = alk_comparison.to_csv(index=False)
        st.download_button(
            label="Download Alkaline Projection Data (CSV)",
            data=alk_csv,
            file_name="alkaline_cost_projections.csv",
            mime="text/csv"
        )

# Tab 2: Deployment Scenarios
with tab2:
    st.subheader("Deployment Scenario Analysis")
    
    st.markdown("""
    This section allows you to explore how different deployment scenarios affect the cost projections
    under both shared and fragmented learning models.
    """)
    
    # Create columns for inputs
    col1, col2 = st.columns([1, 2])
    
    with col1:
        # Common parameters
        st.subheader("Learning Parameters")
        
        scenario_learning_rate = st.slider(
            "Learning Rate (%)",
            min_value=1.0,
            max_value=30.0,
            value=15.0,
            step=1.0,
            key="scenario_lr"
        )
        
        # Convert learning rate to alpha
        scenario_alpha = alpha_from_learning_rate(scenario_learning_rate)
        
        scenario_years = st.slider(
            "Projection Period (years)",
            min_value=5,
            max_value=30,
            value=20,
            step=1,
            key="scenario_years",
            help="Number of years to project costs into the future"
        )
        
        scenario_base_year = st.number_input(
            "Base Year",
            min_value=2020,
            max_value=2030,
            value=2023,
            step=1,
            key="scenario_base_year"
        )
        
        # Stack parameters
        st.subheader("Technology Parameters")
        
        # PEM parameters
        pem_scenario_cost = st.number_input(
            "Current PEM Cost ($/kW)",
            min_value=100.0,
            max_value=5000.0,
            value=1000.0,
            step=50.0,
            key="pem_scenario_cost"
        )
        
        pem_scenario_capacity = st.number_input(
            "Current PEM Capacity (MW)",
            min_value=1.0,
            max_value=10000.0,
            value=500.0,
            step=50.0,
            key="pem_scenario_capacity"
        )
        
        # ALK parameters
        alk_scenario_cost = st.number_input(
            "Current Alkaline Cost ($/kW)",
            min_value=100.0,
            max_value=5000.0,
            value=800.0,
            step=50.0,
            key="alk_scenario_cost"
        )
        
        alk_scenario_capacity = st.number_input(
            "Current Alkaline Capacity (MW)",
            min_value=1.0,
            max_value=10000.0,
            value=2000.0,
            step=50.0,
            key="alk_scenario_capacity"
        )
    
    with col2:
        # Scenario selection
        st.subheader("Deployment Scenarios")
        
        scenarios = {
            "Balanced Growth": {
                "pem_growth": 0.25,
                "alk_growth": 0.25,
                "description": "Both PEM and Alkaline grow at the same rate (25% annually)"
            },
            "PEM Dominant": {
                "pem_growth": 0.40,
                "alk_growth": 0.15,
                "description": "PEM grows faster (40% annually) than Alkaline (15% annually)"
            },
            "Alkaline Dominant": {
                "pem_growth": 0.15,
                "alk_growth": 0.40,
                "description": "Alkaline grows faster (40% annually) than PEM (15% annually)"
            },
            "High Growth": {
                "pem_growth": 0.50,
                "alk_growth": 0.50,
                "description": "Both technologies grow rapidly (50% annually)"
            },
            "Low Growth": {
                "pem_growth": 0.10,
                "alk_growth": 0.10,
                "description": "Both technologies grow slowly (10% annually)"
            }
        }
        
        # Display scenario descriptions
        for scenario, details in scenarios.items():
            st.markdown(f"**{scenario}**: {details['description']}")
        
        # Select scenarios to compare
        selected_scenarios = st.multiselect(
            "Select scenarios to compare",
            options=list(scenarios.keys()),
            default=["Balanced Growth", "PEM Dominant", "Alkaline Dominant"]
        )
    
    # Generate and display scenario data
    if selected_scenarios:
        # Create figures for PEM and ALK
        fig_pem = go.Figure()
        fig_alk = go.Figure()
        
        # Line styles
        line_styles = ['solid', 'dash', 'dot', 'dashdot', 'longdash']
        colors = ['blue', 'green', 'red', 'purple', 'orange']
        
        # Generate data for each selected scenario
        scenario_data = {}
        
        for i, scenario in enumerate(selected_scenarios):
            # Get growth rates for this scenario
            pem_growth = scenarios[scenario]["pem_growth"]
            alk_growth = scenarios[scenario]["alk_growth"]
            
            # Generate data
            pem_shared = generate_shared_learning_curve_data(
                pem_scenario_cost, pem_scenario_capacity, alk_scenario_capacity,
                pem_growth, alk_growth, 
                scenario_alpha, scenario_years, scenario_base_year
            )
            
            pem_fragmented = generate_fragmented_learning_curve_data(
                pem_scenario_cost, pem_scenario_capacity, pem_growth,
                scenario_alpha, scenario_years, scenario_base_year
            )
            
            alk_shared = generate_shared_learning_curve_data(
                alk_scenario_cost, alk_scenario_capacity, pem_scenario_capacity,
                alk_growth, pem_growth, 
                scenario_alpha, scenario_years, scenario_base_year
            )
            
            alk_fragmented = generate_fragmented_learning_curve_data(
                alk_scenario_cost, alk_scenario_capacity, alk_growth,
                scenario_alpha, scenario_years, scenario_base_year
            )
            
            # Store data for tables
            scenario_data[scenario] = {
                'pem_shared': pem_shared,
                'pem_fragmented': pem_fragmented,
                'alk_shared': alk_shared,
                'alk_fragmented': alk_fragmented
            }
            
            # Add traces to PEM figure
            line_style = line_styles[i % len(line_styles)]
            color = colors[i % len(colors)]
            
            fig_pem.add_trace(go.Scatter(
                x=pem_shared['year'], 
                y=pem_shared['cost'],
                mode='lines',
                name=f'{scenario} - Shared',
                line=dict(color=color, dash='solid')
            ))
            
            fig_pem.add_trace(go.Scatter(
                x=pem_fragmented['year'], 
                y=pem_fragmented['cost'],
                mode='lines',
                name=f'{scenario} - Fragmented',
                line=dict(color=color, dash='dash')
            ))
            
            # Add traces to ALK figure
            fig_alk.add_trace(go.Scatter(
                x=alk_shared['year'], 
                y=alk_shared['cost'],
                mode='lines',
                name=f'{scenario} - Shared',
                line=dict(color=color, dash='solid')
            ))
            
            fig_alk.add_trace(go.Scatter(
                x=alk_fragmented['year'], 
                y=alk_fragmented['cost'],
                mode='lines',
                name=f'{scenario} - Fragmented',
                line=dict(color=color, dash='dash')
            ))
        
        # Update layouts
        fig_pem.update_layout(
            title="PEM Cost Projections by Scenario",
            xaxis_title="Year",
            yaxis_title="PEM Stack Cost ($/kW)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            height=450,
            hovermode="x unified"
        )
        
        fig_alk.update_layout(
            title="Alkaline Cost Projections by Scenario",
            xaxis_title="Year",
            yaxis_title="Alkaline Stack Cost ($/kW)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            height=450,
            hovermode="x unified"
        )
        
        # Display charts
        st.plotly_chart(fig_pem, use_container_width=True)
        st.plotly_chart(fig_alk, use_container_width=True)
        
        # Display tables for the final year of projection
        st.subheader(f"Cost Comparison in {scenario_base_year + scenario_years}")
        
        final_year_data = []
        
        for scenario in selected_scenarios:
            pem_shared_final = scenario_data[scenario]['pem_shared'].iloc[-1]
            pem_frag_final = scenario_data[scenario]['pem_fragmented'].iloc[-1]
            alk_shared_final = scenario_data[scenario]['alk_shared'].iloc[-1]
            alk_frag_final = scenario_data[scenario]['alk_fragmented'].iloc[-1]
            
            final_year_data.append({
                'Scenario': scenario,
                'PEM Cost (Shared) [$/kW]': round(pem_shared_final['cost'], 2),
                'PEM Cost (Fragmented) [$/kW]': round(pem_frag_final['cost'], 2),
                'PEM Cost Difference [$/kW]': round(pem_frag_final['cost'] - pem_shared_final['cost'], 2),
                'ALK Cost (Shared) [$/kW]': round(alk_shared_final['cost'], 2),
                'ALK Cost (Fragmented) [$/kW]': round(alk_frag_final['cost'], 2),
                'ALK Cost Difference [$/kW]': round(alk_frag_final['cost'] - alk_shared_final['cost'], 2),
                'Total Capacity [MW]': round(pem_shared_final['capacity_total'], 0)
            })
        
        # Create and display the table
        final_year_df = pd.DataFrame(final_year_data)
        st.dataframe(final_year_df, use_container_width=True)
        
        # Download button for scenario comparison
        scenarios_csv = final_year_df.to_csv(index=False)
        st.download_button(
            label="Download Scenario Comparison (CSV)",
            data=scenarios_csv,
            file_name="electrolysis_scenario_comparison.csv",
            mime="text/csv"
        )

# Explanation section
st.subheader("Understanding the Learning Models")
st.markdown("""
### Shared Learning Model
In the shared learning model, the experience gained from deploying both PEM and Alkaline 
electrolyzers contributes to cost reduction for each technology. The total deployed 
capacity (PEM + Alkaline) is used in the experience curve formula:

$$C_y = C_0 \cdot \left(\frac{x_y^{total}}{x_0}\right)^\\alpha$$

Where $x_y^{total}$ is the sum of both PEM and Alkaline capacity.

### Fragmented Learning Model
In the fragmented learning model, each technology follows its own independent learning curve. 
Only the deployment of the same technology contributes to cost reduction:

$$C_y = C_0 \cdot \left(\frac{x_y^{own}}{x_0}\right)^\\alpha$$

Where $x_y^{own}$ is only the capacity of the same technology.

### Learning Rate
The learning rate represents the percentage reduction in cost for each doubling of installed capacity.
A learning rate of 15% means costs decrease by 15% each time the installed capacity doubles.

### Policy Implications
- **Shared Learning**: Promotes technological diversity without penalizing newer technologies
- **Fragmented Learning**: May create lock-in effects where dominant technologies gain further advantages
""")

st.info("""
**Note**: The model assumes a constant learning rate over time. In reality, learning rates may vary 
as technologies mature or encounter technical limitations.
""")

# Footer
st.markdown("---")
st.markdown("Electrolysis Stack Cost Comparison Tool | Created with Streamlit")