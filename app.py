import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from utils import (
    calculate_future_cost, 
    alpha_from_learning_rate, 
    calculate_shared_learning_cost,
    calculate_fragmented_learning_cost,
    generate_shared_learning_curve_data,
    generate_fragmented_learning_curve_data
)

# Configure page
st.set_page_config(
    page_title="Water Electrolysis Stack Cost Projections",
    page_icon="⚡",
    layout="wide"
)

# Title and introduction
st.title("Water Electrolysis Stack Cost Projections")
st.markdown("""
This application calculates future capital costs for water electrolysis stack technologies (PEM and Alkaline) 
based on experience curve models. Compare how costs evolve under shared learning vs. fragmented learning scenarios.
""")

# Input parameters in sidebar
with st.sidebar:
    st.subheader("Model Parameters")
    
    # Learning rate inputs
    st.subheader("Learning Rates")
    
    pem_learning_rate = st.slider(
        "PEM Learning Rate (%)",
        min_value=1.0,
        max_value=30.0,
        value=15.0,
        step=1.0,
        help="Percentage reduction in cost for each doubling of capacity for PEM technology"
    )
    
    alk_learning_rate = st.slider(
        "Alkaline Learning Rate (%)",
        min_value=1.0,
        max_value=30.0,
        value=10.0,
        step=1.0,
        help="Percentage reduction in cost for each doubling of capacity for Alkaline technology"
    )
    
    # Convert learning rates to alpha parameters
    pem_alpha = alpha_from_learning_rate(pem_learning_rate)
    alk_alpha = alpha_from_learning_rate(alk_learning_rate)
    
    # Current costs
    st.subheader("Current Costs ($/kW)")
    
    pem_current_cost = st.number_input(
        "PEM Current Cost",
        min_value=100.0,
        max_value=5000.0,
        value=1200.0,
        step=50.0,
        help="Current capital cost of PEM electrolysis stacks in $/kW"
    )
    
    alk_current_cost = st.number_input(
        "Alkaline Current Cost",
        min_value=100.0,
        max_value=5000.0,
        value=900.0,
        step=50.0,
        help="Current capital cost of Alkaline electrolysis stacks in $/kW"
    )
    
    # Current capacities
    st.subheader("Installed Capacity (MW)")
    
    # Current capacities (up to 25 GW = 25,000 MW)
    pem_current_capacity = st.number_input(
        "PEM Current Capacity",
        min_value=1.0,
        max_value=25000.0,
        value=500.0,
        step=50.0,
        help="Current global installed capacity of PEM electrolysis in MW"
    )
    
    alk_current_capacity = st.number_input(
        "Alkaline Current Capacity",
        min_value=1.0,
        max_value=25000.0,
        value=2000.0,
        step=50.0,
        help="Current global installed capacity of Alkaline electrolysis in MW"
    )
    
    # Growth rates instead of future capacities
    st.subheader("Annual Growth Rates (%)")
    
    pem_growth = st.slider(
        "PEM Annual Growth",
        min_value=5.0,
        max_value=100.0,
        value=35.0,
        step=5.0,
        help="Annual growth rate for PEM installed capacity"
    ) / 100.0  # Convert to decimal
    
    alk_growth = st.slider(
        "Alkaline Annual Growth",
        min_value=5.0,
        max_value=100.0,
        value=25.0,
        step=5.0,
        help="Annual growth rate for Alkaline installed capacity"
    ) / 100.0  # Convert to decimal
    
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

# Generate projection data for both models
# Shared Learning Model
pem_shared_data = generate_shared_learning_curve_data(
    pem_current_cost, 
    pem_current_capacity, 
    alk_current_capacity, 
    pem_growth, 
    alk_growth, 
    pem_alpha, 
    projection_years, 
    base_year
)

alk_shared_data = generate_shared_learning_curve_data(
    alk_current_cost, 
    alk_current_capacity, 
    pem_current_capacity, 
    alk_growth, 
    pem_growth, 
    alk_alpha, 
    projection_years, 
    base_year
)

# Fragmented Learning Model
pem_fragmented_data = generate_fragmented_learning_curve_data(
    pem_current_cost, 
    pem_current_capacity, 
    pem_growth, 
    pem_alpha, 
    projection_years, 
    base_year
)

alk_fragmented_data = generate_fragmented_learning_curve_data(
    alk_current_cost, 
    alk_current_capacity, 
    alk_growth, 
    alk_alpha, 
    projection_years, 
    base_year
)

# Create tabs for different sections
tab1, tab2, tab3 = st.tabs(["Cost Projections", "Capacity Growth", "Data Tables"])

with tab1:
    st.header("Cost Projections Comparison")
    
    # Display cost projections side by side
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Shared Learning Model")
        
        # Create cost plot for shared learning
        shared_plot_df = pd.DataFrame({
            'Year': pem_shared_data['year'],
            'PEM Cost ($/kW)': pem_shared_data['cost'],
            'Alkaline Cost ($/kW)': alk_shared_data['cost']
        })
        
        # Melt DataFrame for easier plotting
        shared_plot_melted = pd.melt(
            shared_plot_df,
            id_vars=['Year'],
            value_vars=['PEM Cost ($/kW)', 'Alkaline Cost ($/kW)'],
            var_name='Technology',
            value_name='Cost ($/kW)'
        )
        
        # Create line chart for shared learning
        fig_shared = px.line(
            shared_plot_melted,
            x='Year',
            y='Cost ($/kW)',
            color='Technology',
            markers=True,
            title="Shared Learning Model"
        )
        
        # Style the figure
        fig_shared.update_layout(
            autosize=True,
            height=400,
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        # Set y-axis to start at 0
        fig_shared.update_yaxes(rangemode="tozero")
        
        st.plotly_chart(fig_shared, use_container_width=True)
        
        # Display final projected costs for shared learning
        final_pem_shared = pem_shared_data['cost'].iloc[-1]
        final_alk_shared = alk_shared_data['cost'].iloc[-1]
        
        col1a, col1b = st.columns(2)
        
        with col1a:
            st.metric(
                label="PEM Stack Cost",
                value=f"${final_pem_shared:.0f}/kW",
                delta=f"{((final_pem_shared/pem_current_cost)-1)*100:.1f}%"
            )
        
        with col1b:
            st.metric(
                label="Alkaline Stack Cost",
                value=f"${final_alk_shared:.0f}/kW",
                delta=f"{((final_alk_shared/alk_current_cost)-1)*100:.1f}%"
            )
    
    with col2:
        st.subheader("Fragmented Learning Model")
        
        # Create cost plot for fragmented learning
        frag_plot_df = pd.DataFrame({
            'Year': pem_fragmented_data['year'],
            'PEM Cost ($/kW)': pem_fragmented_data['cost'],
            'Alkaline Cost ($/kW)': alk_fragmented_data['cost']
        })
        
        # Melt DataFrame for easier plotting
        frag_plot_melted = pd.melt(
            frag_plot_df,
            id_vars=['Year'],
            value_vars=['PEM Cost ($/kW)', 'Alkaline Cost ($/kW)'],
            var_name='Technology',
            value_name='Cost ($/kW)'
        )
        
        # Create line chart for fragmented learning
        fig_frag = px.line(
            frag_plot_melted,
            x='Year',
            y='Cost ($/kW)',
            color='Technology',
            markers=True,
            title="Fragmented Learning Model"
        )
        
        # Style the figure
        fig_frag.update_layout(
            autosize=True,
            height=400,
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        # Set y-axis to start at 0
        fig_frag.update_yaxes(rangemode="tozero")
        
        st.plotly_chart(fig_frag, use_container_width=True)
        
        # Display final projected costs for fragmented learning
        final_pem_frag = pem_fragmented_data['cost'].iloc[-1]
        final_alk_frag = alk_fragmented_data['cost'].iloc[-1]
        
        col2a, col2b = st.columns(2)
        
        with col2a:
            st.metric(
                label="PEM Stack Cost",
                value=f"${final_pem_frag:.0f}/kW",
                delta=f"{((final_pem_frag/pem_current_cost)-1)*100:.1f}%"
            )
        
        with col2b:
            st.metric(
                label="Alkaline Stack Cost",
                value=f"${final_alk_frag:.0f}/kW",
                delta=f"{((final_alk_frag/alk_current_cost)-1)*100:.1f}%"
            )
    
    # Cost difference analysis
    st.subheader("Model Comparison")
    st.markdown(f"""
    #### Cost Reduction Comparison (in {base_year + projection_years})
    
    | Technology | Shared Learning | Fragmented Learning | Difference |
    |------------|-----------------|---------------------|------------|
    | PEM | -{(1-final_pem_shared/pem_current_cost)*100:.1f}% | -{(1-final_pem_frag/pem_current_cost)*100:.1f}% | {abs((final_pem_shared-final_pem_frag)/pem_current_cost)*100:.1f}% |
    | Alkaline | -{(1-final_alk_shared/alk_current_cost)*100:.1f}% | -{(1-final_alk_frag/alk_current_cost)*100:.1f}% | {abs((final_alk_shared-final_alk_frag)/alk_current_cost)*100:.1f}% |
    """)

with tab2:
    st.header("Installed Capacity Projections")
    
    # Display capacity projections side by side
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Shared Learning Model Capacity")
        
        # Create capacity dataframe for shared learning (convert MW to GW)
        shared_capacity_df = pd.DataFrame({
            'Year': pem_shared_data['year'],
            'PEM Capacity (GW)': pem_shared_data['capacity_own'] / 1000,  # Convert MW to GW
            'Alkaline Capacity (GW)': alk_shared_data['capacity_own'] / 1000,  # Convert MW to GW
            'Total Capacity (GW)': (pem_shared_data['capacity_own'] + alk_shared_data['capacity_own']) / 1000  # Convert MW to GW
        })
        
        # Create capacity growth chart for shared learning
        fig_shared_cap = px.area(
            shared_capacity_df,
            x='Year',
            y=['PEM Capacity (GW)', 'Alkaline Capacity (GW)'],
            title="Installed Capacity Growth (Shared Learning)",
            labels={"value": "Installed Capacity (GW)", "variable": "Technology"}
        )
        
        # Style the figure
        fig_shared_cap.update_layout(
            autosize=True,
            height=400,
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        st.plotly_chart(fig_shared_cap, use_container_width=True)
        
        # Display final projected capacities for shared learning
        final_pem_cap_shared = pem_shared_data['capacity_own'].iloc[-1]
        final_alk_cap_shared = alk_shared_data['capacity_own'].iloc[-1]
        final_total_cap_shared = final_pem_cap_shared + final_alk_cap_shared
        
        col1a, col1b, col1c = st.columns(3)
        
        with col1a:
            st.metric(
                label="PEM Capacity",
                value=f"{final_pem_cap_shared/1000:.1f} GW",
                delta=f"x{final_pem_cap_shared/pem_current_capacity:.1f}"
            )
        
        with col1b:
            st.metric(
                label="Alkaline Capacity",
                value=f"{final_alk_cap_shared/1000:.1f} GW",
                delta=f"x{final_alk_cap_shared/alk_current_capacity:.1f}"
            )
        
        with col1c:
            st.metric(
                label="Total Capacity",
                value=f"{final_total_cap_shared/1000:.1f} GW",
                delta=f"x{final_total_cap_shared/(pem_current_capacity + alk_current_capacity):.1f}"
            )
    
    with col2:
        st.subheader("Fragmented Learning Model Capacity")
        
        # Create capacity dataframe for fragmented learning (convert MW to GW)
        frag_capacity_df = pd.DataFrame({
            'Year': pem_fragmented_data['year'],
            'PEM Capacity (GW)': pem_fragmented_data['capacity'] / 1000,  # Convert MW to GW
            'Alkaline Capacity (GW)': alk_fragmented_data['capacity'] / 1000,  # Convert MW to GW
            'Total Capacity (GW)': (pem_fragmented_data['capacity'] + alk_fragmented_data['capacity']) / 1000  # Convert MW to GW
        })
        
        # Create capacity growth chart for fragmented learning
        fig_frag_cap = px.area(
            frag_capacity_df,
            x='Year',
            y=['PEM Capacity (GW)', 'Alkaline Capacity (GW)'],
            title="Installed Capacity Growth (Fragmented Learning)",
            labels={"value": "Installed Capacity (GW)", "variable": "Technology"}
        )
        
        # Style the figure
        fig_frag_cap.update_layout(
            autosize=True,
            height=400,
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        st.plotly_chart(fig_frag_cap, use_container_width=True)
        
        # Display final projected capacities for fragmented learning
        final_pem_cap_frag = pem_fragmented_data['capacity'].iloc[-1]
        final_alk_cap_frag = alk_fragmented_data['capacity'].iloc[-1]
        final_total_cap_frag = final_pem_cap_frag + final_alk_cap_frag
        
        col2a, col2b, col2c = st.columns(3)
        
        with col2a:
            st.metric(
                label="PEM Capacity",
                value=f"{final_pem_cap_frag/1000:.1f} GW",
                delta=f"x{final_pem_cap_frag/pem_current_capacity:.1f}"
            )
        
        with col2b:
            st.metric(
                label="Alkaline Capacity",
                value=f"{final_alk_cap_frag/1000:.1f} GW",
                delta=f"x{final_alk_cap_frag/alk_current_capacity:.1f}"
            )
        
        with col2c:
            st.metric(
                label="Total Capacity",
                value=f"{final_total_cap_frag/1000:.1f} GW",
                delta=f"x{final_total_cap_frag/(pem_current_capacity + alk_current_capacity):.1f}"
            )

with tab3:
    st.header("Detailed Projection Data")
    
    # Create tabs for different data tables
    tab3a, tab3b = st.tabs(["Shared Learning", "Fragmented Learning"])
    
    with tab3a:
        # Combine data into single table for shared learning
        shared_data = pd.DataFrame({
            'Year': pem_shared_data['year'],
            'PEM Capacity (MW)': pem_shared_data['capacity_own'],
            'ALK Capacity (MW)': alk_shared_data['capacity_own'],
            'Total Capacity (MW)': pem_shared_data['capacity_own'] + alk_shared_data['capacity_own'],
            'PEM Cost ($/kW)': pem_shared_data['cost'],
            'ALK Cost ($/kW)': alk_shared_data['cost']
        })
        
        # Format numeric columns
        shared_data['PEM Capacity (MW)'] = shared_data['PEM Capacity (MW)'].round(0)
        shared_data['ALK Capacity (MW)'] = shared_data['ALK Capacity (MW)'].round(0)
        shared_data['Total Capacity (MW)'] = shared_data['Total Capacity (MW)'].round(0)
        shared_data['PEM Cost ($/kW)'] = shared_data['PEM Cost ($/kW)'].round(0)
        shared_data['ALK Cost ($/kW)'] = shared_data['ALK Cost ($/kW)'].round(0)
        
        st.dataframe(shared_data, use_container_width=True)
        
        # Download button for shared learning data
        csv_shared = shared_data.to_csv(index=False)
        st.download_button(
            label="Download Shared Learning Data (CSV)",
            data=csv_shared,
            file_name="electrolysis_stack_projections_shared_learning.csv",
            mime="text/csv"
        )
    
    with tab3b:
        # Combine data into single table for fragmented learning
        frag_data = pd.DataFrame({
            'Year': pem_fragmented_data['year'],
            'PEM Capacity (MW)': pem_fragmented_data['capacity'],
            'ALK Capacity (MW)': alk_fragmented_data['capacity'],
            'Total Capacity (MW)': pem_fragmented_data['capacity'] + alk_fragmented_data['capacity'],
            'PEM Cost ($/kW)': pem_fragmented_data['cost'],
            'ALK Cost ($/kW)': alk_fragmented_data['cost']
        })
        
        # Format numeric columns
        frag_data['PEM Capacity (MW)'] = frag_data['PEM Capacity (MW)'].round(0)
        frag_data['ALK Capacity (MW)'] = frag_data['ALK Capacity (MW)'].round(0)
        frag_data['Total Capacity (MW)'] = frag_data['Total Capacity (MW)'].round(0)
        frag_data['PEM Cost ($/kW)'] = frag_data['PEM Cost ($/kW)'].round(0)
        frag_data['ALK Cost ($/kW)'] = frag_data['ALK Cost ($/kW)'].round(0)
        
        st.dataframe(frag_data, use_container_width=True)
        
        # Download button for fragmented learning data
        csv_frag = frag_data.to_csv(index=False)
        st.download_button(
            label="Download Fragmented Learning Data (CSV)",
            data=csv_frag,
            file_name="electrolysis_stack_projections_fragmented_learning.csv",
            mime="text/csv"
        )

# Model explanations
st.header("Understanding the Models")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Shared Learning Model")
    st.markdown("""
    In the **Shared Learning Model**, both PEM and Alkaline electrolysis technologies benefit from each other's deployments.
    The total installed capacity of both technologies combined drives cost reduction, following the formula:
    
    $$C_{PEM} = C_{0,PEM} \cdot \left(\frac{x_{PEM} + x_{ALK}}{x_{0,PEM} + x_{0,ALK}}\right)^\\alpha$$
    
    $$C_{ALK} = C_{0,ALK} \cdot \left(\frac{x_{PEM} + x_{ALK}}{x_{0,PEM} + x_{0,ALK}}\right)^\\alpha$$
    
    This represents a scenario where innovations, manufacturing improvements, and supply chain efficiencies in one technology
    benefit the other technology as well.
    """)

with col2:
    st.subheader("Fragmented Learning Model")
    st.markdown("""
    In the **Fragmented Learning Model**, PEM and Alkaline electrolysis technologies develop independently.
    Each technology's cost reduction is driven only by its own deployment, following the formulas:
    
    $$C_{PEM} = C_{0,PEM} \cdot \left(\frac{x_{PEM}}{x_{0,PEM}}\right)^\\alpha$$
    
    $$C_{ALK} = C_{0,ALK} \cdot \left(\frac{x_{ALK}}{x_{0,ALK}}\right)^\\alpha$$
    
    This represents a scenario where innovations in one technology do not significantly benefit the other technology.
    """)

st.markdown("""
### Learning Rate Interpretation

The learning rate represents how quickly costs decrease for each doubling of installed capacity:
- A learning rate of 10% means costs decrease by 10% for every doubling of capacity
- Higher learning rates (e.g., 15-20%) indicate faster cost reductions
- Lower learning rates (e.g., 5-8%) indicate slower cost reductions

The learning rate is converted to a learning parameter ($\\alpha$) using the formula $\\alpha = log_2(1 - LR)$ where LR is the learning rate expressed as a decimal.
""")

st.info("""
**Note**: These projections are based on historical learning rates and assumed growth trajectories. Actual future costs may differ due to technological breakthroughs, policy changes, supply chain constraints, or other factors not captured in the experience curve model.
""")

# Footer
st.markdown("---")
st.markdown("Water Electrolysis Cost Projection Tool | Developed by Mohamed Atouife | Princeton University")