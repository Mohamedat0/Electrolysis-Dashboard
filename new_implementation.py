# ==================== LEARNING INVESTMENT TAB ====================
with main_tabs[4]:
    st.header("Learning Investment Analysis")

    st.markdown("""
    This tab calculates the learning investments needed to achieve specific cost reduction targets.
    Learning investments represent the financial commitments needed to move technologies down their experience curves.
    
    Unlike the other tabs which project based on growth rates, this analysis shows what capacity and investment 
    would be required to reach specific cost targets.
    """)

    # Create tabs for different views
    investment_tabs = st.tabs([
        "Stack Technologies", "BoP & EPC"
    ])

    # Render the Stack Technology Learning Investments tab
    render_stack_investment_tab(TECHNOLOGIES, get_stack_display_name, 
                                stack_costs_0, technologies_capacities_0, stack_alphas)
                               
    # Render the BoP & EPC Learning Investments tab
    render_bop_epc_investment_tab(get_region_display_name, bop_epc_costs_0, 
                                 regional_capacities_0, bop_epc_alphas, REGIONS)