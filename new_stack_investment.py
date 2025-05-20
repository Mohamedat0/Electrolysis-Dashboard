    with investment_tabs[0]:
        st.subheader("Stack Technology Learning Investments")
        
        # Create technology and cost reduction target selections
        col1, col2 = st.columns(2)
        
        with col1:
            # Create stack technology selection
            selected_stack_tech = st.selectbox(
                "Select Stack Technology",
                options=TECHNOLOGIES,
                format_func=get_stack_display_name,
                key="investment_stack_tech")
            
        with col2:
            # Select target cost reduction percentage
            target_cost_reduction = st.slider(
                "Target Cost Reduction (%)",
                min_value=10,
                max_value=90,
                value=80,
                step=5,
                key="stack_target_reduction",
                help="Target cost reduction percentage (e.g., 80 = 80% reduction from initial cost)")
        
        # Calculate target cost factor based on reduction percentage
        target_cost_factor = 1.0 - (target_cost_reduction / 100.0)
        
        # Display initial cost information
        initial_cost = stack_costs_0[selected_stack_tech]
        target_cost = initial_cost * target_cost_factor
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric(
                "Initial Cost", 
                f"${initial_cost:.0f}/kW", 
                delta=None
            )
        with col2:
            st.metric(
                "Target Cost", 
                f"${target_cost:.0f}/kW", 
                delta=f"-{target_cost_reduction}%",
                delta_color="normal"
            )
        
        # Generate data for all learning models
        learning_models = ['shared', 'first_layer', 'second_layer']
        model_display_names = {
            'shared': 'Shared Learning',
            'first_layer': 'First-layer Fragmentation',
            'second_layer': 'Second-layer Fragmentation'
        }
        
        model_colors = {
            'shared': '#636EFA',  # blue
            'first_layer': '#EF553B',  # red
            'second_layer': '#00CC96',  # green
        }
        
        # Cost Reduction vs. Required Investment Plot
        st.subheader("Learning Investment Curves by Model")
        
        # Create figure
        fig = go.Figure()
        
        # Generate a range of target costs from current cost to target cost
        min_cost = initial_cost * target_cost_factor
        cost_steps = 30  # More points for smoother curve
        
        # Store data for the table
        all_models_data = {}
        
        for model in learning_models:
            # Generate data for this model
            target_cost_data = generate_target_cost_data_stack(
                selected_stack_tech,
                stack_costs_0,
                technologies_capacities_0,
                stack_alphas,
                cost_steps=cost_steps,
                min_cost_factor=target_cost_factor,
                learning_model=model
            )
            
            # Store for table display
            all_models_data[model] = target_cost_data
            
            # Add line trace for this model
            fig.add_trace(
                go.Scatter(
                    x=target_cost_data['target_cost'],
                    y=target_cost_data['learning_investment'] / 1e6,  # Convert to millions
                    mode='lines',
                    name=model_display_names[model],
                    line=dict(color=model_colors[model], width=3),
                    hovertemplate='Target Cost: $%{x:.0f}/kW<br>Investment: $%{y:.1f}M<extra></extra>'
                )
            )
        
        # Add reference line for target cost
        fig.add_vline(
            x=target_cost, 
            line_width=2, 
            line_dash="dash", 
            line_color="gray",
            annotation_text=f"Target: ${target_cost:.0f}/kW",
            annotation_position="top right"
        )
        
        # Format axes
        fig.update_layout(
            title=f"Learning Investment for {get_stack_display_name(selected_stack_tech)} by Model",
            hovermode="closest",
            xaxis=dict(
                title="Target Cost ($/kW)",
                range=[min_cost * 0.9, initial_cost * 1.05],  # Give some padding
                autorange="reversed"  # Higher costs on left, lower on right
            ),
            yaxis=dict(
                title="Required Investment ($ million)",
                tickformat=".1f"
            ),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        # Show the plot
        st.plotly_chart(fig, use_container_width=True)
        
        # Show the investment required to reach target
        st.subheader(f"Investment Required to Reach {target_cost_reduction}% Cost Reduction")
        
        # Create comparison table
        comparison_data = []
        for model in learning_models:
            # Find the row closest to the target cost
            target_data = all_models_data[model]
            target_row = target_data[target_data['target_cost'] <= target_cost].iloc[0]
            
            comparison_data.append({
                'Learning Model': model_display_names[model],
                'Target Cost ($/kW)': target_row['target_cost'],
                'Cost Reduction (%)': target_row['cost_reduction_pct'],
                'Required Capacity (MW)': target_row['required_capacity'],
                'Learning Investment ($ million)': target_row['learning_investment'] / 1e6
            })
        
        # Convert to DataFrame and display
        comparison_df = pd.DataFrame(comparison_data)
        
        # Format the numbers
        comparison_df['Target Cost ($/kW)'] = comparison_df['Target Cost ($/kW)'].round(0).astype(int)
        comparison_df['Cost Reduction (%)'] = comparison_df['Cost Reduction (%)'].round(1)
        comparison_df['Required Capacity (MW)'] = comparison_df['Required Capacity (MW)'].round(0).astype(int)
        comparison_df['Learning Investment ($ million)'] = comparison_df['Learning Investment ($ million)'].round(1)
        
        # Display comparison table
        st.dataframe(
            comparison_df,
            hide_index=True,
            use_container_width=True
        )
        
        # Cost Reduction vs. Learning Investment Efficiency
        st.subheader("Learning Investment Efficiency by Cost Reduction")
        
        # Create figure for efficiency
        fig = go.Figure()
        
        for model in learning_models:
            # Get model data
            model_data = all_models_data[model]
            
            # Calculate investment efficiency: $ per % cost reduction
            model_data['efficiency'] = model_data['learning_investment'] / model_data['cost_reduction_pct'] / 1e6  # $M per % reduction
            
            # Add line trace for efficiency
            fig.add_trace(
                go.Scatter(
                    x=model_data['cost_reduction_pct'],
                    y=model_data['efficiency'],
                    mode='lines',
                    name=model_display_names[model],
                    line=dict(color=model_colors[model], width=3),
                    hovertemplate='Cost Reduction: %{x:.1f}%<br>Efficiency: $%{y:.1f}M per 1%<extra></extra>'
                )
            )
        
        # Add reference line for target cost reduction
        fig.add_vline(
            x=target_cost_reduction, 
            line_width=2, 
            line_dash="dash", 
            line_color="gray",
            annotation_text=f"Target: {target_cost_reduction}%",
            annotation_position="top right"
        )
        
        # Format axes
        fig.update_layout(
            title=f"Investment Efficiency for {get_stack_display_name(selected_stack_tech)} by Model",
            hovermode="closest",
            xaxis=dict(
                title="Cost Reduction (%)",
                range=[0, 100]
            ),
            yaxis=dict(
                title="Investment Efficiency ($ million per 1% reduction)",
                tickformat=".1f"
            ),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        # Show the efficiency plot
        st.plotly_chart(fig, use_container_width=True)
        
        # Explanation and notes
        st.info("""
        **How to interpret these charts:**
        
        - **Learning Investment Curves**: Shows the total investment required to reach different cost targets under each learning model.
        - **Investment Required to Reach Target**: Compares the investment needed to achieve your selected cost reduction target under each model.
        - **Investment Efficiency**: Shows how much investment is needed per 1% cost reduction at different reduction levels.
        
        The **Shared Learning** model assumes knowledge transfer across all technologies. The **First-layer Fragmentation** model assumes knowledge transfer within technology types (PEM or ALK). The **Second-layer Fragmentation** model assumes no knowledge transfer between technologies.
        """)