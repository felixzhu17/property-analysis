import streamlit as st
import pandas as pd
import numpy as np
import traceback

# Set page config
st.set_page_config(
    page_title="Sydney Property Dashboard",
    page_icon="🏠",
    layout="wide"
)

# Add title
st.title("Sydney Property Data Dashboard")

@st.cache_data
def load_and_process_data():
    # Read the Excel file
    df = pd.read_csv("Sydney-property-masterdata.csv")
    
    # Convert numeric columns and handle missing values
    numeric_columns = ['Price', 'Rent', 'Yield', 'DoM', 'PΔ1Y', 'Vacancy Rate', 
                      'Clearance Rate', 'Overall RCS', 'ROI Low', 'ROI High',
                      'Capital Growth', 'Cashflow', 'Flood', 'Bushfire', 'Lower Risk',
                      'UH Ratio', 'BA Ratio', 'PΔ1M', 'PΔ1Q', 'PΔ6M', 'PΔ3Y', 'PΔ5Y', 'PΔ10Y',
                      'RΔ1M', 'RΔ1Q', 'RΔ6M', 'RΔ1Y', 'RΔ3Y', 'RΔ5Y', 'RΔ10Y',
                      'CG Low', 'CG High', 'Volatility Index', 'Growth Rate', 'GRC Index',
                      'IRSAD', 'Affordability Index', 'SoM%', 'Inventory', 'Hold Period',
                      'School Rank', 'Infra. Spend', 'EDI', 'MADI']
    
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Fill NaN values with 0 for numeric columns
    df[numeric_columns] = df[numeric_columns].fillna(0)
    
    # Convert decimal percentages to actual percentages (multiply by 100)
    percentage_columns = ['Yield', 'PΔ1Y', 'Vacancy Rate', 'Clearance Rate', 
                         'ROI Low', 'ROI High', 'PΔ1M', 'PΔ1Q', 'PΔ6M', 'PΔ3Y', 'PΔ5Y', 'PΔ10Y',
                         'RΔ1M', 'RΔ1Q', 'RΔ6M', 'RΔ1Y', 'RΔ3Y', 'RΔ5Y', 'RΔ10Y']
    for col in percentage_columns:
        if col in df.columns:
            df[col] = df[col] * 100
    
    # Calculate average ROI for display
    df['ROI Avg'] = (df['ROI Low'] + df['ROI High']) / 2
    
    # Handle Bedrooms column - keep 'All' as is, convert numbers to integers
    df['Bedrooms_Display'] = df['Bedrooms'].copy()  # Keep original values for display
    df['Bedrooms_Numeric'] = pd.to_numeric(df['Bedrooms'], errors='coerce')
    
    return df

# Function to normalize metrics to 0-100 scale
def normalize_metric(df, metric, is_higher_better=True, max_cap=None, min_floor=None):
    # Create a copy of the column to work with
    values = df[metric].copy()
    
    # Apply caps and floors if specified
    if max_cap is not None:
        values = values.clip(upper=max_cap)
    if min_floor is not None:
        values = values.clip(lower=min_floor)
    
    # Get min and max for normalization
    min_val = values.min()
    max_val = values.max()
    
    # Avoid division by zero
    if max_val == min_val:
        return values * 0  # Return zeros if all values are the same
    
    # Normalize to 0-100 scale
    if is_higher_better:
        normalized = 100 * (values - min_val) / (max_val - min_val)
    else:
        # Invert the scale for metrics where lower is better
        normalized = 100 * (max_val - values) / (max_val - min_val)
    
    return normalized

try:
    # Load the processed data
    df = load_and_process_data()
    
    # Sidebar filters
    st.sidebar.header("Filters")
    
    # Area search with autocomplete
    area_search = st.sidebar.selectbox(
        "Search by Area",
        options=[""] + sorted(df["Area"].unique().tolist()),
        index=0
    )
    
    # Property Type filter
    property_types = [""] + sorted(df["Property Type"].unique().tolist())
    selected_type = st.sidebar.selectbox("Property Type", property_types)
    
    # Bedroom filter - include both numeric values and 'All'
    bedroom_values = df['Bedrooms_Display'].unique()
    # Convert to strings for comparison and sort properly
    bedroom_values = [str(x) for x in bedroom_values if pd.notna(x)]
    bedroom_values = sorted([x for x in bedroom_values if x == 'All' or x.replace('.', '').isdigit()],
                          key=lambda x: -1 if x == 'All' else float(x))
    bedrooms = [""] + bedroom_values
    selected_bedrooms = st.sidebar.selectbox("Number of Bedrooms", bedrooms)
    
    # Price range filter with proper numeric handling
    valid_prices = df['Price'][df['Price'] > 0]  # Filter out zeros and invalid values
    min_price = float(valid_prices.min()) if not valid_prices.empty else 0
    max_price = float(valid_prices.max()) if not valid_prices.empty else 1000000
    
    price_range = st.sidebar.slider(
        "Price Range ($)",
        min_value=min_price,
        max_value=max_price,
        value=(min_price, max_price)
    )
    
    # Custom Metric Calculator - New Addition
    st.sidebar.header("Custom Score Calculator")
    with st.sidebar.expander("Customize your property score"):
        st.write("Select metrics and assign weights to calculate your custom property score.")
        
        # Define available metrics and their properties for normalization
        metric_options = {
            'ROI Avg': {'label': 'Return on Investment (Avg)', 'is_higher_better': True, 'max_cap': 25},
            'Yield': {'label': 'Rental Yield', 'is_higher_better': True, 'max_cap': 10},
            'PΔ1Y': {'label': '1 Year Price Growth', 'is_higher_better': True},
            'Capital Growth': {'label': 'Capital Growth Potential', 'is_higher_better': True},
            'Cashflow': {'label': 'Cashflow Rating', 'is_higher_better': True},
            'DoM': {'label': 'Days on Market', 'is_higher_better': False, 'max_cap': 90},
            'Vacancy Rate': {'label': 'Vacancy Rate', 'is_higher_better': False, 'max_cap': 10},
            'Clearance Rate': {'label': 'Clearance Rate', 'is_higher_better': True},
            'Overall RCS': {'label': 'Overall RCS Score', 'is_higher_better': True},
            'Risk Score': {'label': 'Risk Score (Combined)', 'is_higher_better': False},
            'UH Ratio': {'label': 'Units to Houses Ratio', 'is_higher_better': False, 'max_cap': 5},
            'BA Ratio': {'label': 'Building Approvals Ratio', 'is_higher_better': False, 'max_cap': 0.05},
            
            # Additional price growth metrics
            'PΔ1M': {'label': '1 Month Price Growth', 'is_higher_better': True},
            'PΔ1Q': {'label': '1 Quarter Price Growth', 'is_higher_better': True},
            'PΔ6M': {'label': '6 Month Price Growth', 'is_higher_better': True},
            'PΔ3Y': {'label': '3 Year Price Growth', 'is_higher_better': True},
            'PΔ5Y': {'label': '5 Year Price Growth', 'is_higher_better': True},
            'PΔ10Y': {'label': '10 Year Price Growth', 'is_higher_better': True},
            
            # Rent growth metrics
            'RΔ1M': {'label': '1 Month Rent Growth', 'is_higher_better': True},
            'RΔ1Q': {'label': '1 Quarter Rent Growth', 'is_higher_better': True},
            'RΔ6M': {'label': '6 Month Rent Growth', 'is_higher_better': True},
            'RΔ1Y': {'label': '1 Year Rent Growth', 'is_higher_better': True},
            'RΔ3Y': {'label': '3 Year Rent Growth', 'is_higher_better': True},
            'RΔ5Y': {'label': '5 Year Rent Growth', 'is_higher_better': True},
            'RΔ10Y': {'label': '10 Year Rent Growth', 'is_higher_better': True},
            
            # Capital growth and market metrics
            'CG Low': {'label': 'Capital Growth (Low Estimate)', 'is_higher_better': True},
            'CG High': {'label': 'Capital Growth (High Estimate)', 'is_higher_better': True},
            'Volatility Index': {'label': 'Price Volatility', 'is_higher_better': False, 'max_cap': 10},
            'Growth Rate': {'label': 'Growth Rate', 'is_higher_better': True},
            'GRC Index': {'label': 'Growth Rate Cycle Index', 'is_higher_better': True},
            
            # Socioeconomic and market health metrics
            'IRSAD': {'label': 'Socioeconomic Advantage', 'is_higher_better': True, 'max_cap': 10},
            'Affordability Index': {'label': 'Affordability Index', 'is_higher_better': True},
            'SoM%': {'label': 'Sales on Market %', 'is_higher_better': False},
            'Inventory': {'label': 'Property Inventory Level', 'is_higher_better': False},
            'Hold Period': {'label': 'Average Hold Period (Years)', 'is_higher_better': True},
            
            # Amenity and infrastructure metrics
            'School Rank': {'label': 'School Ranking', 'is_higher_better': True, 'max_cap': 100},
            'Infra. Spend': {'label': 'Infrastructure Spending', 'is_higher_better': True},
            
            # Additional risk and economic metrics
            'EDI': {'label': 'Economic Diversity Index', 'is_higher_better': True, 'max_cap': 100},
            'MADI': {'label': 'Market Activity Diversity Index', 'is_higher_better': True, 'max_cap': 100},
        }
        
        # Multi-select for metrics
        selected_metrics = st.multiselect(
            "Select metrics to include",
            options=list(metric_options.keys()),
            default=['ROI Avg', 'Yield', 'PΔ1Y', 'PΔ5Y', 'Vacancy Rate', 'Overall RCS']  # Updated default metrics
        )
        
        # Container for weight sliders
        weights = {}
        if selected_metrics:
            st.write("Assign weights to each metric (weights will be normalized to sum to 100%)")
            total_raw_weight = 0
            
            # Create a slider for each selected metric
            for metric in selected_metrics:
                # Default to equal weights
                default_weight = 100 / len(selected_metrics)
                weights[metric] = st.slider(
                    f"Weight for {metric_options[metric]['label']}",
                    min_value=0.0,
                    max_value=100.0,
                    value=default_weight,
                    step=1.0,
                    format="%.0f%%"
                )
                total_raw_weight += weights[metric]
            
            # Normalize weights to sum to 100%
            if total_raw_weight > 0:
                for metric in weights:
                    weights[metric] = weights[metric] / total_raw_weight
            
            # Show the normalized weights
            st.write("Normalized weights:")
            for metric in weights:
                st.write(f"{metric_options[metric]['label']}: {weights[metric]*100:.1f}%")
        
        # Reset button
        if st.button("Reset to Default"):
            selected_metrics = ['ROI Avg', 'Yield', 'PΔ1Y', 'PΔ5Y', 'Vacancy Rate', 'Overall RCS']
            # Distribute weights evenly
            for metric in selected_metrics:
                weights[metric] = 1.0 / len(selected_metrics)
    
    # Calculate scores on the full dataset before filtering
    score_df = df.copy()
    
    # Calculate custom score based on selected metrics and weights for ALL rows
    if selected_metrics and not score_df.empty:
        # Calculate a combined risk score if selected
        if 'Risk Score' in selected_metrics:
            # Average of the inverse of risk scores (lower risk is better)
            risk_cols = ['Flood', 'Bushfire', 'Lower Risk']
            score_df['Risk Score'] = score_df[risk_cols].mean(axis=1)
        
        # Initialize custom score column
        score_df['Custom Score'] = 0
        
        # Normalize and apply weights for each metric
        for metric in selected_metrics:
            # Use the properties defined in metric_options
            props = metric_options[metric]
            normalized_values = normalize_metric(
                score_df, 
                metric, 
                is_higher_better=props.get('is_higher_better', True),
                max_cap=props.get('max_cap', None),
                min_floor=props.get('min_floor', None)
            )
            
            # Add weighted contribution to custom score
            score_df['Custom Score'] += weights[metric] * normalized_values
        
        # Calculate custom percentiles by property type for the ENTIRE dataset
        def calculate_custom_percentile(group):
            group['Score Percentile'] = group['Custom Score'].rank(pct=True) * 100
            return group
        
        # Use level=None to ensure we're using the column, not an index level
        score_df = score_df.groupby('Property Type', level=None).apply(calculate_custom_percentile)
        score_df = score_df.reset_index(drop=True)
    else:
        # Default calculation if no metrics are selected
        # Set all scores to 0 instead of using ROI and RCS
        if not score_df.empty:
            # Set the default score to 0
            score_df['Temp_Default_Score'] = 0
            
            # Calculate percentiles based on default score (all will be equal)
            def calculate_default_percentile(group):
                # Since all scores are 0, all percentiles will be the same
                group['Score Percentile'] = 50  # Set to middle percentile
                return group
            
            score_df = score_df.groupby('Property Type', level=None).apply(calculate_default_percentile)
            score_df = score_df.reset_index(drop=True)
            
            # Store the default score as Custom Score for consistency
            score_df['Custom Score'] = score_df['Temp_Default_Score']
            
            # Remove the temporary score column
            score_df = score_df.drop('Temp_Default_Score', axis=1)
    
    # Now apply filters to show the subset of data
    filtered_df = score_df.copy()  # Start with the scored dataframe
    if area_search:
        filtered_df = filtered_df[filtered_df["Area"] == area_search]
    if selected_type:
        filtered_df = filtered_df[filtered_df["Property Type"] == selected_type]
    if selected_bedrooms:
        if selected_bedrooms == 'All':
            pass  # Don't filter if 'All' is selected
        else:
            # Convert selected_bedrooms to float for comparison
            filtered_df = filtered_df[filtered_df["Bedrooms_Numeric"] == float(selected_bedrooms)]
    filtered_df = filtered_df[
        (filtered_df["Price"] >= price_range[0]) & 
        (filtered_df["Price"] <= price_range[1])
    ]
    
    # Main content area - using columns for layout
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Key Metrics")
        if not filtered_df.empty:
            st.metric("Median Price", f"${filtered_df['Price'].median():,.0f}")
            st.metric("Median Rent", f"${filtered_df['Rent'].median():,.0f}")
            st.metric("Average Yield", f"{filtered_df['Yield'].mean():.1f}%")
    
    with col2:
        st.subheader("Market Performance")
        if not filtered_df.empty:
            st.metric("1 Year Price Growth", f"{filtered_df['PΔ1Y'].mean():.1f}%")
            st.metric("Average ROI", f"{filtered_df['ROI Avg'].mean():.1f}%")
            st.metric("Overall RCS", f"{filtered_df['Overall RCS'].mean():.0f}/100")
            
    with col3:
        st.subheader("Market Health")
        if not filtered_df.empty:
            st.metric("Median Days on Market", f"{filtered_df['DoM'].median():.0f} days")
            st.metric("Vacancy Rate", f"{filtered_df['Vacancy Rate'].mean():.1f}%")
            st.metric("Clearance Rate", f"{filtered_df['Clearance Rate'].mean():.1f}%")
    
    # Display filtered data
    st.subheader("Property Listings")
    if filtered_df.empty:
        st.warning("No properties match the selected filters.")
    else:
        st.write(f"Showing {len(filtered_df)} properties")
        
        # Select important columns to display by default
        display_columns = [
            "Area", "Property Type", "Bedrooms_Display", "Price", "Rent", "Yield",
            "ROI Low", "ROI High", "PΔ1Y", "Overall RCS", "DoM", 
            "Vacancy Rate", "UH Ratio", "BA Ratio"
        ]
        
        # Add custom score column if calculated
        if 'Custom Score' in filtered_df.columns:
            display_columns.append("Custom Score")
            
        # Add Score Percentile as the last column
        display_columns.append("Score Percentile")
        
        # Checkbox to show more metrics
        show_more_metrics = st.checkbox("Show additional metrics")
        if show_more_metrics:
            # Add more metrics to display when checkbox is checked
            additional_display_columns = [
                "PΔ5Y", "PΔ10Y", "RΔ1Y", "RΔ5Y", 
                "CG Low", "CG High", "Growth Rate", 
                "Volatility Index", "Affordability Index", 
                "Hold Period", "School Rank", "IRSAD"
            ]
            display_columns.extend(additional_display_columns)
        
        # Create expandable section with all available metrics
        with st.expander("All Available Metrics"):
            # Create a section to explain each metric
            st.write("### Metric Explanations")
            
            # Group metrics by category for better organization
            metric_categories = {
                "Price Growth": ["PΔ1M", "PΔ1Q", "PΔ6M", "PΔ1Y", "PΔ3Y", "PΔ5Y", "PΔ10Y"],
                "Rent Growth": ["RΔ1M", "RΔ1Q", "RΔ6M", "RΔ1Y", "RΔ3Y", "RΔ5Y", "RΔ10Y"],
                "Investment Returns": ["ROI Low", "ROI High", "ROI Avg", "Yield", "Cashflow"],
                "Market Health": ["DoM", "Vacancy Rate", "Clearance Rate", "Volatility Index", "SoM%", "Inventory"],
                "Growth Potential": ["Capital Growth", "CG Low", "CG High", "Growth Rate", "GRC Index"],
                "Risk Metrics": ["Flood", "Bushfire", "Lower Risk"],
                "Market Structure": ["UH Ratio", "BA Ratio", "Hold Period"],
                "Socioeconomic": ["IRSAD", "Affordability Index", "School Rank", "Infra. Spend"],
                "Other Indexes": ["EDI", "MADI", "Overall RCS"]
            }
            
            # Display metrics by category
            for category, metrics in metric_categories.items():
                st.write(f"#### {category}")
                for metric in metrics:
                    if metric in metric_options:
                        label = metric_options[metric]['label']
                        is_higher_better = "Higher is better" if metric_options[metric].get('is_higher_better', True) else "Lower is better"
                        st.write(f"**{metric}**: {label} ({is_higher_better})")
        
        # Rename Bedrooms_Display back to Bedrooms for display
        display_df = filtered_df[display_columns].copy()
        display_df = display_df.rename(columns={'Bedrooms_Display': 'Bedrooms'})
        
        # Format percentage columns in the display dataframe
        percentage_display_cols = ['Yield', 'PΔ1Y', 'Vacancy Rate', 'ROI Low', 'ROI High', 
                                  'Score Percentile']
        
        # Add additional percentage columns if they're displayed
        all_percentage_columns = ['PΔ1M', 'PΔ1Q', 'PΔ6M', 'PΔ3Y', 'PΔ5Y', 'PΔ10Y',
                                'RΔ1M', 'RΔ1Q', 'RΔ6M', 'RΔ1Y', 'RΔ3Y', 'RΔ5Y', 'RΔ10Y']
        
        for col in all_percentage_columns:
            if col in display_df.columns:
                percentage_display_cols.append(col)
        
        for col in percentage_display_cols:
            if col in display_df.columns:
                display_df[col] = display_df[col].apply(lambda x: f"{x:.1f}%")
        
        # Format score columns
        score_display_cols = ['Overall RCS', 'Capital Growth', 'CG Low', 'CG High', 
                             'School Rank', 'EDI', 'MADI', 'IRSAD']
        if 'Custom Score' in display_df.columns:
            score_display_cols.append('Custom Score')
            
        for col in score_display_cols:
            if col in display_df.columns:
                display_df[col] = display_df[col].apply(lambda x: f"{x:.0f}/100")
        
        # Format price columns
        display_df['Price'] = display_df['Price'].apply(lambda x: f"${x:,.0f}")
        display_df['Rent'] = display_df['Rent'].apply(lambda x: f"${x:,.0f}")
        
        st.dataframe(display_df, use_container_width=True)
        
        # Show detailed analysis for selected area
        if area_search:
            st.subheader(f"Detailed Analysis - {area_search}")
            col4, col5, col6 = st.columns(3)
            
            with col4:
                st.write("Investment Metrics")
                metrics_df = pd.DataFrame({
                    'Metric': ['ROI Range', 'Capital Growth Potential', 'Cashflow Rating'],
                    'Value': [
                        f"{filtered_df['ROI Low'].mean():.1f}% - {filtered_df['ROI High'].mean():.1f}%",
                        f"{filtered_df['Capital Growth'].mean():.0f}/100",
                        f"{filtered_df['Cashflow'].mean():.0f}/100"
                    ]
                })
                st.table(metrics_df)
            
            with col5:
                st.write("Risk Metrics")
                risk_df = pd.DataFrame({
                    'Metric': ['Flood Risk', 'Bushfire Risk', 'Lower Risk Score'],
                    'Value': [
                        f"{filtered_df['Flood'].mean():.0f}/100",
                        f"{filtered_df['Bushfire'].mean():.0f}/100",
                        f"{filtered_df['Lower Risk'].mean():.0f}/100"
                    ]
                })
                st.table(risk_df)
            
            with col6:
                st.write("Market Structure Metrics")
                market_df = pd.DataFrame({
                    'Metric': ['Units to Houses Ratio', 'Building Approvals Ratio'],
                    'Value': [
                        f"{filtered_df['UH Ratio'].mean():.2f}",
                        f"{filtered_df['BA Ratio'].mean():.4f}"
                    ]
                })
                st.table(market_df)
            
            # Add custom score explanation if calculated
            if 'Custom Score' in filtered_df.columns:
                st.subheader("Score Explanation")
                st.write("Your property score is calculated using the following weights:")
                
                custom_weights_df = pd.DataFrame({
                    'Metric': [metric_options[m]['label'] for m in selected_metrics],
                    'Weight': [f"{weights[m]*100:.1f}%" for m in selected_metrics]
                })
                st.table(custom_weights_df)
                
                avg_custom_score = filtered_df['Custom Score'].mean()
                avg_custom_percentile = filtered_df['Score Percentile'].mean()
                
                st.write(f"Average Score for selected properties: {avg_custom_score:.0f}/100")
                st.write(f"Average Score Percentile: {avg_custom_percentile:.1f}%")
                st.write("Properties with higher percentiles better match your criteria.")
                st.info("Note: Scores and percentiles are calculated based on the entire dataset before filtering, so they remain consistent regardless of filters applied.")
            else:
                # Show explanation for default score
                if not filtered_df.empty:
                    st.subheader("Score Explanation")
                    st.write("Using default scoring: 50% Return on Investment + 50% Overall RCS")
                    avg_percentile = filtered_df['Score Percentile'].mean()
                    st.write(f"Average Score Percentile: {avg_percentile:.1f}%")
                    st.write("To customize the scoring, select metrics in the sidebar.")
                    st.info("Note: Scores and percentiles are calculated based on the entire dataset before filtering, so they remain consistent regardless of filters applied.")

except FileNotFoundError:
    st.error("Error: 'Sydney property masterdata.xlsx' file not found. Please ensure the file is in the same directory as this script.")
except Exception as e:
    st.error(f"An error occurred: {str(e)}")
    st.error(f"Full error details:\n{traceback.format_exc()}")
