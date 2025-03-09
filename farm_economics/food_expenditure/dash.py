import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# Load your data
monthly_sales = pd.read_csv('monthly_sales.csv')
monthly_sales_by_outlet = pd.read_csv('monthly_sales by outlet.csv')

# Clean your data (remove commas and convert to float where applicable)
def clean_data(df):
    for column in df.columns:
        if df[column].dtype == 'object':  # Check if the column is of object type (text)
            # Remove commas and colons, then convert to numeric (float)
            df[column] = df[column].str.replace(',', '').str.replace(':', '').astype(float, errors='ignore')
    return df

# Apply cleaning function
monthly_sales = clean_data(monthly_sales)
monthly_sales_by_outlet = clean_data(monthly_sales_by_outlet)

# Combine both dataframes on Year and Month
merged_data = pd.merge(monthly_sales, monthly_sales_by_outlet, on=['Year', 'Month'], how='outer')

# Convert 'Month' to datetime format by creating a 'Date' column
merged_data['Date'] = pd.to_datetime(merged_data['Month'] + ' ' + merged_data['Year'].astype(str))

# Streamlit title and introduction
st.title("Food Market Analysis Dashboard")
st.write("Explore the trends in monthly food sales over time.")

# Dropdown to select columns to plot (allowing multiple selections)
columns = merged_data.columns.tolist()
columns_to_plot = [col for col in columns if 'sales' in col.lower()]  # Select only columns related to sales

# Create a multiselect for users to choose multiple columns to plot
selected_columns = st.multiselect("Select columns to visualize", columns_to_plot)

# Plot the selected columns over time
if selected_columns:
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot each selected column
    for column in selected_columns:
        ax.plot(merged_data['Date'], merged_data[column], marker='.', linestyle='-', label=column)
    
    ax.set_title("Selected Columns Over Time")
    ax.set_xlabel("Date")
    ax.set_ylabel("Sales (in millions)")
    ax.legend()  # Add legend to distinguish the lines
    ax.grid(True)

    st.pyplot(fig)
else:
    st.write("Please select at least one column to plot.")
