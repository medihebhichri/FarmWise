import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set page title and layout
st.set_page_config(page_title="Agricultural Data Dashboard", layout="wide")

# Title of the app
st.title("Agricultural Data Exploration")

# Sidebar for uploading dataset
st.sidebar.header("Upload Your Data")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type=["csv"])

if uploaded_file is not None:
    # Load the data into a pandas dataframe
    df = pd.read_csv(uploaded_file)
    
    # Dataset Preview
    st.subheader("Dataset Preview")
    st.write(df.head())  # Show the first few rows of the dataset
    
    # Dataset Information
    st.subheader("Dataset Information")
    st.write(df.info())

    # Select Columns for Analysis
    st.sidebar.subheader("Select Analysis Columns")
    column = st.sidebar.selectbox("Select a column for detailed analysis", df.columns)
    
    # Summary Stats for the selected column
    st.subheader(f"Summary Statistics for {column}")
    st.write(df[column].describe())

    # Time Series Analysis: For datasets with 'Year' and price/quantity
    if 'Year' in df.columns and column:
        st.subheader(f"Time Series Plot of {column} Over Time")
        fig, ax = plt.subplots(figsize=(10, 6))
        df.groupby('Year')[column].mean().plot(kind='line', ax=ax, color='b')
        ax.set_title(f"{column} Trend Over Time")
        ax.set_xlabel("Year")
        ax.set_ylabel(f"{column} Value")
        st.pyplot(fig)
    
    # Regional/State Comparison (if applicable)
    if 'State' in df.columns:
        st.sidebar.subheader("Regional Comparison")
        selected_state = st.sidebar.selectbox("Select a State", df['State'].unique())
        state_data = df[df['State'] == selected_state]
        st.subheader(f"Data for {selected_state}")
        st.write(state_data)

        # Plot comparison of attributes for selected state
        st.subheader(f"Comparison of {column} Across States")
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.boxplot(x="State", y=column, data=df, ax=ax)
        ax.set_title(f"{column} Comparison Across States")
        st.pyplot(fig)
    
    # Correlation Matrix for relevant columns
    st.subheader("Correlation Matrix")
    corr_matrix = df.corr()
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    # Interactive Bar Chart: Comparison of agricultural production by state or category
    st.sidebar.subheader("Agricultural Production Comparison")
    production_column = st.sidebar.selectbox("Select Column for Comparison", df.columns)
    bar_data = df.groupby('State')[production_column].mean().sort_values(ascending=False)
    
    st.subheader(f"Bar Chart of {production_column} by State")
    fig, ax = plt.subplots(figsize=(12, 6))
    bar_data.plot(kind='bar', ax=ax, color='g')
    ax.set_title(f"Average {production_column} by State")
    ax.set_ylabel(f"{production_column} Value")
    ax.set_xlabel("State")
    st.pyplot(fig)

    # Footer with additional information
    st.markdown("""
        ---  
        Created with ❤️ by [Your Name]  
        Agricultural data exploration with Streamlit.
    """)

