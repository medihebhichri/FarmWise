import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from prophet import Prophet

# Load the dataset
@st.cache
def load_data():
    return pd.read_csv('table01a.csv')

df = load_data()

# Clean up the dataset
df['Year'] = df['Year'].astype(int)
df_filtered = df[df['Value'] > 0]  # Filter out rows with zero or negative values

# App title and description
st.title("Agricultural Price Indices & Quantities Dashboard")
st.markdown("""
    This interactive dashboard explores agricultural price shifts and production quantities from 1948 to 2021. 
    It offers insights into the market's response to economic changes over time, including inflation and production trends.
""")

# Sidebar with filters
st.sidebar.header("Filters")
selected_year = st.sidebar.slider('Select Year Range', min_value=int(df['Year'].min()), 
                                  max_value=int(df['Year'].max()), value=(int(df['Year'].min()), int(df['Year'].max())))

selected_attribute = st.sidebar.multiselect('Select Attributes', df['Attribute'].unique(), default=df['Attribute'].unique())

# Filter data based on user selection
df_filtered = df[(df['Year'] >= selected_year[0]) & (df['Year'] <= selected_year[1])]
df_filtered = df_filtered[df_filtered['Attribute'].isin(selected_attribute)]

# Time-series Plot (Line chart)
st.subheader("Price and Quantity Trends Over Time")
fig = px.line(df_filtered, x='Year', y='Value', color='Attribute', title="Agricultural Price and Quantity Trends")
fig.update_traces(mode='lines+markers')  # Add markers to the lines
st.plotly_chart(fig)

# Correlation Analysis: Scatter Plot (Price vs Quantity)
st.subheader("Price vs Quantity Analysis")
fig = plt.figure(figsize=(10, 6))
sns.scatterplot(data=df_filtered, x='Value', y='Value')  # Adjust based on available columns
plt.title("Price vs Quantity Scatter Plot")
plt.xlabel("Price/Quantity")
plt.ylabel("Price/Quantity")
st.pyplot(fig)

# Correlation Heatmap
st.subheader("Correlation Heatmap Between Attributes")
correlation_matrix = df_filtered.pivot_table(index='Year', columns='Attribute', values='Value').corr()
fig, ax = plt.subplots(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', ax=ax)
st.pyplot(fig)

# Forecasting with Prophet (Optional)
st.subheader("Price Forecasting with Prophet")
df_prophet = df_filtered[['Year', 'Value']].rename(columns={'Year': 'ds', 'Value': 'y'})
model = Prophet(yearly_seasonality=True)
model.fit(df_prophet)

future = model.make_future_dataframe(periods=12, freq='Y')
forecast = model.predict(future)

fig = model.plot(forecast)
st.pyplot(fig)

# Custom Styling
st.markdown("""
    <style>
        body {
            background-color: #f0f0f5;
            font-family: 'Arial', sans-serif;
        }
        h1 {
            color: #4CAF50;
        }
        .stText {
            font-size: 20px;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            font-size: 16px;
        }
    </style>
""", unsafe_allow_html=True)

# Add interactivity for user to see more insights
st.sidebar.subheader("Interactive Elements")
show_trend = st.sidebar.checkbox("Show Trendline in Time-Series", value=True)

if show_trend:
    st.subheader("Time-Series with Trendline")
    fig = px.line(df_filtered, x='Year', y='Value', color='Attribute', title="Agricultural Price and Quantity Trends with Trendline")
    fig.update_traces(mode='lines+markers+text')
    st.plotly_chart(fig)

# Display Insights
st.sidebar.subheader("Insights")
if len(df_filtered) > 0:
    min_value = df_filtered['Value'].min()
    max_value = df_filtered['Value'].max()
    mean_value = df_filtered['Value'].mean()

    st.sidebar.write(f"Minimum Value: {min_value}")
    st.sidebar.write(f"Maximum Value: {max_value}")
    st.sidebar.write(f"Average Value: {mean_value:.2f}")

# Footer
st.markdown("""
    <footer>
        <p>Created by: Smart Data Science</p>
    </footer>
""", unsafe_allow_html=True)
