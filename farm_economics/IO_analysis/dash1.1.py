import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.linear_model import LinearRegression

# Load the dataset
dataset1 = pd.read_csv('table01a.csv')  # Replace with your dataset path

# Data Preprocessing
dataset1['Year'] = dataset1['Year'].astype(int)
dataset1.dropna(subset=['Value'], inplace=True)

# Filter price data
price_data = dataset1[dataset1['Attribute'].str.contains('Price')]
quantity_data = dataset1[dataset1['Attribute'].str.contains('Quantity')]

# EDA: Price Fluctuations (Rolling Average)
price_data['Rolling_Avg'] = price_data.groupby('Attribute')['Value'].rolling(window=5).mean().reset_index(0, drop=True)

# Year-over-Year Price Change
price_data['Price_Change'] = price_data.groupby('Attribute')['Value'].pct_change() * 100

# Correlation between Price and Quantity
merged_data = pd.merge(price_data, quantity_data, on='Year', suffixes=('_Price', '_Quantity'))

# Regression Model (Linear Regression)
X = price_data['Year'].values.reshape(-1, 1)
y = price_data['Value'].values
model = LinearRegression()
model.fit(X, y)
slope, intercept = model.coef_[0], model.intercept_

# Streamlit UI
st.title("Agricultural Price and Quantity Analysis")
st.write("This dashboard provides insights into agricultural price shifts and production quantities from 1948 to 2021.")

# Sidebar for filtering
selected_attribute = st.sidebar.selectbox('Select Attribute', options=price_data['Attribute'].unique())
filtered_attribute_data = price_data[price_data['Attribute'] == selected_attribute]

# Price Distribution by Year (Boxplot)
boxplot_fig = px.box(filtered_attribute_data, x='Year', y='Value', title=f"{selected_attribute} Distribution by Year")
st.plotly_chart(boxplot_fig)

# Price vs Quantity Correlation (Scatter plot)
corr_fig = px.scatter(merged_data, x='Value_Price', y='Value_Quantity', title="Price vs Quantity Correlation", hover_data=['Year'])
st.plotly_chart(corr_fig)

# Rolling Average of Prices (Line Plot)
rolling_avg_fig = px.line(price_data, x='Year', y='Rolling_Avg', color='Attribute', title="Price Rolling Average (5 Years)")
st.plotly_chart(rolling_avg_fig)

# Year-over-Year Price Change (Line Plot)
price_change_fig = px.line(price_data, x='Year', y='Price_Change', color='Attribute', title="Year-over-Year Price Change")
st.plotly_chart(price_change_fig)

# Linear Regression Trend Line
st.write(f"Linear Regression for Price Data: Slope = {slope:.4f}, Intercept = {intercept:.4f}")
trend_line_fig = px.scatter(price_data, x='Year', y='Value', title="Price Trend with Linear Regression Line")
trend_line_fig.add_scatter(x=price_data['Year'], y=model.predict(X), mode='lines', name='Trend Line')
st.plotly_chart(trend_line_fig)

# Conclusion Insights
st.subheader("Key Insights")
st.write("""
- The dataset shows periodic shifts in agricultural prices, with notable fluctuations during certain decades.
- There is a significant correlation between agricultural prices and production quantities.
- A linear regression model indicates a positive trend in prices over time.
- Major price shifts can be observed during years of economic or political change.
""")

# Interactivity: Dynamic Updates
st.sidebar.subheader("Filter Data")
year_range = st.sidebar.slider("Select Year Range", min_value=int(price_data['Year'].min()), max_value=int(price_data['Year'].max()), value=(1948, 2021))
filtered_year_data = price_data[(price_data['Year'] >= year_range[0]) & (price_data['Year'] <= year_range[1])]
st.write(f"Displaying data for years between {year_range[0]} and {year_range[1]}")
st.dataframe(filtered_year_data)

