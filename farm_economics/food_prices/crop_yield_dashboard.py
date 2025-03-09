# Import necessary libraries
import dash
from dash import dcc, html
import plotly.express as px
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import plotly.graph_objects as go

# Sample DataFrame (assuming df_crop_yield is your final merged DataFrame)
# Replace this with your actual df_crop_yield DataFrame
df_crop_yield = pd.read_csv('crop_yield.csv')


# Create a Dash app
app = dash.Dash(__name__)

# Function to perform ANOVA analysis
def perform_anova(df, commodities):
    """
    Perform ANOVA analysis on the price spread for each commodity.
    Return a dictionary with p-values for each commodity.
    """
    anova_results = {}
    
    # Compare Price Spread across all commodities at once
    commodity_groups = [df[df[commodity] == 1]['Price_Spread'] for commodity in commodities]
    
    # Perform ANOVA for all commodities together
    f_value, p_value = stats.f_oneway(*commodity_groups)
    
    return p_value  # Return p-value for the comparison across all commodities

# Perform ANOVA analysis
commodities = ['bread', 'flour', 'oil', 'sugar']
anova_p_value = perform_anova(df_crop_yield, commodities)

# Create a Dash layout
app.layout = html.Div([
    html.H1("Crop Yield Price Spread Dashboard", style={'text-align': 'center'}),
    
    # Dropdown for selecting commodity
    html.Div([
        html.Label("Select Commodity for Visualization:"),
        dcc.Dropdown(
            id='commodity-dropdown',
            options=[
                {'label': 'Flour', 'value': 'flour'},
                {'label': 'Oil', 'value': 'oil'},
                {'label': 'Sugar', 'value': 'sugar'},
                {'label': 'Bread', 'value': 'bread'}
            ],
            value='flour',  # default value
            style={'width': '50%'}
        )
    ], style={'padding': '20px'}),
    
    # Graphs for various visualizations
    html.Div([
        dcc.Graph(id='price-spread-graph'),
        dcc.Graph(id='farm-share-graph'),
        dcc.Graph(id='retail-vs-farm-graph'),
        dcc.Graph(id='correlation-graph'),
        dcc.Graph(id='price-spread-distribution-graph'),
        dcc.Graph(id='trend-price-spread-graph'),
        dcc.Graph(id='anova-graph')  # ANOVA Result Graph
    ])
])

# Callback for updating the graphs based on selected commodity
@app.callback(
    [dash.dependencies.Output('price-spread-graph', 'figure'),
     dash.dependencies.Output('farm-share-graph', 'figure'),
     dash.dependencies.Output('retail-vs-farm-graph', 'figure'),
     dash.dependencies.Output('correlation-graph', 'figure'),
     dash.dependencies.Output('price-spread-distribution-graph', 'figure'),
     dash.dependencies.Output('trend-price-spread-graph', 'figure'),
     dash.dependencies.Output('anova-graph', 'figure')],  # Adding ANOVA graph
    [dash.dependencies.Input('commodity-dropdown', 'value')]
)
def update_graphs(commodity):
    # Filter data based on selected commodity
    df_filtered = df_crop_yield[df_crop_yield[commodity] == 1]

    # Price Spread by Commodity over Time
    price_spread_figure = px.line(
        df_filtered, x='Year', y='Price_Spread', title=f'Price Spread of {commodity.capitalize()} Over Time'
    )

    # Farm Share (%) for Each Commodity Over Time
    farm_share_figure = px.line(
        df_filtered, x='Year', y='Price_Spread', title=f'Farm Share of {commodity.capitalize()} Over Time'
    )

    # Retail Value vs. Farm Value Comparison
    retail_vs_farm_figure = px.line(
        df_filtered, x='Year', y=['Value_retail', 'Value_farm'], title=f'Retail and Farm Values of {commodity.capitalize()} Over Time'
    )

    # Correlation Between Retail Value and Farm Value (Scatter Plot)
    correlation_figure = px.scatter(
        df_filtered, x='Value_farm', y='Value_retail', title=f'Retail vs. Farm Value Correlation for {commodity.capitalize()}'
    )

    # Price Spread Distribution (Histogram)
    price_spread_distribution_figure = px.histogram(
        df_filtered, x='Price_Spread', title=f'Price Spread Distribution of {commodity.capitalize()}'
    )

    # Trend of Price Spread over Time
    trend_price_spread_figure = px.line(
        df_filtered, x='Year', y='Price_Spread', title=f'Trend of {commodity.capitalize()} Price Spread Over Time'
    )

    # ANOVA Results Visualization
    # Since we're now comparing across all commodities, we can plot this comparison
    anova_figure = go.Figure()
    anova_figure.add_trace(go.Bar(
        x=commodities,
        y=[anova_p_value] * len(commodities),  # All commodities will have the same p-value
        name='ANOVA p-values',
        marker_color='blue'
    ))

    anova_figure.update_layout(
        title="ANOVA p-value for Price Spread Across Commodities",
        xaxis_title="Commodities",
        yaxis_title="p-value",
        showlegend=False
    )

    return price_spread_figure, farm_share_figure, retail_vs_farm_figure, correlation_figure, price_spread_distribution_figure, trend_price_spread_figure, anova_figure

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
