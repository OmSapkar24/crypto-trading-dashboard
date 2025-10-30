import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objs as go
import pandas as pd
import requests
from datetime import datetime, timedelta
import numpy as np

# Initialize the Dash app
app = dash.Dash(__name__)
app.title = "Crypto Trading Dashboard"

# Color scheme
COLORS = {
    'background': '#0e1117',
    'text': '#ffffff',
    'primary': '#00d4ff',
    'secondary': '#ff6b6b',
    'success': '#4caf50',
    'warning': '#ff9800'
}

# Sample cryptocurrency list
CRYPTO_LIST = ['BTC', 'ETH', 'BNB', 'ADA', 'SOL', 'XRP', 'DOT', 'DOGE']

# App layout
app.layout = html.Div(style={'backgroundColor': COLORS['background'], 'minHeight': '100vh', 'padding': '20px'}, children=[
    html.Div([
        html.H1('Crypto Trading Dashboard', 
                style={'textAlign': 'center', 'color': COLORS['primary'], 'marginBottom': '30px'}),
        
        # Control Panel
        html.Div([
            html.Div([
                html.Label('Select Cryptocurrency:', style={'color': COLORS['text'], 'fontWeight': 'bold'}),
                dcc.Dropdown(
                    id='crypto-dropdown',
                    options=[{'label': crypto, 'value': crypto} for crypto in CRYPTO_LIST],
                    value='BTC',
                    style={'backgroundColor': '#1e2130', 'color': '#000000'}
                )
            ], style={'width': '30%', 'display': 'inline-block', 'marginRight': '3%'}),
            
            html.Div([
                html.Label('Time Range:', style={'color': COLORS['text'], 'fontWeight': 'bold'}),
                dcc.Dropdown(
                    id='timerange-dropdown',
                    options=[
                        {'label': '1 Day', 'value': '1'},
                        {'label': '7 Days', 'value': '7'},
                        {'label': '30 Days', 'value': '30'},
                        {'label': '90 Days', 'value': '90'}
                    ],
                    value='7',
                    style={'backgroundColor': '#1e2130', 'color': '#000000'}
                )
            ], style={'width': '30%', 'display': 'inline-block', 'marginRight': '3%'}),
            
            html.Div([
                html.Button('Refresh Data', id='refresh-button', n_clicks=0,
                           style={'backgroundColor': COLORS['primary'], 'color': COLORS['background'],
                                  'border': 'none', 'padding': '10px 20px', 'cursor': 'pointer',
                                  'borderRadius': '5px', 'fontWeight': 'bold', 'marginTop': '25px'})
            ], style={'width': '30%', 'display': 'inline-block'})
        ], style={'marginBottom': '30px'}),
        
        # Stats Cards
        html.Div(id='stats-cards', style={'marginBottom': '30px'}),
        
        # Price Chart
        html.Div([
            dcc.Graph(id='price-chart', style={'height': '400px'})
        ], style={'marginBottom': '30px'}),
        
        # Volume Chart
        html.Div([
            dcc.Graph(id='volume-chart', style={'height': '300px'})
        ], style={'marginBottom': '30px'}),
        
        # Technical Indicators
        html.Div([
            html.H3('Technical Indicators', style={'color': COLORS['text'], 'marginBottom': '20px'}),
            dcc.Graph(id='indicators-chart', style={'height': '300px'})
        ]),
        
        # Auto-refresh interval (every 60 seconds)
        dcc.Interval(
            id='interval-component',
            interval=60*1000,  # in milliseconds
            n_intervals=0
        )
    ])
])

# Generate sample data (replace with real API calls)
def generate_sample_data(crypto, days):
    """Generate sample cryptocurrency data"""
    dates = pd.date_range(end=datetime.now(), periods=int(days)*24, freq='H')
    
    # Generate realistic-looking price data
    base_price = {'BTC': 45000, 'ETH': 3000, 'BNB': 400, 'ADA': 1.5, 
                  'SOL': 100, 'XRP': 0.75, 'DOT': 25, 'DOGE': 0.15}
    
    price = base_price.get(crypto, 100)
    prices = []
    current_price = price
    
    for _ in range(len(dates)):
        change = np.random.randn() * price * 0.02
        current_price += change
        prices.append(current_price)
    
    volumes = np.random.randint(1000000, 10000000, size=len(dates))
    
    df = pd.DataFrame({
        'timestamp': dates,
        'price': prices,
        'volume': volumes
    })
    
    return df

# Calculate technical indicators
def calculate_indicators(df):
    """Calculate technical indicators"""
    # Simple Moving Averages
    df['SMA_20'] = df['price'].rolling(window=20).mean()
    df['SMA_50'] = df['price'].rolling(window=50).mean()
    
    # RSI (Relative Strength Index)
    delta = df['price'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    return df

# Callback to update all charts
@app.callback(
    [Output('stats-cards', 'children'),
     Output('price-chart', 'figure'),
     Output('volume-chart', 'figure'),
     Output('indicators-chart', 'figure')],
    [Input('refresh-button', 'n_clicks'),
     Input('interval-component', 'n_intervals')],
    [State('crypto-dropdown', 'value'),
     State('timerange-dropdown', 'value')]
)
def update_dashboard(n_clicks, n_intervals, selected_crypto, time_range):
    # Generate data
    df = generate_sample_data(selected_crypto, time_range)
    df = calculate_indicators(df)
    
    # Calculate statistics
    current_price = df['price'].iloc[-1]
    price_change = df['price'].iloc[-1] - df['price'].iloc[0]
    price_change_pct = (price_change / df['price'].iloc[0]) * 100
    volume_24h = df['volume'].tail(24).sum()
    high_24h = df['price'].tail(24).max()
    low_24h = df['price'].tail(24).min()
    
    # Stats cards
    stats_cards = html.Div([
        html.Div([
            html.H4('Current Price', style={'color': COLORS['text'], 'margin': '0'}),
            html.H2(f'${current_price:,.2f}', style={'color': COLORS['primary'], 'margin': '10px 0'}),
            html.P(f'{price_change_pct:+.2f}%', 
                   style={'color': COLORS['success'] if price_change > 0 else COLORS['secondary'],
                          'fontSize': '18px', 'margin': '0'})
        ], style={'backgroundColor': '#1e2130', 'padding': '20px', 'borderRadius': '10px',
                  'width': '22%', 'display': 'inline-block', 'marginRight': '3%'}),
        
        html.Div([
            html.H4('24h High', style={'color': COLORS['text'], 'margin': '0'}),
            html.H2(f'${high_24h:,.2f}', style={'color': COLORS['success'], 'margin': '10px 0'})
        ], style={'backgroundColor': '#1e2130', 'padding': '20px', 'borderRadius': '10px',
                  'width': '22%', 'display': 'inline-block', 'marginRight': '3%'}),
        
        html.Div([
            html.H4('24h Low', style={'color': COLORS['text'], 'margin': '0'}),
            html.H2(f'${low_24h:,.2f}', style={'color': COLORS['secondary'], 'margin': '10px 0'})
        ], style={'backgroundColor': '#1e2130', 'padding': '20px', 'borderRadius': '10px',
                  'width': '22%', 'display': 'inline-block', 'marginRight': '3%'}),
        
        html.Div([
            html.H4('24h Volume', style={'color': COLORS['text'], 'margin': '0'}),
            html.H2(f'${volume_24h:,.0f}', style={'color': COLORS['warning'], 'margin': '10px 0'})
        ], style={'backgroundColor': '#1e2130', 'padding': '20px', 'borderRadius': '10px',
                  'width': '22%', 'display': 'inline-block'})
    ])
    
    # Price chart
    price_figure = {
        'data': [
            go.Candlestick(
                x=df['timestamp'],
                open=df['price'],
                high=df['price'] * 1.01,
                low=df['price'] * 0.99,
                close=df['price'],
                name='Price',
                increasing_line_color=COLORS['success'],
                decreasing_line_color=COLORS['secondary']
            ),
            go.Scatter(
                x=df['timestamp'],
                y=df['SMA_20'],
                mode='lines',
                name='SMA 20',
                line=dict(color=COLORS['primary'], width=1)
            ),
            go.Scatter(
                x=df['timestamp'],
                y=df['SMA_50'],
                mode='lines',
                name='SMA 50',
                line=dict(color=COLORS['warning'], width=1)
            )
        ],
        'layout': go.Layout(
            title=f'{selected_crypto} Price Chart',
            paper_bgcolor=COLORS['background'],
            plot_bgcolor='#1e2130',
            font={'color': COLORS['text']},
            xaxis={'gridcolor': '#2e3440'},
            yaxis={'gridcolor': '#2e3440', 'title': 'Price (USD)'},
            hovermode='x unified',
            legend={'orientation': 'h', 'y': 1.1}
        )
    }
    
    # Volume chart
    volume_figure = {
        'data': [
            go.Bar(
                x=df['timestamp'],
                y=df['volume'],
                name='Volume',
                marker={'color': COLORS['primary']}
            )
        ],
        'layout': go.Layout(
            title='Trading Volume',
            paper_bgcolor=COLORS['background'],
            plot_bgcolor='#1e2130',
            font={'color': COLORS['text']},
            xaxis={'gridcolor': '#2e3440'},
            yaxis={'gridcolor': '#2e3440', 'title': 'Volume'},
            hovermode='x unified'
        )
    }
    
    # Indicators chart (RSI)
    indicators_figure = {
        'data': [
            go.Scatter(
                x=df['timestamp'],
                y=df['RSI'],
                mode='lines',
                name='RSI',
                line=dict(color=COLORS['primary'], width=2)
            )
        ],
        'layout': go.Layout(
            title='Relative Strength Index (RSI)',
            paper_bgcolor=COLORS['background'],
            plot_bgcolor='#1e2130',
            font={'color': COLORS['text']},
            xaxis={'gridcolor': '#2e3440'},
            yaxis={'gridcolor': '#2e3440', 'title': 'RSI', 'range': [0, 100]},
            hovermode='x unified',
            shapes=[
                {'type': 'line', 'x0': df['timestamp'].min(), 'x1': df['timestamp'].max(),
                 'y0': 70, 'y1': 70, 'line': {'color': COLORS['secondary'], 'dash': 'dash'}},
                {'type': 'line', 'x0': df['timestamp'].min(), 'x1': df['timestamp'].max(),
                 'y0': 30, 'y1': 30, 'line': {'color': COLORS['success'], 'dash': 'dash'}}
            ]
        )
    }
    
    return stats_cards, price_figure, volume_figure, indicators_figure

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0', port=8050)
