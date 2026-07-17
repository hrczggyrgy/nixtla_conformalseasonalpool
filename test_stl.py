import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
import plotly.graph_objects as go
from plotly.subplots import make_subplots

df = pd.read_csv('aggregated_sales.csv')
series_data = df[df['store'] == 1]['sales'].dropna()
print('Series length:', len(series_data))

decomp = seasonal_decompose(
    series_data, model='additive', period=7, extrapolate_trend='freq'
)

from plotly.subplots import make_subplots
import plotly.graph_objects as go

fig_stl = make_subplots(
    rows=4, cols=1,
    subplot_titles=('Observed', 'Trend', 'Seasonal', 'Residual'),
    shared_xaxes=True,
    vertical_spacing=0.05,
)

components = [
    ('Observed', decomp.observed, '#1f77b4'),
    ('Trend', decomp.trend, '#ff7f0e'),
    ('Seasonal', decomp.seasonal, '#2ca02c'),
    ('Residual', decomp.resid, '#d62728'),
]

for i, (name, data, color) in enumerate(components, 1):
    x_vals = decomp.observed.index if hasattr(decomp.observed.index, '__iter__') else range(len(data))
    y_vals = data.values if hasattr(data, 'values') else data
    print('Component {0}: x_vals type={1}, len={2}, y_vals type={3}, len={4}'.format(name, type(x_vals), len(x_vals), type(y_vals), len(y_vals)))

print('STL decomposition test successful!')