from datetime import datetime, date, timedelta
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objs as plotly_go
from sklearn.preprocessing import MinMaxScaler
import streamlit as st
from tensorflow.keras.models import load_model
import seaborn as sns
import plotly.graph_objs as go
import plotly.express as px
from PIL import Image
im = Image.open("favicon.ico")
st.set_page_config(
        page_title="Stockanalyzee",
        page_icon=im,
    )
primary_color = "#EF5A6F"
secondary_color = "#FFF1DB"
tertiary_color = "#D4BDAC"
text_color = "#536493"
css = f"""
<style>
    .reportview-container .main .block-container{{
        max-width: 90%;
        padding-top: 5rem;
        padding-right: 1rem;
        padding-left: 1rem;
        padding-bottom: 5rem;
    }}
    .reportview-container .main {{
        color: {text_color};
        background-color: {secondary_color};
    }}
    .reportview-container .main .block-container {{
        background-color: {tertiary_color};
        border-radius: 10px;
        padding: 2rem;
    }}
    .reportview-container .main .block-container .element-container {{
        background-color: {tertiary_color};
        border-radius: 10px;
        padding: 1rem;
    }}
    .reportview-container .main .block-container .element-container:hover {{
        background-color: {primary_color};
        color: {secondary_color};
    }}
    .reportview-container .main .block-container .element-container .stButton > button {{
        background-color: {primary_color};
        color: {secondary_color};
    }}
    .reportview-container .main .block-container .element-container .stButton > button:hover {{
        background-color: {text_color};
        color: {secondary_color};
    }}
</style>
"""


# Inject custom CSS
st.markdown(css, unsafe_allow_html=True)


st.title("Stock Trend Predictor")


if 'model_ready' not in st.session_state:
    st.session_state.model_ready = False

if 'predictions' not in st.session_state:
    st.session_state.predictions = None

if 'y_test' not in st.session_state:
    st.session_state.y_test = None

yesterday = datetime.now() - timedelta(1)
lastyear= datetime.now().year;
default_start_date = date(1970, 1, 31)
user_input = st.text_input('Enter Stock Ticker', 'AAPL')
stock = yf.Ticker(user_input)
info = stock.info
st.subheader(info['longName'])
st.markdown('**Sector**:' + info['sector'])
st.markdown('**Industry**: ' + info['industry'])
st.markdown('**Phone**: ' + info['phone'])
st.markdown(
    '**Address**: ' + info['address1'] + ', ' + info['city'] + ', ' + info['zip'] + ', ' + info['country'])
st.markdown('**Website**: ' + info['website'])
with st.expander('See detailed business summary'):
    st.write(info['longBusinessSummary'])

data = yf.download(user_input,start='2000-01-01',end=yesterday)
st.title(f"{user_input}")
# Check if data is empty
if data.empty:
    st.write("No data available for the specified time range.")
else:
    # Extract year from the date
    data['Year'] = data.index.year

    # Create a Streamlit app

    # Create the interactive box plot using Plotly
    fig = px.box(data, x='Year', y='Close',title=f"Year-wise Box Plot of {user_input} Closing Prices",
                 labels={'Year': 'Year', 'Close': 'Closing Price'},
                 color='Year',  # Adds color differentiation for each year
                 boxmode='overlay')  # Overlay the boxes for better comparison

    # Customize the layout
    fig.update_layout(
        xaxis_title='Year',
        yaxis_title='Closing Price',
        xaxis_tickangle=-45,  # Rotate x-axis labels for better readability
        title_font_size=16,
        xaxis_title_font_size=14,
        yaxis_title_font_size=14,
        showlegend=True,
        width = 2000,  # Set the width of the plot
        height = 550
    )
    fig.update_xaxes(
        tickvals=list(range(data['Year'].min(), data['Year'].max() + 1, 2)),
        ticktext=[str(year) for year in range(data['Year'].min(), data['Year'].max() + 1, 2)]
    )

    # Display the interactive plot in Streamlit
    st.plotly_chart(fig)

st.markdown(f"Here the user can select the start day anywhere from 1970 to 31-12-{lastyear}. To increase the accuracy of the model.")
d_start = st.date_input("Enter the Start Date", min_value=date(1970, 1, 1), max_value=date(lastyear, 12, 31), value=default_start_date)
d_end = st.date_input("Enter the End Date", max_value=yesterday, value=yesterday)

# Download data
df = yf.download(user_input, start=d_start, end=d_end)

# Display data summary
st.subheader(f'Data from {d_start} to {d_end}')
st.write(df.describe())

# Closing Price vs Time Chart
st.subheader("Closing Price vs Time Chart")
fig = plotly_go.Figure()
fig.add_trace(plotly_go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Closing Price', line=dict(color='blue')))
fig.update_layout(title='Closing Price vs Time', xaxis_title='Time', yaxis_title='Closing Price')
st.plotly_chart(fig)

# Closing Price vs Time Chart with 100 days moving average
st.subheader("Closing Price vs Time Chart with 100 Days Moving Average")
df['MA100'] = df['Close'].rolling(100).mean()
fig = plotly_go.Figure()
fig.add_trace(plotly_go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Closing Price', line=dict(color='blue')))
fig.add_trace(plotly_go.Scatter(x=df.index, y=df['MA100'], mode='lines', name='100 Day Moving Average', line=dict(color='orange')))
fig.update_layout(title='Closing Price vs Time with 100 Days Moving Average', xaxis_title='Time', yaxis_title='Price')
st.plotly_chart(fig)

# Closing Price vs Time Chart with 200 days moving average
st.subheader("Closing Price vs Time Chart with 200 Days Moving Average")
df['MA200'] = df['Close'].rolling(200).mean()
fig = plotly_go.Figure()
fig.add_trace(plotly_go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Closing Price', line=dict(color='blue')))
fig.add_trace(plotly_go.Scatter(x=df.index, y=df['MA100'], mode='lines', name='100 Day Moving Average', line=dict(color='orange')))
fig.add_trace(plotly_go.Scatter(x=df.index, y=df['MA200'], mode='lines', name='200 Day Moving Average', line=dict(color='red')))
fig.update_layout(title='Closing Price vs Time with 100 and 200 Days Moving Averages', xaxis_title='Time', yaxis_title='Price')
st.plotly_chart(fig)

# Prepare training and testing data
train_size = int(len(df) * 0.70)
train = df[:train_size]
test = df[train_size:]
scaler = MinMaxScaler(feature_range=(0, 1))

train_close = train.iloc[:, 4:5].values
test_close = test.iloc[:, 4:5].values

data_training_array = scaler.fit_transform(train_close)

model = load_model('keras_model.h5')
past_100_days = pd.DataFrame(train_close[-100:])
test_df = pd.DataFrame(test_close)
final_df = pd.concat([past_100_days, test_df], ignore_index=True)
input_data = scaler.transform(final_df)

x_test, y_test = [], []
for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i - 100:i])
    y_test.append(input_data[i, 0])

x_test, y_test = np.array(x_test), np.array(y_test)

# Display a spinner and process the model prediction
with st.spinner('Model is training. Please wait...'):
    y_pred = model.predict(x_test)

    # Rescale the predictions and test values
    scaler_factor = 1 / scaler.scale_[0]
    y_pred = y_pred * scaler_factor
    y_test = y_test * scaler_factor

    # Update session state
    st.session_state.model_ready = True
    st.session_state.predictions = y_pred
    st.session_state.y_test = y_test

if st.session_state.model_ready:
    st.subheader("Predictions vs Original")

    # Create interactive plot with Plotly
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=list(range(len(st.session_state.y_test))),
        y=st.session_state.y_test.flatten(),  # Flatten to convert 2D array to 1D
        mode='lines',
        name='Original Price'
    ))

    fig.add_trace(go.Scatter(
        x=list(range(len(st.session_state.predictions))),
        y=st.session_state.predictions.flatten(),  # Flatten to convert 2D array to 1D
        mode='lines',
        name='Predicted Price'
    ))

    fig.update_layout(
        title='Predictions vs Original',
        xaxis_title='Time',
        yaxis_title='Price',
        legend_title='Legend'
    )

    st.plotly_chart(fig)