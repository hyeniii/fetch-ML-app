import streamlit as st
import pandas as pd
import plotly.graph_objects as go

# Load the data
df = pd.read_csv('assets/predictions_2022.csv', parse_dates=True, index_col=0)

def predict_receipts(month: int):
    return df.iloc[month-1]['Predicted_Receipts']

st.title('Receipts Prediction App')

month = st.slider("Choose a month for 2022", 1, 12)

if st.button('Predict'):
    prediction = predict_receipts(month)
    st.write(f"Predicted Receipts for Month {month} of 2022: {prediction:.2f}")

st.subheader("Monthly Receipts Predictions for 2022")
# 
# Create the plot
fig = go.Figure()
fig.add_trace(go.Scatter(x=df.index, y=df['Predicted_Receipts'], mode='lines+markers', name='Predicted Receipts'))
fig.add_trace(go.Scatter(x=[df.index[month-1]], y=[df['Predicted_Receipts'].iloc[month-1]], mode='markers', marker=dict(size=15, color='red'), name=f'Month {month}'))

# Display the interactive plot in Streamlit
st.plotly_chart(fig)

st.image("assets/receipts_plot.png", caption='Monthly Receipts Predictions for 2022', use_column_width=True)