import altair as alt
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from src.helper_functions import custom_metric_box, pollution_box
from src.predict import get_data_and_predictions, update_data_and_predictions

st.set_page_config(
    page_title="Utrecht Pollution Dashboard ",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)

alt.themes.enable("dark")

update_data_and_predictions()

week_data, predictions_O3, predictions_NO2 = get_data_and_predictions()

today = week_data.iloc[-1]
previous_day = week_data.iloc[-2]

dates_past = pd.date_range(end=pd.Timestamp.today(), periods=8).to_list()
dates_future = pd.date_range(
    start=pd.Timestamp.today() + pd.Timedelta(days=1), periods=3
).to_list()

# O3 and NO2 values for the past 7 days
o3_past_values = week_data["O3"]
no2_past_values = week_data["NO2"]
o3_future_values = pd.Series(predictions_O3[0].flatten())
no2_future_values = pd.Series(predictions_NO2[0].flatten())
o3_values = pd.concat([o3_past_values, o3_future_values], ignore_index=True)
no2_values = pd.concat([no2_past_values, no2_future_values], ignore_index=True)

dates = dates_past + dates_future
df = pd.DataFrame({"Date": dates, "O3": o3_values, "NO2": no2_values})

# App Title
st.title("Utrecht Pollution Dashboard 🌱")

col1, col2 = st.columns((1, 3))
# Create a 3-column layout
with col1:
    st.subheader("Current Weather")

    custom_metric_box(
        label="🥵 Temperature",
        value=f"{round(today['mean_temp'] * 0.1)} °C",
    )
    custom_metric_box(
        label="💧 Humidity",
        value=f"{round(today['humidity'])} %",
    )
    custom_metric_box(
        label="🪨 Pressure",
        value=f"{round(today['pressure'] * 0.1)} hPa",
    )

    custom_metric_box(
        label="🌧️ Precipitation",
        value=f"{round(today['percipitation'] * 0.1)} mm",
    )
    custom_metric_box(
        label="🌤️ Solar Radiation",
        value=f"{round(today['global_radiation'])} J/m²",
    )
    custom_metric_box(
        label="🌪️ Wind Speed",
        value=f"{round(today['wind_speed'] * 0.1, 1)} m/s",
    )

with col2:
    st.subheader("Current Pollution Levels")
    sub1, sub2 = st.columns((1, 1))

    # Ozone (O₃) Pollution Box
    with sub1:
        pollution_box(
            label="O<sub>3</sub>",
            value=f"{round(today['O3'])} µg/m³",
            delta=f"{round(int(today['O3']) - int(previous_day['O3']))} µg/m³",
            threshold=120,
        )
        with st.expander("Learn more about O3", expanded=False):
            st.markdown(
                """
                *Ozone (O<sub>3</sub>)*: A harmful gas at ground level that can irritate the respiratory system and aggravate asthma.<br>
                **Good/Bad**: "Good" means safe levels for most people, while "Bad" suggests harmful levels, especially for sensitive groups.
                """,
                unsafe_allow_html=True,
            )

    # Nitrogen Dioxide (NO₂) Pollution Box
    with sub2:
        pollution_box(
            label="NO<sub>2</sub>",
            value=f"{round(today['NO2'])} µg/m³",
            delta=f"{round(int(today['NO2']) - int(previous_day['NO2']))} µg/m³",
            threshold=40,
        )
        with st.expander("Learn more about NO2", expanded=False):
            st.markdown(
                """
                *Nitrogen Dioxide (NO<sub>2</sub>)*: A toxic gas that contributes to lung irritation and worsens asthma and other respiratory issues.<br>
                **Good/Bad**: "Good" means safe air quality, while "Bad" indicates levels that could cause respiratory problems, especially for vulnerable individuals.
                """,
                unsafe_allow_html=True,
            )

    # Create two columns for two separate graphs
    st.subheader("O3 Forecast")

    # Define the new color logic: green, orange, and red based on the threshold
    def get_simple_color_scale(values, threshold):
        """Returns green for values below the threshold, orange for values between the threshold and 2x the threshold, and red for values above 2x the threshold."""
        return [
            "#77C124"
            if v < threshold
            else "#E68B0A"
            if v < 2 * threshold
            else "#E63946"
            for v in values
        ]

    # O3 Bar Plot (threshold: 40)
    o3_past_values = o3_values[:-3]  # Last 3 values are predictions
    o3_future_values = o3_values[-3:]  # Last 3 values are predictions
    o3_colors = get_simple_color_scale(o3_past_values, 40)  # Color for past values

    fig_o3 = go.Figure()

    # Add past values
    fig_o3.add_trace(
        go.Bar(
            x=df["Date"][:-3],  # Dates for past values
            y=o3_past_values,
            name="O3 Past",
            marker=dict(color=o3_colors),  # Apply the color scale
            hovertemplate="%{x|%d-%b-%Y}<br>%{y} µg/m³<extra></extra>",
        )
    )

    # Add predicted values with reduced opacity
    predicted_o3_colors = get_simple_color_scale(
        o3_future_values, 40
    )  # Color for future values
    fig_o3.add_trace(
        go.Bar(
            x=df["Date"][-3:],  # Dates for predicted values
            y=o3_future_values,
            name="O3 Predicted",
            marker=dict(
                color=predicted_o3_colors, opacity=0.5
            ),  # Set opacity to 0.5 for predictions
            hovertemplate="%{x|%d-%b-%Y}<br>%{y} µg/m³<extra></extra>",
        )
    )

    fig_o3.add_shape(
        dict(
            type="line",
            x0=pd.Timestamp.today(),
            x1=pd.Timestamp.today(),
            y0=min(o3_values),
            y1=max(o3_values),
            line=dict(color="White", width=3, dash="dash"),
        )
    )

    fig_o3.update_layout(
        plot_bgcolor="rgba(0, 0, 0, 0)",
        paper_bgcolor="rgba(0, 0, 0, 0)",
        yaxis_title="O3 Concentration (µg/m³)",
        font=dict(size=14),
        hovermode="x",
        xaxis=dict(
            title="Date",
            type="date",
            tickmode="array",
            tickvals=df["Date"],
            tickformat="%d-%b",
            tickangle=-45,
            tickcolor="gray",
        ),
        showlegend=False,  # Disable legend
    )

    st.plotly_chart(fig_o3, key="fig_o3")

    # NO2 Bar Plot (threshold: 120)
    st.subheader("NO2 Forecast")
    no2_past_values = no2_values[:-3]  # Last 3 values are predictions
    no2_future_values = no2_values[-3:]  # Last 3 values are predictions
    no2_colors = get_simple_color_scale(no2_past_values, 120)  # Color for past values

    fig_no2 = go.Figure()

    # Add past values
    fig_no2.add_trace(
        go.Bar(
            x=df["Date"][:-3],  # Dates for past values
            y=no2_past_values,
            name="NO2 Past",
            marker=dict(color=no2_colors),  # Apply the color scale
            hovertemplate="%{x|%d-%b-%Y}<br>%{y} µg/m³<extra></extra>",
        )
    )

    # Add predicted values with reduced opacity
    predicted_no2_colors = get_simple_color_scale(
        no2_future_values, 120
    )  # Color for future values
    fig_no2.add_trace(
        go.Bar(
            x=df["Date"][-3:],  # Dates for predicted values
            y=no2_future_values,
            name="NO2 Predicted",
            marker=dict(
                color=predicted_no2_colors, opacity=0.5
            ),  # Set opacity to 0.5 for predictions
            hovertemplate="%{x|%d-%b-%Y}<br>%{y} µg/m³<extra></extra>",
        )
    )

    fig_no2.add_shape(
        dict(
            type="line",
            x0=pd.Timestamp.today(),
            x1=pd.Timestamp.today(),
            y0=min(no2_values),
            y1=max(no2_values),
            line=dict(color="White", width=3, dash="dash"),
        )
    )

    fig_no2.update_layout(
        plot_bgcolor="rgba(0, 0, 0, 0)",
        paper_bgcolor="rgba(0, 0, 0, 0)",
        yaxis_title="NO<sub>2</sub> Concentration (µg/m³)",
        font=dict(size=14),
        hovermode="x",
        xaxis=dict(
            title="Date",
            type="date",
            tickmode="array",
            tickvals=df["Date"],
            tickformat="%d-%b",
            tickangle=-45,
            tickcolor="gray",
        ),
        showlegend=False,  # Disable legend
    )

    st.plotly_chart(fig_no2, key="fig_no2")
