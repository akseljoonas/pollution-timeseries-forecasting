import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import mean_squared_error
from src.data_api_calls import get_combined_data

USERNAME = "admin"
PASSWORD = "password"

st.title("Admin Panel")

# Use session state to remember login state
if "login_success" not in st.session_state:
    st.session_state.login_success = False

# Login Form
if not st.session_state.login_success:
    with st.form("login_form"):
        st.write("Please login to access the admin dashboard:")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        login_button = st.form_submit_button("Login")

        if login_button:
            if username == USERNAME and password == PASSWORD:
                st.session_state.login_success = True
                st.success("Login successful!")
            else:
                st.error("Invalid username or password.")
else:
    # Fetching the combined data
    table_data = get_combined_data()

    # Check for missing values
    missing_values = table_data.isnull()

    # Display the main data table
    st.subheader("Data used for the prediction")

    # Display message based on whether data is complete
    if missing_values.values.any():
        # Warning message if there are missing values
        st.markdown(
            "<h4 style='color: #E68B0A;'>Warning: Some data is missing!</h4>",
            unsafe_allow_html=True,
        )

        # Identify columns with missing values
        missing_columns = table_data.columns[missing_values.any()].tolist()

        # Identify rows (dates) with missing values
        missing_rows = table_data[missing_values.any(axis=1)]["Date"].tolist()

        # Display additional information about missing columns and rows
        if missing_columns:
            st.markdown(f"**Columns with missing data:** {', '.join(missing_columns)}")
        if missing_rows:
            st.markdown(
                f"**Rows with missing data (dates):** {', '.join(missing_rows)}"
            )
    else:
        # Success message if no data is missing
        st.markdown(
            "<h4 style='color: #77C124;'>All data is complete!</h4>",
            unsafe_allow_html=True,
        )
    st.dataframe(table_data)
    # Actual data vs 1,2,3 days ahead predictions
    actual_data = pd.read_csv("pollution_data.csv")
    prediction_data = pd.read_csv("predictions_history.csv")

    col1, col2 = st.columns(2)
    with col1:
        pollutant = st.radio("Select a pollutant", ("O3", "NO2"))
    with col2:
        days_ahead = st.radio("Select days ahead for prediction", (1, 2, 3))

    predictions = prediction_data[prediction_data["pollutant"] == pollutant]
    actual = actual_data[["date", pollutant]].rename(
        columns={pollutant: "actual_value"}
    )

    predictions_filtered = predictions[
        predictions["date_predicted"]
        == (
            pd.to_datetime(predictions["date"]) - pd.Timedelta(days=days_ahead)
        ).dt.strftime("%Y-%m-%d")
    ]

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=actual["date"],
            y=actual["actual_value"],
            mode="lines+markers",
            name="Ground Truth",
            line=dict(color="green", width=3),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=predictions_filtered["date"],
            y=predictions_filtered["prediction_value"],
            mode="lines+markers",
            name=f"Prediction {days_ahead} day(s) ahead",
            line=dict(dash="dash", color="orange", width=3),
        )
    )

    fig.update_layout(
        title=f"{pollutant} Predictions vs Actual Values",
        xaxis_title="Date",
        yaxis_title=f"{pollutant} Concentration",
        legend=dict(x=0, y=1),
        yaxis=dict(range=[0, 60]),
        template="plotly_white",
        xaxis=dict(
            title="Date",
            type="date",
            tickmode="array",
            tickvals=predictions["date"],
            tickformat="%d-%b",
            tickangle=-45,
            tickcolor="gray",
        ),
    )

    st.plotly_chart(fig)

    # Evaluation Function
    def evaluate_predictions_all_days(actual, predictions):
        rmse_values_all = {"O3": [], "NO2": []}
        smape_values_all = {"O3": [], "NO2": []}

        for pollutant in ["O3", "NO2"]:
            predictions_pollutant = predictions[predictions["pollutant"] == pollutant]
            actual_pollutant = actual_data[["date", pollutant]].rename(
                columns={pollutant: "actual_value"}
            )

            # Calculate RMSE and SMAPE for each day (1st, 2nd, and 3rd)
            for i in range(1, 4):
                predictions_filtered = predictions_pollutant[
                    predictions_pollutant["date_predicted"]
                    == (
                        pd.to_datetime(predictions_pollutant["date"])
                        - pd.Timedelta(days=i)
                    ).dt.strftime("%Y-%m-%d")
                ]
                actual_filtered = actual_pollutant[
                    actual_pollutant["date"].isin(predictions_filtered["date"])
                ]
                merged = pd.merge(
                    actual_filtered,
                    predictions_filtered,
                    left_on="date",
                    right_on="date",
                )

                if not merged.empty:
                    actual_values = merged["actual_value"].values
                    prediction_values = merged["prediction_value"].values

                    rmse = np.sqrt(mean_squared_error(actual_values, prediction_values))
                    rmse_values_all[pollutant].append(rmse)
                    smape = (
                        100
                        / len(actual_values)
                        * np.sum(
                            2
                            * np.abs(prediction_values - actual_values)
                            / (np.abs(actual_values) + np.abs(prediction_values))
                        )
                    )
                    smape_values_all[pollutant].append(smape)

        # Plot RMSE and SMAPE for both pollutants
        fig_rmse = go.Figure()
        for day in range(3):
            fig_rmse.add_trace(
                go.Bar(
                    x=["O3", "NO2"],
                    y=[rmse_values_all["O3"][day], rmse_values_all["NO2"][day]],
                    name=f"Day {day + 1}",
                )
            )
        fig_rmse.update_layout(
            title="RMSE for Predictions Over 3 Days",
            yaxis_title="RMSE",
            xaxis_title="Pollutant",
            barmode="group",
        )
        st.plotly_chart(fig_rmse)

        fig_smape = go.Figure()
        for day in range(3):
            fig_smape.add_trace(
                go.Bar(
                    x=["O3", "NO2"],
                    y=[smape_values_all["O3"][day], smape_values_all["NO2"][day]],
                    name=f"Day {day + 1}",
                )
            )
        fig_smape.update_layout(
            title="SMAPE for Predictions Over 3 Days",
            yaxis_title="SMAPE (%)",
            xaxis_title="Pollutant",
            barmode="group",
        )
        st.plotly_chart(fig_smape)

        # Calculate total current SMAPE and RMSE
        total_O3_smape = sum(smape_values_all["O3"]) / len(smape_values_all)
        total_NO2_smape = sum(smape_values_all["NO2"]) / len(smape_values_all)
        total_O3_rmse = sum(rmse_values_all["O3"]) / len(rmse_values_all)
        total_NO2_rmse = sum(rmse_values_all["NO2"]) / len(rmse_values_all)

        # Display metrics table
        metrics_data = {
            "Metric": [
                "Current NO2 SMAPE (%)",
                "Current NO2 RMSE (µg/m3)",
                "Current O3 SMAPE (%)",
                "Current O3 RMSE (µg/m3)",
            ],
            "Value": [total_NO2_smape, total_NO2_rmse, total_O3_smape, total_O3_rmse],
        }
        metrics_df = pd.DataFrame(metrics_data)
        st.table(metrics_df)

    evaluate_predictions_all_days(actual_data, prediction_data)
