import streamlit as st


def custom_metric_box(label: str, value: str) -> None:
    """
    Create a styled metric box with a compact layout.

    This function generates a styled markdown box displaying a label and its corresponding value.

    Parameters:
    ----------
    label : str
        The text label to display in the metric box.

    value : str
        The value to be displayed in the metric box, typically representing a metric.

    Returns:
    -------
    None
    """
    st.markdown(
        f"""
        <div style="
            padding: 5px;
            margin-bottom: 5px;
            width: 100%;  /* Full width */
            display: flex;
            flex-direction: column;  /* Align items vertically */
            align-items: flex-start;  /* Align all content to the left */
        ">
            <div>
                <h4 style="font-size: 14px; font-weight: normal; margin: 0;">{label}</h4>  <!-- Smaller label -->
            </div>
            <div>
                <p style="font-size: 18px; font-weight: bold; margin: 0;">{value}</p>  <!-- Smaller metric -->
            </div>
        </div>
    """,
        unsafe_allow_html=True,
    )


def pollution_box(label: str, value: str, delta: str, threshold: float) -> None:
    """
    Create a pollution metric box with a side-by-side layout and fixed width.

    This function generates a styled markdown box displaying pollution level status, value, and other related information.

    Parameters:
    ----------
    label : str
        The text label representing the type of pollution or metric.

    value : str
        The value of the pollution metric, typically a string that can be converted to a float.

    delta : str
        A string representing the change in pollution level, though not currently used in the rendering.

    threshold : float
        The threshold value to determine if the pollution level is "Good" or "Bad".

    Returns:
    -------
    None
    """
    # Determine if the pollution level is "Good" or "Bad"
    status = "Good" if float(value.split()[0]) < threshold else "Bad"
    status_color = "#77C124" if status == "Good" else "#E68B0A"

    # Render the pollution box
    st.markdown(
        f"""
        <div style="
            background: rgba(255, 255, 255, 0.05);
            border-radius: 16px;
            box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);
            backdrop-filter: blur(5px);
            -webkit-backdrop-filter: blur(5px);
            border: 1px solid rgba(255, 255, 255, 0.15);
            padding: 15px;
            margin-bottom: 10px;
        ">
            <h4 style="font-size: 24px; font-weight: bold; margin: 0;">{label}</h4>  <!-- Bigger label -->
            <p style="font-size: 36px; font-weight: bold; color: {status_color}; margin: 0;">{status}</p>  <!-- Good/Bad with color -->
            <p style="font-size: 18px; margin: 0;">{value}</p>  <!-- Smaller value where delta used to be -->
        </div>
    """,
        unsafe_allow_html=True,
    )
