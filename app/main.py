import streamlit as st
import pickle
import pandas as pd
import plotly.graph_objects as go
import numpy as np


def get_clean_data():
    df = pd.read_csv("data/data.csv")
    df.drop(["Unnamed: 32", "id"], axis = 1, inplace = True)
    df["diagnosis"] = df["diagnosis"].map({"M": 1, "B": 0})
    return df


def get_scaled_values(input_data):
    df = get_clean_data()

    X = df.drop("diagnosis", axis = 1)
    scaled_dict = {}

    for key, value in input_data.items():
        max_val = float(X[key].max())
        min_val = float(X[key].min())
        scaled_value = (value - min_val) / (max_val - min_val)
        scaled_dict[key] = scaled_value
    return scaled_dict


def add_sidebar():
    st.sidebar.header("Cell Nuclei Measurements")

    df = get_clean_data()
    X = df.drop("diagnosis", axis = 1)

    input_dict = {}
    for col in X.columns:
        list_of_words = col.split("_")
        if len(list_of_words) > 1:
            list_of_words[0] = list_of_words[0].title()
            list_of_words[-1] = f"({list_of_words[-1]})"
            word = " ".join(list_of_words)
            input_dict[col] = st.sidebar.slider(
                word,
                min_value = 0.,
                max_value = float(X[col].max()),
                value = float(X[col].mean())
            )
        else:
            word = list_of_words[0]
            input_dict[col] = st.sidebar.slider(
                word,
                min_value = 0.,
                max_value = float(X[col].max()),
                value = float(X[col].mean())
            )
    
    return input_dict


def get_radar_chart(input_data):

    input_data = get_scaled_values(input_data)

    categories = ["Radius", "Texture", "Perimeter", "Area", "Smoothness", "Compactness", "Concavity", "Concave Points", "Symmetry", "Fractal Dimension"]

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r = [value for key, value in input_data.items() if "mean" in key],
        theta = categories,
        fill = "toself",
        name = "Mean Value"
    ))
    
    fig.add_trace(go.Scatterpolar(
        r = [value for key, value in input_data.items() if "se" in key],
        theta = categories,
        fill = "toself",
        name = "Standard Error"
    ))

    fig.add_trace(go.Scatterpolar(
        r = [value for key, value in input_data.items() if "worst" in key],
        theta = categories,
        fill = "toself",
        name = "Worst Value"
    ))

    fig.update_layout(
        polar = dict(
            radialaxis = dict(
                visible = True,
                range = [0, 1]
            )
        ),
        showlegend = True
    )
    return fig


def add_predictions(input_data):
    model = pickle.load(open("model/model.pkl", "rb"))
    scaler = pickle.load(open("model/scaler.pkl", "rb"))

    input_array = np.array(list(input_data.values())).reshape((1, -1))

    input_array_scaled = scaler.transform(input_array)

    prediction = model.predict(input_array_scaled)

    st.subheader("Cell cluster prediction")
    st.write("The cell cluster is:")

    if prediction[0] == 0:
        st.write("<span class='diagnosis benign'>Benign</span>", unsafe_allow_html = True)
    else:
        st.write("<span class='diagnosis malicious'>Malicious</span>", unsafe_allow_html = True)

    st.write("Probability of being benign: ", f"{100 * round(model.predict_proba(input_array_scaled)[0][0], 2)}%")
    st.write("Probability of being malicious: ", f"{100 * round(model.predict_proba(input_array_scaled)[0][1], 2)}%")


def main():
    st.set_page_config(
        page_title = "Breast Cancer Prediction",
        page_icon = ":female-doctor:",
        layout = "wide",
        initial_sidebar_state = "expanded")
    
    with open("assets/style.css") as f:
        st.markdown("<style>{}</style>".format(f.read()), unsafe_allow_html = True)
    
    input_data = add_sidebar()

    with st.container():
        st.title("Breast Cancer Prediction")
        st.write("""Please connect this app to your cytology lab to help diagnose breast cancer from your tissue sample.
                 This app predicts using a machine learning model whether a breast mass is benign or malignant based on the measurements it receives from your cytosis lab.
                 You can also update the measurements by hand using the sliders in the sidebar.""")
    
    col1, col2 = st.columns([4, 1])
    with col1:
        radar_chart = get_radar_chart(input_data)
        st.plotly_chart(radar_chart)

    with col2:
        add_predictions(input_data)

if __name__ == "__main__":
    main() 