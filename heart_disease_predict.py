import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from plotly import graph_objs as go

# Load the trained model
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

# Load the dataset
dataset_path = "Heart_Disease_Prediction.csv"  # Replace with your dataset file path
df = pd.read_csv(dataset_path)

# Sidebar
st.sidebar.title("Navigation")
section = st.sidebar.radio("Go to", ["Home", "Prediction", "Dataset"])

# Home Section
if section == "Home":
    st.title("Heart Disease Prediction webpage")
    st.write(
        """
        Welcome to the Heart Disease Prediction Application.
        
        This tool uses a machine learning model to predict the likelihood of heart disease based on various health parameters.
        
        **Explore the Sections**:
        - **Prediction**: Input your health parameters to get a prediction.
        - **Dataset**: View the dataset used for training the model.
        """
    )
    st.image("home_image.png", caption="Heart Health Awareness", use_column_width=True)

# Prediction Section
elif section == "Prediction":
    st.image("prediction_image.png")
    st.title("Heart Disease Prediction")
    st.write("Enter the following details:")

    # Input fields for prediction
    age = st.number_input("Age", min_value=0, max_value=120, value=25)
    sex = st.selectbox("Sex", options=[0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
    cp = st.selectbox("Chest Pain Type (cp)", options=[0, 1, 2, 3])
    trestbps = st.slider("Resting Blood Pressure (trestbps)", 80, 200, 120)
    chol = st.slider("Serum Cholesterol in mg/dl (chol)", 100, 400, 200)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (fbs)", options=[0, 1])
    restecg = st.selectbox("Resting Electrocardiographic Results (restecg)", options=[0, 1, 2])
    thalach = st.slider("Maximum Heart Rate Achieved (thalach)", 60, 220, 150)
    exang = st.selectbox("Exercise Induced Angina (exang)", options=[0, 1])
    oldpeak = st.number_input("ST Depression Induced by Exercise (oldpeak)", value=1.0, step=0.1)
    slope = st.selectbox("Slope of the Peak Exercise ST Segment (slope)", options=[0, 1, 2])
    ca = st.selectbox("Number of Major Vessels (ca)", options=[0, 1, 2, 3, 4])
    thal = st.selectbox("Thalassemia (thal)", options=[0, 1, 2, 3])

    # Prediction button
    if st.button("Predict"):
        features = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
        prediction = model.predict(features)
        result = "Heart Disease Detected" if prediction[0] == 1 else "No Heart Disease"
        st.success(f"Prediction: {result}")

# Dataset Section
elif section == "Dataset":
    st.title("Dataset")
    st.write("Here is the dataset used for training the model:")
    if st.checkbox("Show Table"):
        st.table(df.head(20))

    graph = st.selectbox("What kind of Graph?", ["Non-interactive", "Interactive"])

    val = st.slider("Filter data using Cholesterol level", 0, 500)
    filtered_data = df[df["Cholesterol"] >= val]

    if graph == "Non-interactive":
        set1 = ['blue', 'red']

        fig= sns.lmplot(
       
            y="Age",
            x="Cholesterol",
            hue="Heart Disease",
            col="Sex",
            palette=set1,
            data=filtered_data,
            height=5,
            aspect=1
           
        )
        st.pyplot(fig)

    elif graph == "Interactive":
        st.title("Hexbin Plot in Streamlit with Plotly")

        x_column = st.selectbox("Select X-axis Column", df.columns)
        y_column = st.selectbox("Select Y-axis Column", df.columns)
        bin_size = st.slider("Select Bin Size", 5, 50, 20)

        # Create the hexbin plot using Plotly
        fig = go.Figure()

        fig.add_trace(
            go.Histogram2d(
                x=filtered_data[x_column],
                y=filtered_data[y_column],
                colorscale="Reds",
                nbinsx=bin_size,
                nbinsy=bin_size,
                colorbar=dict(title="Density"),
            )
        )

        fig.update_layout(
            title=f"Hexbin Plot: {x_column} vs {y_column}",
            xaxis_title=x_column,
            yaxis_title=y_column,
            plot_bgcolor="white",
            width=800,
            height=600,
        )

        st.plotly_chart(fig)
