import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import plotly.express as px

st.set_page_config(page_title="ML Predictor App", layout="wide")

st.title("📈 House Price Prediction App")

uploaded_file = st.file_uploader("Upload CSV Dataset", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.dataframe(df, use_container_width=True)

    target = st.selectbox("Select Target Column", df.columns)
    feature_options = [col for col in df.columns if col != target]
    features = st.multiselect("Select Feature Columns", feature_options)

    if len(features) > 0:
        X = df[features]
        y = df[target]

        X = pd.get_dummies(X, drop_first=True)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        model = LinearRegression()
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        score = r2_score(y_test, predictions)

        st.metric("Model R² Score", round(score, 4))

        results = pd.DataFrame({
            "Actual": y_test.values,
            "Predicted": predictions
        })

        fig = px.scatter(results, x="Actual", y="Predicted", trendline="ols")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Make a Prediction")

        input_data = {}
        for col in features:
            if df[col].dtype == "object":
                input_data[col] = st.selectbox(col, df[col].unique())
            else:
                min_val = float(df[col].min())
                max_val = float(df[col].max())
                input_data[col] = st.number_input(col, min_value=min_val, max_value=max_val)

        if st.button("Predict"):
            input_df = pd.DataFrame([input_data])
            input_df = pd.get_dummies(input_df)
            input_df = input_df.reindex(columns=X.columns, fill_value=0)
            prediction = model.predict(input_df)[0]
            st.success(f"Predicted Value: {prediction:.2f}")
else:
    st.info("Upload a dataset to begin.")
