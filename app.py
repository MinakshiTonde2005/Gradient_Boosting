```python
import streamlit as st
import pandas as pd
import joblib
import os

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Gradient Boosting Predictor",
    page_icon="🤖",
    layout="wide"
)

# =========================================================
# CUSTOM CSS
# =========================================================
st.markdown("""
<style>

.stApp {
    background: linear-gradient(135deg, #eef2ff, #f8fafc);
}

.header {
    background: linear-gradient(135deg, #4f46e5, #7c3aed);
    padding: 30px;
    border-radius: 20px;
    text-align: center;
    color: white;
    box-shadow: 0px 8px 25px rgba(0,0,0,0.15);
    margin-bottom: 30px;
}

.header h1 {
    font-size: 40px;
    margin-bottom: 5px;
}

.header p {
    font-size: 18px;
}

.card {
    background: white;
    padding: 25px;
    border-radius: 18px;
    box-shadow: 0px 6px 20px rgba(0,0,0,0.10);
    margin-bottom: 20px;
}

.result {
    background: white;
    padding: 30px;
    border-radius: 18px;
    text-align: center;
    box-shadow: 0px 8px 25px rgba(0,0,0,0.12);
}

.result h2 {
    color: #4f46e5;
}

.result-value {
    font-size: 42px;
    font-weight: bold;
    color: #7c3aed;
}

.stButton > button {
    width: 100%;
    height: 52px;
    border-radius: 12px;
    border: none;
    background: linear-gradient(135deg, #4f46e5, #7c3aed);
    color: white;
    font-size: 18px;
    font-weight: bold;
}

.stButton > button:hover {
    background: linear-gradient(135deg, #3730a3, #6d28d9);
    color: white;
}

.info-box {
    padding: 15px;
    background: #eef2ff;
    border-left: 5px solid #4f46e5;
    border-radius: 10px;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# LOAD MODEL
# =========================================================
MODEL_FILE = "GradientBoosting.pkl"

if not os.path.exists(MODEL_FILE):
    st.error(
        "❌ GradientBoosting.pkl not found. "
        "Make sure the .pkl file is in the same folder as app.py."
    )
    st.stop()


@st.cache_resource
def load_model():
    return joblib.load(MODEL_FILE)


try:
    model = load_model()
except Exception as e:
    st.error("❌ Error while loading the model.")
    st.code(str(e))
    st.stop()

# =========================================================
# HEADER
# =========================================================
st.markdown("""
<div class="header">
    <h1>🤖 Gradient Boosting Predictor</h1>
    <p>Machine Learning Prediction Dashboard</p>
</div>
""", unsafe_allow_html=True)

# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:

    st.title("⚙️ Model Details")

    st.markdown("""
    **Algorithm**

    Gradient Boosting Classifier

    **Features**

    • Age  
    • Gender  
    • Review  
    • Education

    **Model Type**

    Classification
    """)

    st.markdown("---")

    st.info(
        "Enter the required information and click "
        "Predict Result."
    )

# =========================================================
# INPUT SECTION
# =========================================================
st.markdown("## 📝 Enter Details")

col1, col2 = st.columns(2)

# =========================================================
# COLUMN 1
# =========================================================
with col1:

    st.markdown('<div class="card">', unsafe_allow_html=True)

    age = st.number_input(
        "👤 Age",
        min_value=1,
        max_value=100,
        value=25,
        step=1
    )

    gender = st.selectbox(
        "⚧ Gender",
        options=[0, 1],
        format_func=lambda x:
            "Category 0" if x == 0 else "Category 1"
    )

    st.markdown('</div>', unsafe_allow_html=True)

# =========================================================
# COLUMN 2
# =========================================================
with col2:

    st.markdown('<div class="card">', unsafe_allow_html=True)

    review = st.selectbox(
        "⭐ Review",
        options=[0, 1, 2, 3],
        format_func=lambda x: f"Category {x}"
    )

    education = st.selectbox(
        "🎓 Education",
        options=[0, 1, 2, 3],
        format_func=lambda x: f"Category {x}"
    )

    st.markdown('</div>', unsafe_allow_html=True)

# =========================================================
# PREDICTION
# =========================================================
st.markdown("---")

if st.button("🔮 PREDICT RESULT"):

    try:

        # IMPORTANT:
        # Keep the feature names and order exactly
        # as required by the trained model.

        input_data = pd.DataFrame({
            "age": [age],
            "gender": [gender],
            "review": [review],
            "education": [education]
        })

        # Prediction
        prediction = model.predict(input_data)[0]

        # =====================================================
        # RESULT
        # =====================================================
        st.markdown(
            '<div class="result">',
            unsafe_allow_html=True
        )

        st.markdown(
            "<h2>🎯 Prediction Result</h2>",
            unsafe_allow_html=True
        )

        st.markdown(
            f'<div class="result-value">{prediction}</div>',
            unsafe_allow_html=True
        )

        # Probability
        if hasattr(model, "predict_proba"):

            probability = model.predict_proba(
                input_data
            )[0]

            confidence = max(probability) * 100

            st.write(
                f"**Confidence: {confidence:.2f}%**"
            )

        st.markdown(
            '</div>',
            unsafe_allow_html=True
        )

        # =====================================================
        # INPUT SUMMARY
        # =====================================================
        st.markdown("## 📊 Input Summary")

        result_df = pd.DataFrame({
            "Feature": [
                "Age",
                "Gender",
                "Review",
                "Education"
            ],
            "Value": [
                age,
                f"Category {gender}",
                f"Category {review}",
                f"Category {education}"
            ]
        })

        st.dataframe(
            result_df,
            use_container_width=True,
            hide_index=True
        )

    except Exception as e:

        st.error("❌ Prediction Error")
        st.code(str(e))

# =========================================================
# FOOTER
# =========================================================
st.markdown("---")

st.markdown(
    "<center>🚀 Gradient Boosting ML App | Built with Streamlit</center>",
    unsafe_allow_html=True
)
```
