```python
import streamlit as st
import pandas as pd
import joblib

# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="Gradient Boosting Prediction",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM CSS
# ============================================================
st.markdown("""
<style>

    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #eef2ff, #f8fafc);
    }

    /* Header */
    .main-header {
        background: linear-gradient(135deg, #4f46e5, #7c3aed);
        padding: 25px;
        border-radius: 18px;
        text-align: center;
        color: white;
        margin-bottom: 25px;
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
    }

    .main-header h1 {
        margin: 0;
        font-size: 38px;
        font-weight: 700;
    }

    .main-header p {
        margin-top: 8px;
        font-size: 17px;
    }

    /* Input card */
    .input-card {
        background: white;
        padding: 25px;
        border-radius: 18px;
        box-shadow: 0 6px 20px rgba(0,0,0,0.10);
        margin-bottom: 20px;
    }

    /* Prediction result */
    .prediction-card {
        background: white;
        padding: 30px;
        border-radius: 18px;
        text-align: center;
        box-shadow: 0 8px 25px rgba(0,0,0,0.12);
        margin-top: 20px;
    }

    .prediction-title {
        font-size: 22px;
        font-weight: 600;
        margin-bottom: 10px;
    }

    .prediction-value {
        font-size: 42px;
        font-weight: 800;
        color: #4f46e5;
    }

    /* Button */
    div.stButton > button {
        width: 100%;
        border-radius: 12px;
        height: 50px;
        font-size: 18px;
        font-weight: 600;
        background: linear-gradient(135deg, #4f46e5, #7c3aed);
        color: white;
        border: none;
        box-shadow: 0 5px 15px rgba(79,70,229,0.30);
    }

    div.stButton > button:hover {
        background: linear-gradient(135deg, #4338ca, #6d28d9);
        color: white;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: #ffffff;
    }

    /* Info boxes */
    .info-box {
        background: #f1f5f9;
        padding: 15px;
        border-radius: 12px;
        border-left: 5px solid #4f46e5;
        margin-top: 15px;
    }

</style>
""", unsafe_allow_html=True)

# ============================================================
# LOAD MODEL
# ============================================================
@st.cache_resource
def load_model():
    return joblib.load("GradientBoosting.pkl")

try:
    model = load_model()
except Exception as e:
    st.error("❌ Model could not be loaded.")
    st.code(str(e))
    st.stop()

# ============================================================
# HEADER
# ============================================================
st.markdown("""
<div class="main-header">
    <h1>🤖 Gradient Boosting Prediction</h1>
    <p>Machine Learning Prediction System</p>
</div>
""", unsafe_allow_html=True)

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.title("⚙️ Model Information")

    st.markdown("""
    **Algorithm:**  
    Gradient Boosting Classifier

    **Input Features:**  
    • Age  
    • Gender  
    • Review  
    • Education

    **Prediction:**  
    No / Yes
    """)

    st.markdown("---")

    st.info(
        "Select the categorical values and enter the age, "
        "then click Predict."
    )

# ============================================================
# INPUT SECTION
# ============================================================
st.markdown("### 📝 Enter Input Details")

col1, col2 = st.columns(2)

with col1:
    st.markdown('<div class="input-card">', unsafe_allow_html=True)

    age = st.number_input(
        "👤 Age",
        min_value=1,
        max_value=100,
        value=30,
        step=1
    )

    # Gender is encoded as 0 / 1 in the trained model
    gender = st.selectbox(
        "⚧ Gender",
        options=[0, 1],
        format_func=lambda x: "Category 0" if x == 0 else "Category 1"
    )

    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown('<div class="input-card">', unsafe_allow_html=True)

    # Review is categorical and encoded numerically
    review = st.selectbox(
        "⭐ Review",
        options=[0, 1, 2, 3],
        format_func=lambda x: f"Category {x}"
    )

    # Education is categorical and encoded numerically
    education = st.selectbox(
        "🎓 Education",
        options=[0, 1, 2, 3],
        format_func=lambda x: f"Category {x}"
    )

    st.markdown("</div>", unsafe_allow_html=True)

# ============================================================
# PREDICTION BUTTON
# ============================================================
st.markdown("---")

if st.button("🔮 Predict Result"):

    # IMPORTANT:
    # The model was trained using these exact feature names
    # and this exact order.
    input_data = pd.DataFrame({
        "age": [age],
        "gender": [gender],
        "review": [review],
        "education": [education]
    })

    try:
        prediction = model.predict(input_data)[0]

        # Probability, if available
        probability = None

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(input_data)[0]
            probability = max(probabilities) * 100

        # ====================================================
        # RESULT
        # ====================================================
        st.markdown("""
        <div class="prediction-card">
            <div class="prediction-title">
                🎯 Prediction Result
            </div>
        """, unsafe_allow_html=True)

        st.markdown(
            f'<div class="prediction-value">{prediction}</div>',
            unsafe_allow_html=True
        )

        if probability is not None:
            st.markdown(
                f"<p><b>Confidence:</b> {probability:.2f}%</p>",
                unsafe_allow_html=True
            )

        st.markdown("</div>", unsafe_allow_html=True)

        # ====================================================
        # INPUT SUMMARY
        # ====================================================
        st.markdown("### 📊 Input Summary")

        summary = pd.DataFrame({
            "Feature": [
                "Age",
                "Gender",
                "Review",
                "Education"
            ],
            "Selected Value": [
                age,
                f"Category {gender}",
                f"Category {review}",
                f"Category {education}"
            ]
        })

        st.dataframe(
            summary,
            use_container_width=True,
            hide_index=True
        )

    except Exception as e:
        st.error("❌ Prediction failed.")
        st.code(str(e))

# ============================================================
# FOOTER
# ============================================================
st.markdown("---")

st.markdown(
    "<center>🚀 Built with Streamlit & Gradient Boosting</center>",
    unsafe_allow_html=True
)
```
