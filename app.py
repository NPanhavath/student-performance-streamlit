import streamlit as st
import pandas as pd
import joblib

# -------------------------------
# 1. Page config
# -------------------------------
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="centered"
)

# -------------------------------
# 2. Load the trained pipeline
#    (ColumnTransformer + StandardScaler + SGDRegressor)
# -------------------------------
@st.cache_resource
def load_model():
    model = joblib.load(
        r"student_performance_model.pkl"
    )
    return model

model = load_model()

# -------------------------------
# 3. Title & description
# -------------------------------
st.title("🎓 Student Performance Prediction")

st.markdown("""
This app predicts a student's **Performance Index** using:

- `Previous Scores`
- `Hours Studied`

The model already includes **preprocessing + training** inside a saved pipeline.
""")

# -------------------------------
# 4. Inputs (ONLY 2 FEATURES)
# -------------------------------
previous_scores = st.slider(
    "Previous Scores",
    min_value=40,
    max_value=100,
    value=None,
    help="Student's previous academic scores (40–100)"
)

hours_studied = st.slider(
    "Hours Studied per Day",
    min_value=0,
    max_value=24,
    value=None,
    help="Number of hours the student studies per day (0-24)"
)

# -------------------------------
# 5. Prediction
# -------------------------------
st.markdown("---")

if st.button("🔮 Predict Performance Index", type="primary", use_container_width=True):
    # Build a 1-row DataFrame with the SAME column names as training
    input_data = pd.DataFrame(
        [[previous_scores, hours_studied]],
        columns=["Previous Scores", "Hours Studied"]
    )

    # Pipeline handles scaling + prediction internally
    prediction = model.predict(input_data)[0]

    # 🔒 Clip the prediction to the valid range 0–100
    prediction = max(0, min(100, prediction))

    # Display result
    st.subheader("📊 Prediction Result")

    if prediction >= 80:
        color = "green"
        emoji = "🌟"
        message = "Excellent Performance!"
    elif prediction >= 60:
        color = "orange"
        emoji = "👍"
        message = "Good Performance"
    else:
        color = "red"
        emoji = "📉"
        message = "Needs Improvement"

    st.markdown(
        f"""
        <div style='text-align: center; padding: 20px; background-color: {color}15;
                    border-radius: 10px; border: 2px solid {color}'>
            <h1 style='color: {color}; margin: 0;'>{emoji} {prediction:.2f}</h1>
            <h3 style='color: {color}; margin-top: 10px;'>{message}</h3>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Show a tiny summary
    st.markdown("### 📝 Input Summary")
    st.write(f"- **Previous Scores**: {previous_scores}/100")
    st.write(f"- **Hours Studied**: {hours_studied} hours/day")

# -------------------------------
# 6. Footer
# -------------------------------
st.markdown("---")
st.markdown("""
### 📌 About the Model
- **Algorithm**: Linear Regression trained with `SGDRegressor`
- **Features used**:
  - `Previous Scores`
  - `Hours Studied`
- **Preprocessing**:
  - `StandardScaler` applied inside the pipeline (`ColumnTransformer`)

The Streamlit app just sends these 2 values into the saved pipeline and shows the prediction.
""")

st.caption("Made with ❤️ using Streamlit")
