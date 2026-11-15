import plotly.graph_objects as go
from src.predict import predict_cvd
import streamlit as st

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="CardioAI | Cardiovascular Prediction",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------------------------------
# CUSTOM DESIGN
# --------------------------------------------------


st.markdown("""
<style>

/* Main background */
.stApp {
    background: #F5F7FA;
}

/* Main content width and spacing */
.block-container {
    max-width: 1250px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

/* Headings */
h1, h2, h3 {
    color: #16324F !important;
    font-weight: 700;
    line-height: 1.35 !important;
}

/* Normal text */
.stMarkdown p {
    color: #46576B;
    line-height: 1.65;
}

/* Hero section */
.hero {
    background: linear-gradient(120deg, #102A43, #087F8C);
    padding: 32px;
    border-radius: 16px;
    margin-bottom: 25px;
}

.hero h1 {
    color: #FFFFFF !important;
    font-size: 34px;
    line-height: 1.3;
}

.hero p {
    color: #E5F4F4 !important;
    font-size: 16px;
    line-height: 1.7;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #102A43;
}

[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] label {
    color: #FFFFFF !important;
}

/* Input labels */
[data-testid="stWidgetLabel"] p {
    color: #263B50 !important;
    font-weight: 600;
    line-height: 1.5;
}

/* Input fields */
.stNumberInput input,
.stSelectbox div[data-baseweb="select"] {
    border-radius: 8px;
}

/* Section separators */
hr {
    border-color: #DCE4EC;
    margin-top: 22px;
    margin-bottom: 22px;
}

/* Buttons */
.stFormSubmitButton button {
    background-color: #087F8C;
    color: white;
    border: none;
    border-radius: 9px;
    min-height: 48px;
    font-weight: 600;
    width: 100%;
}

.stFormSubmitButton button:hover {
    background-color: #065F68;
    color: white;
}

/* Metrics */
[data-testid="stMetric"] {
    background: white;
    padding: 18px;
    border: 1px solid #E2E8F0;
    border-radius: 12px;
}

/* Alerts */
.stAlert {
    border-radius: 10px;
}

/* Responsive headings */
@media (max-width: 768px) {
    .hero {
        padding: 22px;
    }

    .hero h1 {
        font-size: 27px;
    }

    .block-container {
        padding-left: 1rem;
        padding-right: 1rem;
    }
}

</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# APPLICATION HEADER
# --------------------------------------------------


st.markdown("""
<style>

/* Main dark background */
.stApp {
    background: #0B1726;
    color: #EAF2F8;
}

/* Main layout */
.block-container {
    max-width: 1250px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

/* Headings */
h1, h2, h3 {
    color: #EAF4FA !important;
    font-weight: 700;
    line-height: 1.4 !important;
}

/* General text */
.stMarkdown p {
    color: #B8C7D8;
    line-height: 1.7;
}

/* Hero section */
.hero {
    background: linear-gradient(120deg, #142C46, #075E68);
    padding: 35px;
    border: 1px solid #24546B;
    border-radius: 18px;
    margin-bottom: 28px;
}

.hero h1 {
    color: #FFFFFF !important;
    font-size: 36px;
}

.hero p {
    color: #D4F1F0 !important;
    font-size: 16px;
    line-height: 1.7;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #101F32;
    border-right: 1px solid #263D55;
}

[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] label {
    color: #EAF2F8 !important;
}

/* Form labels */
[data-testid="stWidgetLabel"] p {
    color: #DCE8F2 !important;
    font-weight: 600;
    line-height: 1.5;
}

/* Input fields */
.stNumberInput input {
    background: #172B40 !important;
    color: #FFFFFF !important;
    border-radius: 8px;
}

.stSelectbox div[data-baseweb="select"] {
    background: #172B40 !important;
    border-radius: 8px;
}

/* Input borders */
div[data-baseweb="select"] > div {
    border-color: #36516B !important;
}

/* Section dividers */
hr {
    border-color: #2B4057;
    margin-top: 24px;
    margin-bottom: 24px;
}

/* Buttons */
.stFormSubmitButton button {
    background: #0D9BA6;
    color: #FFFFFF;
    border: none;
    border-radius: 10px;
    min-height: 50px;
    font-weight: 700;
}

.stFormSubmitButton button:hover {
    background: #087B85;
    color: #FFFFFF;
}

/* Metrics */
[data-testid="stMetric"] {
    background: #14263A;
    padding: 20px;
    border: 1px solid #29435D;
    border-radius: 14px;
}

[data-testid="stMetricLabel"] {
    color: #B8C7D8 !important;
}

[data-testid="stMetricValue"] {
    color: #47D6CE !important;
}

/* Alerts */
.stAlert {
    border-radius: 12px;
}

/* Responsive layout */
@media (max-width: 768px) {
    .hero {
        padding: 23px;
    }

    .hero {
    text-align: center;
}

.hero h1,
.hero h2,
.hero p {
    text-align: center;
}

</style>
""", unsafe_allow_html=True)


st.markdown(
    "<p style='text-align: center; font-size: 20px; font-weight: 500;'>"
    "Machine learning-powered insights into cardiovascular health"
    "</p>",
    unsafe_allow_html=True
)


# --------------------------------------------------
# PATIENT INFORMATION FORM
# --------------------------------------------------

st.markdown("## 👤 Patient Information")
st.write(
    "Enter the information below. All fields are required "
    "to generate a prediction."
)

with st.form("patient_information_form"):

    # PERSONAL INFORMATION
    st.markdown("### Personal Details")

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input(
            "Age (years)",
            min_value=1,
            max_value=120,
            value=50,
            step=1,
            help="Enter age in years."
        )

    with col2:
        gender = st.selectbox(
            "Gender",
            options=[1, 2],
            format_func=lambda x: {
                1: "Male",
                2: "Female"
            }[x]
        )

    with col3:
        height = st.number_input(
            "Height (cm)",
            min_value=100.0,
            max_value=230.0,
            value=165.0,
            step=1.0
        )

    col1, col2 = st.columns(2)

    with col1:
        weight = st.number_input(
            "Weight (kg)",
            min_value=20.0,
            max_value=300.0,
            value=70.0,
            step=0.5
        )

    with col2:
        st.markdown(
            "**Body measurements**"
        )
        st.caption(
            "Height and weight are used as separate "
            "features by the machine learning model."
        )

    st.divider()

    # BLOOD PRESSURE
    st.markdown("### 🩺 Blood Pressure")

    col1, col2 = st.columns(2)

    with col1:
        systolic_bp = st.number_input(
            "Systolic Blood Pressure (mmHg)",
            min_value=60,
            max_value=300,
            value=120,
            step=1,
            help="The upper blood pressure reading."
        )

    with col2:
        diastolic_bp = st.number_input(
            "Diastolic Blood Pressure (mmHg)",
            min_value=30,
            max_value=200,
            value=80,
            step=1,
            help="The lower blood pressure reading."
        )

    st.divider()

    # HEALTH INDICATORS
    st.markdown("### 🧪 Health Indicators")

    col1, col2 = st.columns(2)

    with col1:
        cholesterol = st.selectbox(
            "Cholesterol Level",
            options=[1, 2, 3],
            format_func=lambda x: {
                1: "Normal",
                2: "Above Normal",
                3: "Well Above Normal"
            }[x]
        )

    with col2:
        glucose = st.selectbox(
            "Glucose Level",
            options=[1, 2, 3],
            format_func=lambda x: {
                1: "Normal",
                2: "Above Normal",
                3: "Well Above Normal"
            }[x]
        )

    st.divider()

    # LIFESTYLE INFORMATION
    st.markdown("### 🏃 Lifestyle Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        smoke = st.selectbox(
            "Smoking",
            options=[0, 1],
            format_func=lambda x: {
                0: "No",
                1: "Yes"
            }[x]
        )

    with col2:
        alcohol = st.selectbox(
            "Alcohol Intake",
            options=[0, 1],
            format_func=lambda x: {
                0: "No",
                1: "Yes"
            }[x]
        )

    with col3:
        physical_activity = st.selectbox(
            "Physical Activity",
            options=[0, 1],
            format_func=lambda x: {
                0: "Inactive",
                1: "Active"
            }[x]
        )

    st.divider()

    submitted = st.form_submit_button(
        "🫀 Generate Prediction",
        use_container_width=True
    )


# --------------------------------------------------
# FORM VALIDATION
# --------------------------------------------------

if submitted:

    if systolic_bp <= diastolic_bp:
        st.error(
            "Please check your blood pressure readings. "
            "Systolic pressure must be greater than "
            "diastolic pressure for this application."
        )

    else:

        patient_input = {
            "age": age,
            "gender": gender,
            "height": height,
            "weight": weight,
            "systolic_bp": systolic_bp,
            "diastolic_bp": diastolic_bp,
            "cholesterol": cholesterol,
            "glucose": glucose,
            "smoke": smoke,
            "alcohol": alcohol,
            "physical_activity": physical_activity
        }

        st.session_state["patient_input"] = patient_input

        st.success(
            "Patient information captured successfully!"
        )


# --------------------------------------------------
# MODEL PREDICTION
# --------------------------------------------------

if submitted and "patient_input" in st.session_state:

    patient_input = st.session_state["patient_input"]

    st.markdown("---")
    st.markdown("## 📊 Prediction Results")

    with st.spinner("Analyzing patient information..."):

        prediction, probability = predict_cvd(patient_input)

    # Get probability belonging to class 1
    # Class 1 = dataset label for cardiovascular disease
    disease_probability = float(probability)

    # Display prediction result

    result_col1,result_col2 = st.columns (2)

    with result_col1:
        st.markdown ("### Prediction Outcome")

        if prediction == 1:
            st.error ("Possible Cardiovascular Disease Pattern Detected 😟")
            st.write ("Predicted dataset class: **1**")

        else:
            st.success ("No Cardiovascular Disease Pattern Detected 😊")
            st.write ("Predicted dataset class: **0**")

    with result_col2:
        st.markdown ("### Model Probability")

        st.metric (
            label="Estimated probability of class 1",
            value=f"{disease_probability * 100:.2f}%"
        )

    # Probability visualization
    st.markdown("### Probability Visualization")

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=disease_probability * 100,
            number={"suffix": "%"},
            title={
                "text": "Estimated probability of class 1"
            },
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#087F8C"},
                "steps": [
                    {"range": [0, 50], "color": "#E3F2FD"},
                    {"range": [50, 100], "color": "#B2DFDB"}
                ]
            }
        )
    )

    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=50, b=20)
    )

    st.plotly_chart(fig, use_container_width=True)

    st.info(
        "This probability is the model's estimated probability "
        "for dataset class 1. It is not a clinically validated "
        "estimate of an individual's medical risk."
    )

    st.warning(
        "This application is for educational and research "
        "purposes only. It is not a medical diagnostic tool. "
        "Consult a qualified healthcare professional for "
        "medical assessment."
    )
