import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Load the model and transformer
@st.cache_resource
def load_model():
    model = joblib.load("best_model.pkl")
    transformer = joblib.load("transformer_data.pkl")
    return model, transformer

# Create a function to get user input
def get_user_input():
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Academic Factors")
        hours_studied = st.slider("Hours Studied (per week)", 0, 45, 15)
        attendance = st.slider("Attendance (%)", 0, 100, 90)
        previous_scores = st.slider("Previous Scores", 0, 100, 70)
        tutoring_sessions = st.slider("Tutoring Sessions (per month)", 0, 20, 4)
        
    with col2:
        st.subheader("Environmental Factors")
        teacher_quality = st.selectbox("Teacher Quality", ["Low", "Medium", "High"])
        parental_involvement = st.selectbox("Parental Involvement", ["Low", "Medium", "High"])
        access_to_resources = st.selectbox("Access to Resources", ["Low", "Medium", "High"])
        internet_access = st.selectbox("Internet Access", ["Yes", "No"])
        school_type = st.selectbox("School Type", ["Public", "Private"])
        distance_from_home = st.selectbox("Distance from Home", ["Near", "Moderate", "Far"])
        
    with col3:
        st.subheader("Personal Factors")
        sleep_hours = st.slider("Sleep Hours (per night)", 0, 12, 7)
        physical_activity = st.slider("Physical Activity (hours/week)", 0, 20, 5)
        motivation_level = st.selectbox("Motivation Level", ["Low", "Medium", "High"])
        family_income = st.selectbox("Family Income", ["Low", "Medium", "High"])
        peer_influence = st.selectbox("Peer Influence", ["Negative", "Neutral", "Positive"])
        learning_disabilities = st.selectbox("Learning Disabilities", ["No", "Yes"])
        parental_education_level = st.selectbox("Parental Education Level", ["High School", "College", "Postgraduate"])
        extracurricular_activities = st.selectbox("Extracurricular Activities", ["No", "Yes"])
        gender = st.selectbox("Gender", ["Male", "Female"])
    
    # Create a DataFrame with user inputs
    user_data = {
        'Hours_Studied': hours_studied,
        'Attendance': attendance,
        'Parental_Involvement': parental_involvement,
        'Access_to_Resources': access_to_resources,
        'Extracurricular_Activities': extracurricular_activities,
        'Sleep_Hours': sleep_hours,
        'Previous_Scores': previous_scores,
        'Motivation_Level': motivation_level,
        'Internet_Access': internet_access,
        'Tutoring_Sessions': tutoring_sessions,
        'Family_Income': family_income,
        'Teacher_Quality': teacher_quality,
        'School_Type': school_type,
        'Peer_Influence': peer_influence,
        'Physical_Activity': physical_activity,
        'Learning_Disabilities': learning_disabilities,
        'Parental_Education_Level': parental_education_level,
        'Distance_from_Home': distance_from_home,
        'Gender': gender
    }
    
    features_df = pd.DataFrame(user_data, index=[0])
    return features_df

# Function to show score distribution histogram
def display_score_histogram(prediction):
    try:
        df = pd.read_csv('StudentPerformanceFactors.csv')
        scores = df['Exam_Score']
        
        fig = go.Figure()
        
        # Add histogram trace
        fig.add_trace(go.Histogram(
            x=scores,
            nbinsx=50,
            name='All Students',
            marker_color='lightblue',
            opacity=0.7,
            hovertemplate="Score: %{x}<br>Freq: %{y}<extra></extra>"
        ))
        
        # Add vertical line for predicted score
        fig.add_vline(
            x=prediction[0], 
            line=dict(color='red', width=2, dash='dash'),
            annotation_text=f"Your Score: {prediction[0]:.1f}",
            annotation_position="top right"
        )
        
        # Calculate percentile of predicted score
        percentile = sum(scores <= prediction[0]) / len(scores) * 100
        
        # Update layout
        fig.update_layout(
            title=f'Your Score Compared to All Students (Percentile: {percentile:.1f}%)',
            xaxis_title='Exam Score',
            yaxis_title='Number of Students',
            template='plotly_white',
            height=300,
            margin=dict(l=0, r=0, t=50, b=0),
            hovermode='closest'
        )
        
        # Display the plotly chart
        st.plotly_chart(fig, use_container_width=True)
        
        if percentile >= 90:
            st.success(f"Your predicted score is higher than {percentile:.1f}% of all students!")
        elif percentile >= 70:
            st.info(f"Your predicted score is higher than {percentile:.1f}% of all students.")
        elif percentile >= 50:
            st.info(f"Your predicted score is around average, higher than {percentile:.1f}% of all students.")
        else:
            st.warning(f"Your predicted score is lower than average, only higher than {percentile:.1f}% of all students.")
        
        return percentile
            
    except FileNotFoundError:
        # If original dataset isn't available, create a synthetic distribution
        mean_score = 70
        std_score = 15
        synthetic_scores = np.random.normal(mean_score, std_score, 1000)
        synthetic_scores = np.clip(synthetic_scores, 0, 100)
        
        fig = go.Figure()
        
        # Add histogram trace
        fig.add_trace(go.Histogram(
            x=synthetic_scores,
            nbinsx=20,
            name='Estimated Distribution',
            marker_color='lightblue',
            opacity=0.7,
            hovertemplate="Score: %{x}<br>Freq: %{y}<extra></extra>"
        ))
        
        # Add vertical line for predicted score
        fig.add_vline(
            x=prediction[0], 
            line=dict(color='red', width=2, dash='dash'),
            annotation_text=f"Your Score: {prediction[0]:.1f}",
            annotation_position="top right"
        )
        
        # Calculate percentile of predicted score
        percentile = sum(synthetic_scores <= prediction[0]) / len(synthetic_scores) * 100
        
        # Update layout
        fig.update_layout(
            title=f'Your Score Compared to Estimated Distribution (Est. Percentile: {percentile:.1f}%)',
            xaxis_title='Exam Score',
            yaxis_title='Number of Students',
            template='plotly_white',
            height=300,
            margin=dict(l=0, r=0, t=40, b=0),
            hovermode='closest'
        )
        
        # Display plotly chart
        st.plotly_chart(fig, use_container_width=True)
        
        st.info("Note: This is based on an estimated distribution since the full dataset wasn't available.")
        if percentile >= 90:
            st.success(f"Your predicted score is higher than approximately {percentile:.1f}% of all students!")
        elif percentile >= 70:
            st.info(f"Your predicted score is higher than approximately {percentile:.1f}% of all students.")
        elif percentile >= 50:
            st.info(f"Your predicted score is around average, higher than approximately {percentile:.1f}% of all students.")
        else:
            st.warning(f"Your predicted score is lower than average, only higher than approximately {percentile:.1f}% of all students.")
        
        return percentile

# Function to display recommendations
def display_recommendations(prediction, user_input):
    st.subheader("Performance Improvement Recommendations")
    
    if prediction[0] < 70:
        if user_input['Hours_Studied'].values[0] < 15:
            st.info("📚 **Increase Study Hours**: Consider increasing weekly study hours to at least 15-20 hours.")
        if user_input['Attendance'].values[0] < 90:
            st.info("📅 **Improve Attendance**: Aim for at least 90% attendance to avoid missing important lessons.")
        if user_input['Sleep_Hours'].values[0] < 7:
            st.info("😴 **Improve Sleep Schedule**: Work on getting 7-8 hours of sleep to improve cognitive function.")
        if user_input['Tutoring_Sessions'].values[0] < 4:
            st.info("🧑‍🏫 **Seek Additional Help**: Consider increasing tutoring sessions to at least 4 per month.")
        if user_input['Physical_Activity'].values[0] < 3:
            st.info("🏃 **Increase Physical Activity**: Regular exercise has been linked to better academic performance.")
        if user_input['Motivation_Level'].iloc[0] == "Low":
            st.info("🔥 **Boost Motivation**: Consider setting specific goals and finding study methods that work for you.")
    else:
        st.success("Great job! Keep up the good work.")
        if user_input['Extracurricular_Activities'].iloc[0] == "No":
            st.info("🏆 **Join Extracurricular Activities**: This can help develop soft skills and provide a balanced approach to education.")

# Function to display the sidebar
def display_sidebar():
    with st.sidebar:
        st.header("About")
        st.info("""
        This prediction model was built using Ridge Regression and trained on student performance data.
        The model analyzes various factors that influence academic performance and predicts exam scores.
        
        **Note**: This is a prediction tool and actual results may vary based on individual circumstances.
        """)
        
        st.header("Key Factors Influencing Predictions")
        st.markdown("""
        - **Hours Studied**: Time spent studying per week
        - **Previous Scores**: Past academic performance
        - **Attendance**: Regular class attendance
        - **Sleep Hours**: Adequate rest for cognitive function
        - **Physical Activity**: Regular exercise
        - **Tutoring Sessions**: Additional academic support
        - **Parental Involvement**: Level of family support
        - **Teacher Quality**: Quality of instruction
        - **Access to Resources**: Educational materials availability
        - **Motivation Level**: Student's drive to succeed
        """)
        
        st.header("How It Works")
        st.markdown("""
        This app uses a Ridge Regression model to predict student performance. 
        The model was trained on a dataset of student characteristics and their corresponding exam scores.
        
        For best results, provide accurate information for all fields.
        """)

# Main function
def main():
    # Page configuration
    st.set_page_config(page_title="Student Performance Predictor", layout="wide")
    st.title("Student Performance Prediction Tool")
    st.markdown("""
    This app predicts student exam scores based on various factors like study hours, 
    parental involvement, resources, and more. Enter the student's details below to get a prediction.
    """)
    
    # Load model and transformer
    model, transformer = load_model()
    
    # Display sidebar
    display_sidebar()
    
    # Get user input
    user_input = get_user_input()
    
    # Display inputed data
    with st.expander("View your input data"):
        st.write(user_input)
    
    # Prediction button
    if st.button("Predict Exam Score"):
        user_input_transformed = transformer.transform(user_input)
        prediction = model.predict(user_input_transformed)
        
        # Display the prediction
        st.subheader("Prediction Results")
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Predicted Exam Score", f"{prediction[0]:.2f}/100")
            
            if prediction[0] >= 90:
                performance = "Excellent"
                color = "green"
            elif prediction[0] >= 80:
                performance = "Very Good"
                color = "lightgreen"
            elif prediction[0] >= 70:
                performance = "Good"
                color = "blue"
            elif prediction[0] >= 60:
                performance = "Satisfactory"
                color = "orange"
            else:
                performance = "Needs Improvement"
                color = "red"
                
            st.markdown(f"<h3 style='color:{color}'>Performance Category: {performance}</h3>", unsafe_allow_html=True)
        
        with col2:
            percentile = display_score_histogram(prediction)
        
        # Display recommendations
        display_recommendations(prediction, user_input)

if __name__ == "__main__":
    main()