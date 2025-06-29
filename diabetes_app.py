import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn import svm
import joblib
import google.generativeai as genai
import os
from datetime import datetime

# Configure Gemini API
GEMINI_API_KEY = "AIzaSyAE5035UsAf9xznrmWy3RTmsBYQKKsnYMc"
genai.configure(api_key=GEMINI_API_KEY)

# Page configuration
st.set_page_config(
    page_title="Diabetes Health Assistant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': "Diabetes Health Assistant - ML Prediction + AI Chatbot"
    }
)

# Custom CSS for better styling
st.markdown("""
<style>
    /* Main title styling */
    .main-title {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 1rem;
    }
    
    /* Card-like containers */
    .stApp > div > div > div > div > div > div {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        border: 1px solid #e9ecef;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.5rem 2rem;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin: 0.5rem 0;
    }
    
    /* Success and error styling */
    .success-box {
        background: linear-gradient(135deg, #56ab2f 0%, #a8e6cf 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #28a745;
    }
    
    .error-box {
        background: linear-gradient(135deg, #ff6b6b 0%, #ffa07a 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #dc3545;
    }
    
    /* Chat message styling */
    .chat-message {
        background: #f8f9fa;
        border-radius: 15px;
        padding: 1rem;
        margin: 0.5rem 0;
        border: 1px solid #e9ecef;
    }
    
    /* Input field styling */
    .stNumberInput > div > div > input {
        border-radius: 10px;
        border: 2px solid #e9ecef;
    }
    
    .stNumberInput > div > div > input:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 0.2rem rgba(102, 126, 234, 0.25);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
    }
    
    /* Header styling */
    h1, h2, h3 {
        color: #2c3e50;
        font-weight: bold;
    }
    
    /* Dataframe styling */
    .dataframe {
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Divider styling */
    hr {
        border: none;
        height: 2px;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        margin: 2rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Gemini model
@st.cache_resource
def get_gemini_model():
    try:
        model = genai.GenerativeModel('gemini-2.5-flash')
        return model
    except Exception as e:
        st.error(f"Error initializing Gemini model: {e}")
        return None

# Diabetes chatbot function
def get_diabetes_response(user_question, chat_history):
    model = get_gemini_model()
    if not model:
        return "Sorry, I'm having trouble connecting to my knowledge base. Please try again later."
    
    # Create a comprehensive prompt for diabetes-related questions
    system_prompt = """You are a knowledgeable and empathetic diabetes health assistant. Your role is to provide accurate, helpful, and supportive information about diabetes. 

IMPORTANT GUIDELINES:
1. Always provide evidence-based medical information
2. Be clear that you're not a replacement for professional medical advice
3. Encourage users to consult healthcare professionals for specific medical decisions
4. Use simple, understandable language
5. Be supportive and encouraging
6. Focus on diabetes-related topics (diabetes types, symptoms, management, prevention, lifestyle, etc.)

When responding:
- Be informative but concise
- Use bullet points when helpful
- Include practical tips when relevant
- Always remind users to consult healthcare professionals for medical decisions
- Stay focused on diabetes-related topics

Previous conversation context:
{chat_history}

User's current question: {user_question}

Please provide a helpful, accurate, and supportive response about diabetes."""
    
    try:
        # Format chat history for context
        formatted_history = ""
        if chat_history:
            formatted_history = "\n".join([f"User: {msg['user']}\nAssistant: {msg['assistant']}" for msg in chat_history[-5:]])  # Last 5 messages for context
        
        prompt = system_prompt.format(
            chat_history=formatted_history,
            user_question=user_question
        )
        
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"I apologize, but I'm experiencing technical difficulties. Please try again. Error: {str(e)}"

# Load the diabetes prediction model
@st.cache_resource
def load_model():
    try:
        # Try to load existing model and scaler
        model = joblib.load('diabetes_model.pkl')
        scaler = joblib.load('diabetes_scaler.pkl')
        return model, scaler
    except:
        # If no saved model exists, train a new one
        return train_new_model()

def train_new_model():
    # Load and prepare data
    diab = pd.read_csv("diabetes.csv")
    X = diab.drop(columns='Outcome', axis=1)
    Y = diab['Outcome']
    
    # Standardize the data
    scaler = StandardScaler()
    scaler.fit(X)
    X_scaled = scaler.transform(X)
    
    # Train test split
    from sklearn.model_selection import train_test_split
    X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.2, stratify=Y, random_state=2)
    
    # Create and train the SVM classifier
    classifier = svm.SVC(kernel='linear', probability=True)
    classifier.fit(X_train, Y_train)
    
    # Save the model and scaler
    joblib.dump(classifier, 'diabetes_model.pkl')
    joblib.dump(scaler, 'diabetes_scaler.pkl')
    
    return classifier, scaler

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'current_page' not in st.session_state:
    st.session_state.current_page = "Diabetes Prediction"

# Sidebar navigation
st.sidebar.title("🏥 Diabetes Health Assistant")
st.sidebar.markdown("---")

# Page selection
page = st.sidebar.radio(
    "Choose a page:",
    ["Diabetes Prediction", "Diabetes Chatbot"],
    index=0 if st.session_state.current_page == "Diabetes Prediction" else 1
)

st.session_state.current_page = page

# Main content based on page selection
if page == "Diabetes Prediction":
    st.markdown('<h1 class="main-title">🏥 Diabetes Prediction System</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Create an attractive info card
    with st.container():
        st.markdown("""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 1.5rem; border-radius: 15px; margin: 1rem 0;">
            <h3 style="color: white; margin-bottom: 1rem;">🔬 AI-Powered Diabetes Risk Assessment</h3>
            <p style="margin: 0; font-size: 1.1rem;">This application uses advanced Machine Learning to predict diabetes risk based on your medical parameters. 
            Enter your details below to get an instant, accurate prediction with confidence scores.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Load model
    model, scaler = load_model()
    
    # Create main page layout for input
    st.markdown("---")
    st.markdown('<h2 style="color: #2c3e50; text-align: center;">📊 Patient Information</h2>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #6c757d; font-size: 1.1rem;">Enter the patient\'s medical parameters below</p>', unsafe_allow_html=True)
    
    # Create two columns for better layout
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div style="background: #f8f9fa; padding: 1.5rem; border-radius: 15px; border: 1px solid #e9ecef;">', unsafe_allow_html=True)
        st.markdown('<h4 style="color: #2c3e50; margin-bottom: 1rem;">📋 Basic Information</h4>', unsafe_allow_html=True)
        pregnancies = st.number_input("Number of Pregnancies", min_value=0, max_value=20, value=0)
        glucose = st.number_input("Glucose Level (mg/dL)", min_value=0, max_value=300, value=120)
        blood_pressure = st.number_input("Blood Pressure (mm Hg)", min_value=0, max_value=200, value=80)
        skin_thickness = st.number_input("Skin Thickness (mm)", min_value=0, max_value=100, value=20)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div style="background: #f8f9fa; padding: 1.5rem; border-radius: 15px; border: 1px solid #e9ecef;">', unsafe_allow_html=True)
        st.markdown('<h4 style="color: #2c3e50; margin-bottom: 1rem;">🔬 Medical Parameters</h4>', unsafe_allow_html=True)
        insulin = st.number_input("Insulin Level (mu U/ml)", min_value=0, max_value=1000, value=80)
        bmi = st.number_input("BMI (kg/m²)", min_value=10.0, max_value=70.0, value=25.0, step=0.1)
        diabetes_pedigree = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5, step=0.01)
        age = st.number_input("Age (years)", min_value=1, max_value=120, value=30)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Prediction button in center
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🔍 Predict Diabetes Risk", type="primary", use_container_width=True):
            # Create input array
            input_data = np.array([pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree, age])
            input_reshaped = input_data.reshape(1, -1)
            
            # Standardize input
            input_scaled = scaler.transform(input_reshaped)
            
            # Make prediction
            prediction = model.predict(input_scaled)[0]
            probability = model.predict_proba(input_scaled)[0]
            
            # Display results with enhanced styling
            st.markdown("---")
            st.markdown('<h2 style="color: #2c3e50; text-align: center;">📋 Prediction Results</h2>', unsafe_allow_html=True)
            
            # Create columns for better layout
            result_col1, result_col2 = st.columns(2)
            
            with result_col1:
                if prediction == 0:
                    st.markdown("""
                    <div class="success-box">
                        <h3 style="color: white; margin-bottom: 0.5rem;">🎉 NON-DIABETIC</h3>
                        <p style="color: white; margin: 0;">Based on the provided parameters, the model predicts that the patient is <strong>NOT DIABETIC</strong>.</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="error-box">
                        <h3 style="color: white; margin-bottom: 0.5rem;">⚠️ DIABETIC</h3>
                        <p style="color: white; margin: 0;">Based on the provided parameters, the model predicts that the patient <strong>HAS DIABETES</strong>.</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            with result_col2:
                st.markdown("""
                <div class="metric-card">
                    <h4 style="color: white; margin-bottom: 0.5rem;">Confidence Score</h4>
                    <h2 style="color: white; margin: 0;">{:.1f}%</h2>
                </div>
                """.format(max(probability)*100), unsafe_allow_html=True)
                
                st.markdown("""
                <div class="metric-card">
                    <h4 style="color: white; margin-bottom: 0.5rem;">Risk Level</h4>
                    <h2 style="color: white; margin: 0;">{}</h2>
                </div>
                """.format("HIGH" if prediction == 1 else "LOW"), unsafe_allow_html=True)
            
            # Show probability breakdown
            st.markdown("---")
            st.markdown('<h3 style="color: #2c3e50; text-align: center;">📊 Probability Breakdown</h3>', unsafe_allow_html=True)
            prob_col1, prob_col2 = st.columns(2)
            
            with prob_col1:
                st.markdown("""
                <div style="background: linear-gradient(135deg, #56ab2f 0%, #a8e6cf 100%); color: white; padding: 1.5rem; border-radius: 15px; text-align: center;">
                    <h4 style="color: white; margin-bottom: 0.5rem;">Non-Diabetic Probability</h4>
                    <h2 style="color: white; margin: 0;">{:.1f}%</h2>
                </div>
                """.format(probability[0]*100), unsafe_allow_html=True)
            
            with prob_col2:
                st.markdown("""
                <div style="background: linear-gradient(135deg, #ff6b6b 0%, #ffa07a 100%); color: white; padding: 1.5rem; border-radius: 15px; text-align: center;">
                    <h4 style="color: white; margin-bottom: 0.5rem;">Diabetic Probability</h4>
                    <h2 style="color: white; margin: 0;">{:.1f}%</h2>
                </div>
                """.format(probability[1]*100), unsafe_allow_html=True)
            
            # Show input summary
            st.markdown("---")
            st.markdown('<h3 style="color: #2c3e50; text-align: center;">📝 Input Summary</h3>', unsafe_allow_html=True)
            input_summary = pd.DataFrame({
                'Parameter': ['Pregnancies', 'Glucose', 'Blood Pressure', 'Skin Thickness', 
                             'Insulin', 'BMI', 'Diabetes Pedigree', 'Age'],
                'Value': [pregnancies, glucose, blood_pressure, skin_thickness, 
                         insulin, bmi, diabetes_pedigree, age]
            })
            st.dataframe(input_summary, use_container_width=True)
    
    # Information section
    st.markdown("---")
    st.markdown('<h2 style="color: #2c3e50; text-align: center;">ℹ️ About This Model</h2>', unsafe_allow_html=True)
    
    with st.container():
        st.markdown("""
        <div style="background: #f8f9fa; padding: 2rem; border-radius: 15px; border: 1px solid #e9ecef;">
            <h4 style="color: #2c3e50; margin-bottom: 1rem;">🔬 Machine Learning Model Details</h4>
            <p style="color: #6c757d; line-height: 1.6;">This diabetes prediction model uses <strong>Support Vector Machine (SVM)</strong> with the following features:</p>
            <ul style="color: #6c757d; line-height: 1.6;">
                <li><strong>Pregnancies:</strong> Number of times pregnant</li>
                <li><strong>Glucose:</strong> Plasma glucose concentration (mg/dL)</li>
                <li><strong>Blood Pressure:</strong> Diastolic blood pressure (mm Hg)</li>
                <li><strong>Skin Thickness:</strong> Triceps skin fold thickness (mm)</li>
                <li><strong>Insulin:</strong> 2-Hour serum insulin (mu U/ml)</li>
                <li><strong>BMI:</strong> Body mass index (kg/m²)</li>
                <li><strong>Diabetes Pedigree Function:</strong> Diabetes family history</li>
                <li><strong>Age:</strong> Age in years</li>
            </ul>
            <div style="background: #fff3cd; border: 1px solid #ffeaa7; border-radius: 10px; padding: 1rem; margin-top: 1rem;">
                <p style="color: #856404; margin: 0;"><strong>⚠️ Important:</strong> This is a machine learning model for educational purposes only. Always consult with healthcare professionals for medical decisions.</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

elif page == "Diabetes Chatbot":
    st.markdown('<h1 class="main-title">💬 Diabetes Health Assistant</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Create an attractive welcome card
    with st.container():
        st.markdown("""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 1.5rem; border-radius: 15px; margin: 1rem 0;">
            <h3 style="color: white; margin-bottom: 1rem;">🤖 AI-Powered Diabetes Support</h3>
            <p style="margin: 0; font-size: 1.1rem;">Welcome to your personal diabetes health assistant! I'm here to help you with questions about diabetes, 
            including types, symptoms, management, prevention, and lifestyle tips.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Chat interface
    st.markdown('<h3 style="color: #2c3e50; text-align: center;">💭 Chat with your Diabetes Assistant</h3>', unsafe_allow_html=True)
    
    # Display chat history with enhanced memory (last 7 messages)
    chat_container = st.container()
    
    with chat_container:
        # Show only the last 7 messages for better memory management
        recent_messages = st.session_state.chat_history[-7:] if len(st.session_state.chat_history) > 7 else st.session_state.chat_history
        for message in recent_messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask me anything about diabetes..."):
        # Add user message to chat
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get bot response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # Prepare chat history for context (last 6 messages for better memory)
                chat_context = []
                recent_history = st.session_state.chat_history[-6:] if len(st.session_state.chat_history) > 6 else st.session_state.chat_history
                
                for i in range(0, len(recent_history) - 1, 2):  # Process pairs of messages
                    if i + 1 < len(recent_history):
                        user_msg = recent_history[i]["content"] if recent_history[i]["role"] == "user" else recent_history[i+1]["content"]
                        assistant_msg = recent_history[i+1]["content"] if recent_history[i+1]["role"] == "assistant" else recent_history[i]["content"]
                        chat_context.append({"user": user_msg, "assistant": assistant_msg})
                
                response = get_diabetes_response(prompt, chat_context)
                st.markdown(response)
        
        # Add assistant response to chat history
        st.session_state.chat_history.append({"role": "assistant", "content": response})
    
    # Suggested questions
    st.markdown("---")
    st.markdown('<h3 style="color: #2c3e50; text-align: center;">💡 Suggested Questions</h3>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("What are the symptoms of diabetes?", use_container_width=True):
            st.session_state.chat_history.append({"role": "user", "content": "What are the symptoms of diabetes?"})
            with st.chat_message("user"):
                st.markdown("What are the symptoms of diabetes?")
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = get_diabetes_response("What are the symptoms of diabetes?", [])
                    st.markdown(response)
            st.session_state.chat_history.append({"role": "assistant", "content": response})
            st.rerun()
        
        if st.button("How can I prevent diabetes?", use_container_width=True):
            st.session_state.chat_history.append({"role": "user", "content": "How can I prevent diabetes?"})
            with st.chat_message("user"):
                st.markdown("How can I prevent diabetes?")
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = get_diabetes_response("How can I prevent diabetes?", [])
                    st.markdown(response)
            st.session_state.chat_history.append({"role": "assistant", "content": response})
            st.rerun()
    
    with col2:
        if st.button("What's the difference between Type 1 and Type 2?", use_container_width=True):
            st.session_state.chat_history.append({"role": "user", "content": "What's the difference between Type 1 and Type 2 diabetes?"})
            with st.chat_message("user"):
                st.markdown("What's the difference between Type 1 and Type 2 diabetes?")
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = get_diabetes_response("What's the difference between Type 1 and Type 2 diabetes?", [])
                    st.markdown(response)
            st.session_state.chat_history.append({"role": "assistant", "content": response})
            st.rerun()
        
        if st.button("What should I eat if I have diabetes?", use_container_width=True):
            st.session_state.chat_history.append({"role": "user", "content": "What should I eat if I have diabetes?"})
            with st.chat_message("user"):
                st.markdown("What should I eat if I have diabetes?")
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = get_diabetes_response("What should I eat if I have diabetes?", [])
                    st.markdown(response)
            st.session_state.chat_history.append({"role": "assistant", "content": response})
            st.rerun()
    
    # Clear chat button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>🏥 Diabetes Health Assistant | Built with ❤️ using Streamlit and Gemini AI</p>
    <p><small>⚠️ This application is for educational purposes only. Always consult healthcare professionals for medical decisions.</small></p>
</div>
""", unsafe_allow_html=True) 