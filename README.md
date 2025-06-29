# Diabetes Health Assistant

A comprehensive web application that combines Machine Learning-based diabetes prediction with an AI-powered diabetes chatbot for complete diabetes health support.

## Features

### Diabetes Prediction System
- ML-Powered Prediction using Support Vector Machine (SVM)
- Real-time analysis with instant predictions and confidence scores
- Comprehensive input with 8 medical parameters for thorough assessment
- Visual results with clear indicators and probability breakdown
- Professional medical-themed user interface

### Diabetes Health Assistant Chatbot
- AI-Powered responses using Google's Gemini AI
- Comprehensive knowledge covering diabetes types, symptoms, management, and prevention
- Contextual conversations with chat history maintenance
- Suggested questions for quick access to common diabetes queries
- Professional medical disclaimers throughout

## Quick Start

### Installation

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the application:
   ```bash
   streamlit run diabetes_app.py
   ```

3. Open in browser:
   ```
   http://localhost:8501
   ```

## Detailed Features

### Diabetes Prediction Page
Input Parameters:
- Pregnancies (0-20)
- Glucose Level (0-300 mg/dL)
- Blood Pressure (0-200 mm Hg)
- Skin Thickness (0-100 mm)
- Insulin Level (0-1000 mu U/ml)
- BMI (10-70 kg/m²)
- Diabetes Pedigree Function (0-3)
- Age (1-120 years)

Output Features:
- Binary prediction (Diabetic/Non-Diabetic)
- Confidence score percentage
- Risk level assessment
- Probability breakdown for both outcomes
- Input parameter summary table

### Diabetes Chatbot Page
AI Capabilities:
- Evidence-based medical information
- Supportive and empathetic responses
- Practical lifestyle tips
- Prevention strategies
- Management advice

Chat Features:
- Real-time conversation
- Chat history preservation
- Suggested question buttons
- Clear chat functionality
- Professional medical disclaimers

## Technical Architecture

### Backend Technologies
- Streamlit: Web application framework
- Scikit-learn: Machine learning library (SVM classifier)
- Google Generative AI: Gemini Pro model for chatbot
- Pandas & NumPy: Data processing
- Joblib: Model serialization

### AI Integration
- Gemini API: Google's advanced language model
- Prompt Engineering: Specialized prompts for diabetes-related queries
- Context Management: Maintains conversation history
- Error Handling: Graceful fallbacks for API issues

## File Structure

```
├── diabetes_app.py              # Main multi-page Streamlit application
├── diabetes_prediction.py       # Standalone prediction script
├── diabetes.csv                 # Diabetes dataset
├── requirements.txt             # Python dependencies
├── README_MultiPage.md          # This documentation
├── diabetes_model.pkl           # Saved ML model (auto-generated)
└── diabetes_scaler.pkl          # Saved data scaler (auto-generated)
```

## Use Cases

### For Healthcare Professionals
- Quick Assessment Tool for rapid diabetes risk evaluation
- Patient Education with AI-powered diabetes information
- Screening Support for preliminary risk assessment

### For Patients
- Self-Assessment for personal diabetes risk evaluation
- Health Education with comprehensive diabetes information
- Lifestyle Guidance with practical tips and advice

### For Researchers
- Model Validation for testing ML predictions
- Data Analysis for exploring diabetes patterns
- AI Integration studies for healthcare applications

## Configuration

### API Key Setup
The Gemini API key is currently hardcoded in the application. For production use, consider:
- Using environment variables
- Implementing secure key management
- Adding API key validation

### Model Customization
- SVM Parameters: Modify kernel, C, gamma values
- Feature Engineering: Add/remove input parameters
- Threshold Adjustment: Change prediction thresholds

## Important Disclaimers

Medical Disclaimer:
- This application is for educational purposes only
- Not a replacement for professional medical advice
- Always consult healthcare professionals for medical decisions
- Results should not be used for self-diagnosis

AI Limitations:
- AI responses are general information only
- May not cover all individual cases
- Should not replace professional medical consultation

## Troubleshooting

### Common Issues

1. Gemini API Errors:
   - Check API key validity
   - Ensure internet connection
   - Verify google-generativeai package installation

2. Model Loading Issues:
   - App automatically trains new model if files missing
   - Ensure diabetes.csv is in correct location
   - Check file permissions

3. Streamlit Issues:
   ```bash
   # Port conflicts
   streamlit run diabetes_app.py --server.port 8502
   
   # Dependencies
   pip install --upgrade streamlit
   ```

### Performance Optimization
- Model Caching: Uses Streamlit's caching for better performance
- API Rate Limiting: Implements proper error handling
- Memory Management: Efficient data processing

## Future Enhancements

### Planned Features
- Multi-language Support for international diabetes information
- Voice Interface for speech-to-text accessibility
- Mobile App for native mobile application
- Integration with EHR systems
- Advanced Analytics for detailed health insights

### Technical Improvements
- Model Ensemble with multiple ML algorithms
- Real-time Updates with live model retraining
- API Security with enhanced authentication
- Performance optimization for response times

## Support

For technical support or questions:
1. Check the troubleshooting section
2. Verify all dependencies are installed
3. Ensure proper file structure
4. Test with minimal installation approach

## License

This project is for educational purposes. Please ensure compliance with:
- Medical device regulations (if applicable)
- Data privacy laws
- API usage terms and conditions

---

Built with Streamlit, Scikit-learn, and Google Gemini AI 
