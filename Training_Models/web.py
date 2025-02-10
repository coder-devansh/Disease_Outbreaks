import os
import pickle  # pre-trained model loading
import streamlit as st  # web app
from streamlit_option_menu import option_menu

# Set the page configuration
st.set_page_config(page_title='Prediction of Disease Outbreaks',
                   layout='wide',
                   page_icon="🧑‍⚕️")

# Load pre-trained models
diabetes_model = pickle.load(open(r"D:\Interactive Dashboard\Training_Models\diabetes_model.sav", 'rb'))
heart_disease_model = pickle.load(open(r"D:\Interactive Dashboard\Training_Models\heart_model.sav", 'rb'))
parkinsons_model = pickle.load(open(r"D:\Interactive Dashboard\Training_Models\parkinson_model.sav", 'rb'))

# Sidebar navigation
with st.sidebar:
    selected = option_menu('Prediction of Disease Outbreak System',
                           ['Diabetes Prediction', 'Heart Disease Prediction', 'Parkinsons Prediction'],
                           menu_icon='hospital-fill', icons=['activity', 'heart', 'person'], default_index=0)

# Diabetes Prediction section
if selected == 'Diabetes Prediction':
    st.title('Diabetes Prediction using ML')

    # Create input fields for the user
    col1, col2, col3 = st.columns(3)
    
    with col1:
        Pregnancies = st.text_input('Number of Pregnancies', '0')
    with col2:
        Glucose = st.text_input('Glucose Level', '0')
    with col3:
        Bloodpressure = st.text_input('Blood Pressure Value', '0')
    
    with col1:
        SkinThickness = st.text_input('Skin Thickness Value', '0')
    with col2:
        Insulin = st.text_input('Insulin Level', '0')
    with col3:
        BMI = st.text_input('BMI Value', '0')
    
    with col1:
        DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function Value', '0')
    with col2:
        Age = st.text_input('Age of the Person', '0')

    # Check if the button is pressed to make a prediction
    diab_diagnosis = ''
    if st.button('Diabetes Test Result'):
        try:
            # Convert inputs to float for model prediction
            user_input = [float(Pregnancies), float(Glucose), float(Bloodpressure), float(SkinThickness),
                          float(Insulin), float(BMI), float(DiabetesPedigreeFunction), float(Age)]

            # Predict diabetes using the pre-trained model
            diab_prediction = diabetes_model.predict([user_input])
            
            # Output result
            if diab_prediction[0] == 1:
                diab_diagnosis = 'The person is diabetic'
            else:
                diab_diagnosis = 'The person is not diabetic'
        except ValueError:
            diab_diagnosis = 'Please enter valid numeric values for all inputs.'

    # Display the result
    st.success(diab_diagnosis)

# Heart Disease Prediction section
if selected == 'Heart Disease Prediction':
    st.title('Heart Disease Prediction using ML')

    # Create input fields for the user
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.text_input('Age', '0')
    with col2:
        sex = st.selectbox('Sex', ['Male', 'Female'])
    with col3:
        cp = st.selectbox('Chest Pain Type', ['0', '1', '2', '3'])
    
    with col1:
        trestbps = st.text_input('Resting Blood Pressure', '0')
    with col2:
        chol = st.text_input('Serum Cholestoral', '0')
    with col3:
        fbs = st.selectbox('Fasting Blood Sugar', ['0', '1'])
    
    with col1:
        restecg = st.selectbox('Resting Electrocardiographic Results', ['0', '1', '2'])
    with col2:
        thalach = st.text_input('Maximum Heart Rate Achieved', '0')
    with col3:
        exang = st.selectbox('Exercise Induced Angina', ['0', '1'])
    
    with col1:
        oldpeak = st.text_input('Depression Induced by Exercise', '0')
    with col2:
        slope = st.selectbox('Slope of the Peak Exercise ST Segment', ['0', '1', '2'])
    with col3:
        ca = st.selectbox('Number of Major Vessels Colored by Fluoroscopy', ['0', '1', '2', '3'])
    
    with col1:
        thal = st.selectbox('Thalassemia', ['3', '6', '7'])

    # Check if the button is pressed to make a prediction
    heart_diagnosis = ''
    if st.button('Heart Disease Test Result'):
        try:
            # Convert inputs to float for model prediction
            user_input = [float(age), 1 if sex == 'Male' else 0, float(cp), float(trestbps), float(chol),
                          float(fbs), float(restecg), float(thalach), float(exang), float(oldpeak),
                          float(slope), float(ca), float(thal)]

            # Predict heart disease using the pre-trained model
            heart_prediction = heart_disease_model.predict([user_input])

            # Output result
            if heart_prediction[0] == 1:
                heart_diagnosis = 'The person has heart disease'
            else:
                heart_diagnosis = 'The person does not have heart disease'
        except ValueError:
            heart_diagnosis = 'Please enter valid numeric values for all inputs.'

    # Display the result
    st.success(heart_diagnosis)

# Parkinson's Prediction section
if selected == 'Parkinsons Prediction':
    st.title('Parkinsons Disease Prediction using ML')

    # Create input fields for the user
    col1, col2, col3 = st.columns(3)

    with col1:
        feature_1 = st.text_input('Feature 1', '0')
    with col2:
        feature_2 = st.text_input('Feature 2', '0')
    with col3:
        feature_3 = st.text_input('Feature 3', '0')
    
    with col1:
        feature_4 = st.text_input('Feature 4', '0')
    with col2:
        feature_5 = st.text_input('Feature 5', '0')
    with col3:
        feature_6 = st.text_input('Feature 6', '0')
    
    with col1:
        feature_7 = st.text_input('Feature 7', '0')
    with col2:
        feature_8 = st.text_input('Feature 8', '0')
    with col3:
        feature_9 = st.text_input('Feature 9', '0')

    # Check if the button is pressed to make a prediction
    parkinsons_diagnosis = ''
    if st.button('Parkinsons Test Result'):
        try:
            # Convert inputs to float for model prediction
            user_input = [float(feature_1), float(feature_2), float(feature_3), float(feature_4),
                          float(feature_5), float(feature_6), float(feature_7), float(feature_8), float(feature_9)]

            # Predict Parkinson's disease using the pre-trained model
            parkinsons_prediction = parkinsons_model.predict([user_input])

            # Output result
            if parkinsons_prediction[0] == 1:
                parkinsons_diagnosis = 'The person has Parkinson\'s disease'
            else:
                parkinsons_diagnosis = 'The person does not have Parkinson\'s disease'
        except ValueError:
            parkinsons_diagnosis = 'Please enter valid numeric values for all inputs.'

    # Display the result
    st.success(parkinsons_diagnosis)
