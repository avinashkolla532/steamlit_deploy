# -*- coding: utf-8 -*-
"""
Created on Sat Jun  1 13:31:50 2024

@author: avina
"""

import pickle
from streamlit_option_menu import option_menu
import streamlit as st
import numpy as np

""" load the datasets """
diabetic_pred = pickle.load(open('diabetes_model.sav','rb'))
heart_pred = pickle.load(open('hear_disease_data.sav','rb'))
parkisons_pred = pickle.load(open('Parkinsons_disease.sav','rb'))         

with st.sidebar:
     selected = option_menu('Multiple Disease Prediction System',
                          
                          ['Diabetes Prediction',
                           'Heart Disease Prediction',
                           'Parkinsons Prediction'],
                          icons=['activity','heart','person'],
                          default_index=0)
if(selected == 'Diabetes Prediction'):
    st.title('Diabetes Prediction using ml')
    col1, col2, col3 = st.columns(3)
    
    with col1:
        Pregnancies = st.text_input('Number of Pregnancies')
        
    with col2:
        Glucose = st.text_input('Glucose Level')
    
    with col3:
        BloodPressure = st.text_input('Blood Pressure value')
    
    with col1:
        SkinThickness = st.text_input('Skin Thickness value')
    
    with col2:
        Insulin = st.text_input('Insulin Level')
    
    with col3:
        BMI = st.text_input('BMI value')
    
    with col1:
        DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value')
    
    with col2:
        Age = st.text_input('Age of the Person')
    
    
    # code for Prediction
    diab_diagnosis = ''
    
    # creating a button for Prediction
    
    if st.button('Diabetes Test Result'):
        diab_prediction = diabetic_pred.predict([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])
        
        if (diab_prediction[0] == 1):
          diab_diagnosis = 'The person is diabetic'
        else:
          diab_diagnosis = 'The person is not diabetic'
        
    st.success(diab_diagnosis)
    
if (selected == 'Heart Disease Prediction'):
    
    # Page title
    st.title('Heart Disease Prediction using ML')
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.text_input('Age')
        
    with col2:
        sex = st.text_input('Sex')
        
    with col3:
        cp = st.text_input('Chest Pain types')
        
    with col1:
        trestbps = st.text_input('Resting Blood Pressure')
        
    with col2:
        chol = st.text_input('Serum Cholestoral in mg/dl')
        
    with col3:
        fbs = st.text_input('Fasting Blood Sugar > 120 mg/dl')
        
    with col1:
        restecg = st.text_input('Resting Electrocardiographic results')
        
    with col2:
        thalach = st.text_input('Maximum Heart Rate achieved')
        
    with col3:
        exang = st.text_input('Exercise Induced Angina')
        
    with col1:
        oldpeak = st.text_input('ST depression induced by exercise')
        
    with col2:
        slope = st.text_input('Slope of the peak exercise ST segment')
        
    with col3:
        ca = st.text_input('Major vessels colored by flourosopy')
        
    with col1:
        thal = st.text_input('thal: 0 = normal; 1 = fixed defect; 2 = reversible defect')
        
    # Code for Prediction
    heart_diagnosis = ''
    
    # Creating a button for Prediction
    if st.button('Heart Disease Test Result'):
        try:
            # Convert inputs to appropriate types
            age = int(age)
            sex = int(sex)
            cp = int(cp)
            trestbps = float(trestbps)
            chol = float(chol)
            fbs = int(fbs)
            restecg = int(restecg)
            thalach = float(thalach)
            exang = int(exang)
            oldpeak = float(oldpeak)
            slope = int(slope)
            ca = int(ca)
            thal = int(thal)
            
            # Prepare input data for prediction
            input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
            
            # Make prediction
            heart_prediction = heart_pred.predict(input_data)
            
            if (heart_prediction[0] == 1):
                heart_diagnosis = 'The person is having heart disease'
            else:
                heart_diagnosis = 'The person does not have any heart disease'
                
        except ValueError as e:
            heart_diagnosis = f"Error in input data: {e}"
        
    st.success(heart_diagnosis)
       
if(selected == 'Parkinsons Prediction'):
    st.title('Parkisons preediction using ML')
    
 
    

    col1, col2, col3, col4, col5 = st.columns(5)  

    with col1: 
        fo = st.text_input('MDVP:Fo(Hz)')
    
    with col2:
        fhi = st.text_input('MDVP:Fhi(Hz)')
    
    with col3:
        flo = st.text_input('MDVP:Flo(Hz)')
    
    with col4:
        Jitter_percent = st.text_input('MDVP:Jitter(%)')
    
    with col5:
        Jitter_Abs = st.text_input('MDVP:Jitter(Abs)')
    
    with col1:
        RAP = st.text_input('MDVP:RAP')
    
    with col2:
        PPQ = st.text_input('MDVP:PPQ')
    
    with col3:
        DDP = st.text_input('Jitter:DDP')
    
    with col4:
        Shimmer = st.text_input('MDVP:Shimmer')
    
    with col5:
        Shimmer_dB = st.text_input('MDVP:Shimmer(dB)')
    
    with col1:
        APQ3 = st.text_input('Shimmer:APQ3')
    
    with col2:
        APQ5 = st.text_input('Shimmer:APQ5')
    
    with col3:
        APQ = st.text_input('MDVP:APQ')
    
    with col4:
        DDA = st.text_input('Shimmer:DDA')
    
    with col5:
        NHR = st.text_input('NHR')
    
    with col1:
        HNR = st.text_input('HNR')
    
    with col2:
        RPDE = st.text_input('RPDE')
    
    with col3:
        DFA = st.text_input('DFA')
    
    with col4:
        spread1 = st.text_input('spread1')
    
    with col5:
        spread2 = st.text_input('spread2')
    
    with col1:
        D2 = st.text_input('D2')
    
    with col2:
        PPE = st.text_input('PPE')
    


# code for Prediction
    parkinsons_diagnosis = ''
    
    if st.button("Parkinson's Test Result"):
        try:
            # Convert inputs to appropriate types
            fo = float(fo.replace(',', ''))  # Remove commas
            fhi = float(fhi.replace(',', ''))  # Remove commas
            flo = float(flo.replace(',', ''))  # Remove commas
            Jitter_percent = float(Jitter_percent.replace(',', ''))  # Remove commas
            Jitter_Abs = float(Jitter_Abs.replace(',', ''))  # Remove commas
            RAP = float(RAP.replace(',', ''))  # Remove commas
            PPQ = float(PPQ.replace(',', ''))  # Remove commas
            DDP = float(DDP.replace(',', ''))  # Remove commas
            Shimmer = float(Shimmer.replace(',', ''))  # Remove commas
            Shimmer_dB = float(Shimmer_dB.replace(',', ''))  # Remove commas
            APQ3 = float(APQ3.replace(',', ''))  # Remove commas
            APQ5 = float(APQ5.replace(',', ''))  # Remove commas
            APQ = float(APQ.replace(',', ''))  # Remove commas
            DDA = float(DDA.replace(',', ''))  # Remove commas
            NHR = float(NHR.replace(',', ''))  # Remove commas
            HNR = float(HNR.replace(',', ''))  # Remove commas
            RPDE = float(RPDE.replace(',', ''))  # Remove commas
            DFA = float(DFA.replace(',', ''))  # Remove commas
            spread1 = float(spread1.replace(',', ''))  # Remove commas
            spread2 = float(spread2.replace(',', ''))  # Remove commas
            D2 = float(D2.replace(',', ''))  # Remove commas
            PPE = float(PPE.replace(',', ''))  # Remove commas
            input_data = np.array([[fo, fhi, flo, Jitter_percent, Jitter_Abs, RAP, PPQ, DDP, Shimmer, Shimmer_dB, APQ3, APQ5, APQ, DDA, NHR, HNR, RPDE, DFA, spread1, spread2, D2, PPE]])
            
            # Make prediction
            parkinsons_prediction = parkisons_pred.predict(input_data)
            
            if parkinsons_prediction[0] == 0:
                parkinsons_diagnosis = "The person has Parkinson's disease"
            else:
                parkinsons_diagnosis = "The person does not have Parkinson's disease"
                
        except ValueError as e:
            parkinsons_diagnosis = f"Error in input data: {e}"
    
    st.success(parkinsons_diagnosis)


     