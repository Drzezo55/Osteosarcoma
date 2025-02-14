import pandas as pd
import joblib
import streamlit as st
from sklearn.preprocessing import StandardScaler

# Load the model and mappings
model = joblib.load('log.pkl')
# Mapping dictionaries
Sex_mapping = {'Male': 1, 'Female': 0}
race_mapping = {
    'White': 1,
    'Asian or Pacific Islander': 2,
    'Black': 3,
    'American Indian/Alaska Native': 4
}
Marital_status_mapping = {'Married': 1, 'Not married': 0}
AJCC_Stage_mapping= {'IA': 0, 'IB': 1, 'IIA': 2, 'IIB': 3, 'III': 4, 'IVA': 5, 'IVB': 6}
AJCC_T_stage_mapping= {'T1': 0, 'T2': 1, 'T3': 2}
AJCC_N_stage_mapping= {'N0': 0, 'N1': 1}
AJCC_M_stage_mapping = {'M0': 0, 'M1a': 1, 'M1b': 2}
Histology_mapping= {'9180/3: Osteosarcoma, NOS': 0, '9181/3: Chondroblastic osteosarcoma': 1, 
            '9182/3: Fibroblastic osteosarcoma': 2, '9183/3: Telangiectatic osteosarcoma': 3,
                  '9184/3: Osteosarcoma in Paget disease of bone': 4, 
                  '9185/3: Small cell osteosarcoma': 5, '9186/3: Central osteosarcoma': 6}
Primary_Site_mapping= {'C40.0-Long bones: upper limb, scapula, and associated joints': 0,
                'C40.1-Short bones of upper limb and associated joints': 1,
                     'C40.2-Long bones of lower limb and associated joints': 2,
                       'C40.3-Short bones of lower limb and associated joints': 3, 
                       'C41.0-Bones of skull and face and associated joints': 4,
                     'C41.4-Pelvic bones, sacrum, coccyx and associated joints': 5}
SEER_stage_mapping= {'Distant': 0, 'Localized': 1, 'Regional': 2}
Surgery_mapping= {'Yes': 1, 'No': 0}
Radiation_mapping= {'Yes': 1, 'No': 0}
Chemotherapy_mapping= {'No': 0, 'Yes': 1}


# Streamlit App

Age = st.slider("Age (years)", min_value=0, max_value=100, value=50, step=1)
Sex = st.radio("Sex", ['Male', 'Female'])
Race = st.selectbox("Race", ['White', 'Asian or Pacific Islander', 'Black', 'American Indian/Alaska Native'])
Maritalstatus = st.radio("Marital status", ['Married', 'Not married'])
AJCC_Stage= st.selectbox("AJCC Stage",['IA', 'IB', 'IIA', 'IIB', 'III', 'IVA', 'IVB'])
AJCC_T_stage= st.selectbox("AJCC T Stage", ['T1', 'T2', 'T3'])
AJCC_N_stage= st.radio("AJCC N Stage", ['N0', 'N1'])
AJCC_M_stage = st.selectbox("AJCC M Stage", ['M0', 'M1a', 'M1b'])
Histology= st.selectbox("Histology", ['9180/3: Osteosarcoma, NOS', '9181/3: Chondroblastic osteosarcoma', 
            '9182/3: Fibroblastic osteosarcoma', '9183/3: Telangiectatic osteosarcoma',
                  '9184/3: Osteosarcoma in Paget disease of bone', 
                  '9185/3: Small cell osteosarcoma', '9186/3: Central osteosarcoma'])
Site = st.selectbox("Primary Site", ['C40.0-Long bones: upper limb, scapula, and associated joints',
                'C40.1-Short bones of upper limb and associated joints',
                     'C40.2-Long bones of lower limb and associated joints',
                       'C40.3-Short bones of lower limb and associated joints', 
                       'C41.0-Bones of skull and face and associated joints',
                     'C41.4-Pelvic bones, sacrum, coccyx and associated joints'])
SEER_Stage = st.selectbox("SEER Stage", ['Localized', 'Regional', 'Distant'])
Tumorsize = st.slider("tumor size", min_value=0, max_value=500, value=50, step=1)
Surgery= st.radio("Surgery", ['Yes', 'No'])
Radiation= st.radio("Radiation", ['Yes', 'No'])
Chemotherapy= st.radio("Chemotherapy", ['No', 'Yes'])

# Preprocess the user input using the same mappings
user_df = pd.DataFrame({
    'Sex': [Sex],
    'Race': [Race],
    'Marital status': [Maritalstatus],
    'AJCC Stage': [AJCC_Stage],
    'AJCC T stage': [AJCC_T_stage],
    'AJCC N stage':[AJCC_N_stage],
    'AJCC M stage': [AJCC_M_stage],
    'Surgery': [Surgery],
    'Radiation': [Radiation],
    'Chemotherapy': [Chemotherapy],
    'tumor size ': [Tumorsize],
    'Histology' : [Histology],
    'Primary Site': [Site],
    'SEER stage': [SEER_Stage],    
    'Age': [Age]

})

# Apply mappings to the dataframe
user_df['Sex'] = user_df['Sex'].map(Sex_mapping)
user_df['Race'] = user_df['Race'].map(race_mapping)
user_df['Marital status'] = user_df['Marital status'].map(Marital_status_mapping)
user_df['Primary Site'] = user_df['Primary Site'].map(Primary_Site_mapping)
user_df['AJCC Stage'] = user_df['AJCC Stage'].map(AJCC_Stage_mapping)
user_df['Histology'] = user_df['Histology'].map(Histology_mapping)
user_df['AJCC T stage'] = user_df['AJCC T stage'].map(AJCC_T_stage_mapping)
user_df['AJCC M stage'] = user_df['AJCC M stage'].map(AJCC_M_stage_mapping)
user_df['AJCC N stage'] = user_df['AJCC N stage'].map(AJCC_N_stage_mapping)
user_df['SEER stage'] = user_df['SEER stage'].map(SEER_stage_mapping)
user_df['Chemotherapy'] = user_df['Chemotherapy'].map(Chemotherapy_mapping)
user_df['Radiation'] = user_df['Radiation'].map(Radiation_mapping)
user_df['Surgery'] = user_df['Surgery'].map(Surgery_mapping)
# Reshape 'Age' into a 2D array and apply the scaler
scaler = StandardScaler()
user_df['Age'] = scaler.fit_transform(user_df[['Age']])
# Reshape 'Age' into a 2D array and apply the scaler
user_df['tumor size '] = scaler.fit_transform(user_df[['tumor size ']])

user_df['Age'] = user_df['Age'].astype(float)  # Convert 'Age' to float (you may adjust the data type)
# Reshape 'Age' into a 2D array and apply the scaler
user_df['tumor size '] = user_df['tumor size '].astype(float) 


if st.button("Make Prediction"):
    # Assuming 'model' is your pre-trained model
    prediction = model.predict(user_df)  # Make predictions using the model
    
    # Assuming the model is a classification model, predicting the survival status (alive or dead)
    if model.predict_proba:
        survival_probability = model.predict_proba(user_df)[0][1]  # Get the probability of being alive (1)

        # Decision rule: If survival probability is greater than 50%, patient is alive; otherwise dead
        if survival_probability > 0.5:
            st.write(f"The patient is predicted to be alive with a survival probability of {survival_probability * 100:.2f}%")
        else:
            st.write(f"The patient is predicted to be dead with a survival probability of {survival_probability * 100:.2f}%")
    else:
        # If the model doesn't support probabilities, you can just print the prediction result
        if prediction < 0:
            st.write("The predicted survival years for this patient is negative. Unable to provide accurate prediction.")
        else:
            st.write(f"The predicted survival years for this patient is: {prediction[0] * 12} months")

