import streamlit as st
import pickle
import numpy as np


def load_model():
    with open('credit_predict4.sav', 'rb') as file:
        data = pickle.load(file)
    return data

data = load_model()

svm_loaded=data["model"]
S_encoder= data["S_encoder"]
J_encoder=data["J_encoder"]
H_encoder=data["H_encoder"]
SA_encoder=data["SA_encoder"]
CA_encoder=data["CA_encoder"]
P_encoder=data["P_encoder"]

def show_predict_page():
    st.title("Software Developer Credit Risk Prediction")

    st.write("""### We need some information for prediction""")

    Sex =("male", "female",)
    Job = ("unskilled and non-resident", "unskilled and resident", "skilled", "highly skilled",)
    Housing = ("own", "free", "rent",)
    Saving_accounts = ("none", "little", "quite rich", "rich", "moderate",)
    Checking_account = ("little", "moderate", "none", "rich",)
    Purpose = ("radio/TV", "education", "furniture/equipment", "car", "business", "domestic appliances", "repairs", "vacation/others",)

    Sex = st.selectbox("Sex", Sex)
    Job = st.selectbox("Skill Level", Job)
    Housing = st.selectbox("Housing", Housing)
    Saving_accounts = st.selectbox("Saving accounts rate", Saving_accounts)
    Checking_account = st.selectbox("Checking account Rate", Checking_account)
    Purpose = st.selectbox("Purpose of Credit", Purpose)

    Age = st.number_input("Age", min_value=18, step=1)
    Credit_amount = st.number_input("Credit amount in Naira", min_value=0, step=1)
    Duration = st.number_input("Duration", min_value=0, step=1)

    ok = st.button("Predict")
    if ok:
        X = np.array([[Sex, Job, Housing,Saving_accounts,Checking_account,Purpose,Age,Credit_amount,Duration ]])
        X[:, 0] = S_encoder.transform(X[:, 0])
        X[:, 1] = J_encoder.transform(X[:, 1])
        X[:, 2] = H_encoder.transform(X[:, 2])  # Reshape for compatibility
        X[:, 3] = SA_encoder.transform(X[:, 3])  # Reshape for compatibility
        X[:, 4] = CA_encoder.transform(X[:, 4])  # Reshape for compatibility
        X[:, 5] = P_encoder.transform(X[:, 5])  # Reshape for compatibility 
        X=X.astype(float)
        predi = int(svm_loaded.predict(X))
        if predi == 1:
            st.write("The customer is eligible for a loan.")
        elif predi == 0:
            st.write("The customer is not eligible for a loan.")
        else:
            st.write("Investigate the customer thoroughly.")        
