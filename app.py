#Esta aplicação foi desenvolvida para prever as despesas hospitalares de pacientes.

from pycaret.regression import load_model, predict_model

import streamlit as st
import pandas as pd
import numpy as np

#carrega o modelo treinado para previsão

#model = load_model('deployment_18062020')
model = load_model('deployment_18062020')

# Define a chamada da função


def predict(model, imput_df):
    predictions_df = predict_model(estimator=model, data=imput_df)
    predictions = prediction_df['Label'][0]

    return predictions

#Função principal


def run():
    from PIL import Image
    image = image.open('logo.png')
    image_hospital = Image.open('hospital.png')

    st.image(image, use_column_width = false)

#adicionando a barra lateral

add_selectbox = st.sidebar.selectbox(
        "Como gostaria de fazer a previsão?",
        ("online", "Batch"))


st.sidebar.info(
    'Esta aplicação foi desenvolvida para prever as despesas hospitalares de pacientes.')
st.sidebar.success('https://pycaret.org')

#st.sidebar.image(image_hospital)

st.title('Aplicação de previsão de encargos de seguro')


if add_selectbox == 'online':  # apenas para previsão online
    #capturando todas as entradas para realizar a previsão utilizado o streamlit
    age = st.number_input('Age', min_value=1, max_value=100, value=25)
    sex = st.selectbox('Sex', ['male', 'female'])
    bmi = st.number_input('BMI', min_value=10, max_value=50, value=10)
    children = st.selectbox('Children', [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

if st.checkbox == 'Smoker':
    smoker = 'yes'
else:
    smoker = 'no'

region = st.selectbox(
    'Region', ['Sudeste', 'Nordeste', 'Suldoeste', 'Noroeste'])

output = ""

input_dict = {'age': age, 'sex': sex, 'bmi': bmi,'children': children, 'smoker': smoker, 'region': region}
input_df = pd.DataFrame([input_dict])

# retornará a previsão quando aperta o botão
if st.button("Predict"):
    #output = predict(model=model, input_df=input_df)
    output = predict(model=model, input_df=input_df)
    output = '$' + str(output)


st.success('A Saída é {}'.format(output))


if add_selectbox == 'Batch':
    #Upload do arquivo CSV para previsão
    file_upload = st.file_uploader(
        "D:/DevSpace/web/ML/pycaretlearn/data/datasets_13720_18513_insurance.csv", type=["CSV"])

    if file_upload is not None:
        data = pd.read_csv(file_upload)
        predictions = predict_model(estimator=model, data=data)
        st.write(predictions)
