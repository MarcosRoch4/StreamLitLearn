#Esta aplicação foi desenvolvida para prever as despesas hospitalares de pacientes.

from pycaret.regression import load_model, predict_model
import streamlit as st
import pandas as pd
import numpy as np

model = load_model('deployment_18062020')

def predict(model, input_df):
    predictions_df = predict_model(estimator=model, data=input_df)
    predictions = predictions_df['Label'][0]
    return predictions

def run():

    from PIL import Image
    image = Image.open('logo.png')
    image_hospital = Image.open('hospital.jpg')

    st.image(image,use_column_width=False)

    add_selectbox = st.sidebar.selectbox(
    "Como gostaria de fazer a previsão?",
    ("Online", "Batch"))

    st.sidebar.info('Esta aplicação foi desenvolvida para prever as despesas hospitalares de pacientes.')
    st.sidebar.success('http://sibylconsultoria.com.br  ')
    
    st.sidebar.image(image_hospital)

    st.title("Aplicação de previsão de encargos de seguro")

    if add_selectbox == 'Online':

        age = st.number_input('Idade', min_value=1, max_value=100, value=25)
        sex = st.selectbox('Sexo', ['Masculino', 'Feminino'])
        bmi = st.number_input('BMI', min_value=10, max_value=50, value=10)
        children = st.selectbox('Filhos', [0,1,2,3,4,5,6,7,8,9,10])
        if st.checkbox('Fumante'):
            smoker = 'sim'
        else:
            smoker = 'não'
        region = st.selectbox('Região', ['Sudeste', 'Nordeste', 'Suldoeste', 'Noroeste'])

        output=""

        input_dict = {'Idade': age, 'Sexo': sex, 'BMI': bmi,'Filhos': children, 'Fumante': smoker, 'Região': region}
        input_df = pd.DataFrame([input_dict])

        if st.button("Prever"):
            output = predict(model=model, input_df=input_df)
            output = '$' + str(output)

        st.success('A saída é {}'.format(output))

    if add_selectbox == 'Batch':

        file_upload = st.file_uploader("Faça o upload do arquivo em no formato CSV.", type=["csv"])

        if file_upload is not None:
            data = pd.read_csv(file_upload)
            predictions = predict_model(estimator=model,data=data)
            st.write(predictions)

if __name__ == '__main__':
    run()