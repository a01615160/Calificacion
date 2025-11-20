import numpy as np
import streamlit as st
import pandas as pd

st.write(''' # Predicción de la calificacion  ''')
st.image("exa.jpg", caption="Predicción de la calificación")

st.header('Datos personales')

def user_input_features():
  # Entrada
  Horas_de_estudio = st.number_input('Horas de estudio:', min_value=0.0, max_value=12.0, value = 0)
  Horas_de_sueño = st.number_input('Horas de sueño:',  min_value=0.0, max_value=10.0, value = 0)
  Pocentaje_de_clases_atendidas = st.number_input('Porcentaje de clases atendidas:', min_value=0.0, max_value=100.0, value = 0)
  Puntajes_anteriores = st.number_input('Puntajes anteriores:', min_value=0.0, max_value=100.0, value = 0)


  user_input_data = {'hours_studied': Horas_de_estudio,
                     'sleep_hours': Horas_de_sueño,
                     'attendance_percent': Pocentaje_de_clases_atendidas,
                     'previous_scores': Puntajes_anteriores,
                     }

  features = pd.DataFrame(user_input_data, index=[0])

  return features

df = user_input_features()
datos =  pd.read_csv('Examen_df.csv', encoding='latin-1')
X = datos.drop(columns='exam_score')
y = datos['exam_score']

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=1615160)
LR = LinearRegression()
LR.fit(X_train,y_train)

b1 = LR.coef_
b0 = LR.intercept_
prediccion = b0 + b1[0]*df['hours_studied'] + b1[1]*df['sleep_hours'] + b1[2]*df['attendance_percent'] + b1[3]*df['previous_scores']

st.subheader('Cálculo de la calificación')
st.write('La calificación de la persona es ', prediccion)
