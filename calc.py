import pandas as pd
import numpy as np

import streamlit as st
from streamlit.logger import get_logger
from tensorflow import keras


# Load model

model_loaded = keras.models.load_model('model_af')


# Interface

LOGGER = get_logger(__name__)
def run():
    st.set_page_config(
        page_title="AF_predict-TTE-NW",
        page_icon="https://papik.pro/uploads/posts/2022-01/thumbs/1643604901_4-papik-pro-p-serdtse-logotip-5.png",
    )

    # st.markdown("![Alt ​​Text](https://media.giphy.com/media/vFKqnCdLPNOKc/giphy.gif)")

    st.write(" # Трансторакальная эхокардиография")

    st.markdown(""" с интегрированной нейронной сетью """)
    
    ao = st.slider('Аорта, см', min_value=2.0, max_value=8.0, value=0.0, step=0.1, 
                   format=None, key=None, help='Введите размер восходящего отдела аорты, см', 
                   on_change=None, args=None, kwargs=None, disabled=False, label_visibility="visible")
    
    la = st.slider('Левое предсердие, см', min_value=2.0, max_value=8.0, value=0.0, step=0.1, 
                   format=None, key=None, help='Введите размер левого предсердия в парастернальной позиции по длинной оси, см', 
                   on_change=None, args=None, kwargs=None, disabled=False, label_visibility="visible")
    
    lv = st.slider('Левый желудочек, см', min_value=2.0, max_value=8.0, value=0.0, step=0.1, 
                   format=None, key=None, help='Введите косой диастолический размер левого желудочка в парастернальной позиции по длинной оси, см', 
                   on_change=None, args=None, kwargs=None, disabled=False, label_visibility="visible")
    
    rv = st.slider('Правый желудочек, см', min_value=1.0, max_value=6.0, value=0.0, step=0.1, 
                   format=None, key=None, help='Введите косой диастолический размер правого желудочка в парастернальной позиции по длинной оси, см', 
                   on_change=None, args=None, kwargs=None, disabled=False, label_visibility="visible")
    
    ra = st.slider('Правое предсердие, см', min_value=2.0, max_value=8.0, value=0.0, step=0.1, 
                   format=None, key=None, help='Введите поперечный размер правого предсердия в апикальной четырехкамерной позиции, см', 
                   on_change=None, args=None, kwargs=None, disabled=False, label_visibility="visible")
    
    pa = st.slider('Легочная артерия, см', min_value=1.0, max_value=4.0, value=0.0, step=0.1, 
                   format=None, key=None, help='Введите размер легочной артерии на уровне клапана в парастернальной позиции по короткой оси, см', 
                   on_change=None, args=None, kwargs=None, disabled=False, label_visibility="visible")
    
    lvaw = st.slider('Межжелудочковая перегородка, см', min_value=0.6, max_value=3.0, value=0.0, step=0.1, 
                   format=None, key=None, help='Введите максимальную толщину межжелудочковой перегородки в парастернальной позиции по длинной оси на уровне створок митрального клапана, см', 
                   on_change=None, args=None, kwargs=None, disabled=False, label_visibility="visible")
    
    lvpw = st.slider('Задняя стенка левого желудочка, см', min_value=0.6, max_value=3.0, value=0.0, step=0.1, 
                   format=None, key=None, help='Введите максимальную толщину задней стенки левого желудочка в парастернальной позиции по длинной оси на уровне створок митрального клапана, см', 
                   on_change=None, args=None, kwargs=None, disabled=False, label_visibility="visible")
    
    ef = st.slider('Фракция выброса левого желудочка, %', min_value=10.0, max_value=80.0, value=0.0, step=1.0, 
                   format=None, key=None, help='Введите фракцию выброса, измеренную методом Тейхольца, %', 
                   on_change=None, args=None, kwargs=None, disabled=False, label_visibility="visible")
    
    ar = st.slider('Аортальная регургитация, степень', min_value=0.0, max_value=4.0, value=0.0, step=0.5, 
                   format=None, key=None, help='Введите степень аортальной регургитации, определенную в режиме ЦДК', 
                   on_change=None, args=None, kwargs=None, disabled=False, label_visibility="visible")
    
    mr = st.slider('Митральная регургитация, степень', min_value=0.0, max_value=4.0, value=0.0, step=0.5, 
                   format=None, key=None, help='Введите степень митральной регургитации, определенную в режиме ЦДК', 
                   on_change=None, args=None, kwargs=None, disabled=False, label_visibility="visible")
    
    tr = st.slider('Трикуспидальная регургитация, степень', min_value=0.0, max_value=4.0, value=0.0, step=0.5, 
                   format=None, key=None, help='Введите степень трикуспидальной регургитации, определенную в режиме ЦДК', 
                   on_change=None, args=None, kwargs=None, disabled=False, label_visibility="visible")
    
    cv = st.slider('Нижняя полая вена, см', min_value=0.6, max_value=4.0, value=0.0, step=0.1, 
                   format=None, key=None, help='Введите размер нижней полой вены на выдохе в субкостальной позиции по длинной оси, см', 
                   on_change=None, args=None, kwargs=None, disabled=False, label_visibility="visible")
    
    
    
    # Preprocessing variables
    
    ao1 = (ao - 1.8) / (5.4 - 1.8)
    la1 = (la - 2.4) / (7.8 - 2.4)
    lv1 = (lv - 2.4) / (8.1 - 2.4)
    rv1 = (rv - 1.5) / (4.8 - 1.5)
    ra1 = (ra - 2.3) / (6.0 - 2.3)
    pa1 = (pa - 1.0) / (3.6 - 1.0)
    lvaw1 = (lvaw - 0.7) / (2.6 - 0.7)
    lvpw1 = (lvpw - 0.7) / (2.2 - 0.7)
    ef1 = (ef - 18.0) / (75.0 - 18.0)
    ar1 = (ar - 0) / (2.5 - 0)
    mr1 = (mr - 0) / (3.5 - 0)
    tr1 = (tr - 0) / (3.5 - 0)
    cv1 = (cv - 0) / (3.5 - 0)
    
    
    # Prediction
    
    x = [ao1, la1, lv1, rv1, ra1, pa1, lvaw1, lvpw1, ef1, ar1, mr1, tr1, cv1]
    x = np.array(x)
    nn = np.expand_dims(x, axis=0)
    res = model_loaded.predict(nn)
    
    report = res[0].tolist()
    
    
    if report[0] > 0.85:
        st.write("Высокая вероятность фибрилляции предсердий")
    elif report[0] > 0.50 and report[0] < 0.84:
        st.write("Умеренная вероятность фибрилляции пресердий")
    else:
        st.write("Фибрилляця пресердий маловероятна")
    
    st.write('Показатели эхокардиографии соответствуют таковым при ФП на', round(float(report[0]) * 100, 2), ' %')

    st.markdown(""" Разработка идеи и реализация: Любавин А. В., Котляров С. Н. """)
    
    
    
    

if __name__ == "__main__":
    run()
