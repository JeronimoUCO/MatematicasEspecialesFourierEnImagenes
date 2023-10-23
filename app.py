import streamlit as st
from algoritmo import algoritmo
import numpy as np
import cv2

st.title("Sube una imagen borrosa que quieras restaurar")
test_image=st.file_uploader("Escoge una imagen")
sigma=st.slider("Mueve este slider para ajustar la limpieza de tu imagen",0,1000,1)

if test_image is not None:
  col1,col2=st.columns(2)
  test_image = np.asarray(bytearray(test_image.read()), dtype=np.uint8)
  test_image = cv2.imdecode(test_image, 1)
  imagen_restaurada=algoritmo.restaurar(test_image,sigma)
  
  with col1:
    st.subheader("Imagen original")
    st.image(test_image)
  with col2:
    st.subheader("Imagen restaurada")
    st.image(imagen_restaurada)

