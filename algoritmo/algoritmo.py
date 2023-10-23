import cv2
import os
import numpy as np
import math


def cargar_imagen(imagen):
    # Cargar la imagen utilizando OpenCV y escala de grises

    if imagen is not None:
        imagen_gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
        return imagen_gris
    else:
        print("No se pudo cargar la imagen.")
        return None


def calcular_matriz_de_distancias(imagen):
    # Obtener las dimensiones de la imagen
    alto, ancho = imagen.shape


    u_central = ancho // 2
    v_central = alto // 2

    D = np.zeros((alto, ancho), dtype=np.float32)

    # Calcular las distancias euclidianas y almacenarlas en D
    for v in range(alto):
        for u in range(ancho):
            distancia = np.sqrt((u - u_central) ** 2 + (v - v_central) ** 2)
            D[v, u] = distancia

    return D

def calcular_H(D, theta):
    # Calcular H(u, v)
    E = math.e
    H = E ** (-((D ** 2) / (2 * (theta ** 2))))

    return H


def calcular_transformada_fourier(imagen):
    return np.fft.fft2(imagen)

def calcular_transformada_inversa(G, H):
   
    F_deconvolucion = np.fft.ifft2(G / H)
    return np.abs(F_deconvolucion).astype(np.uint8)

def guardar_imagen(carpeta, nombre_archivo, imagen):
    ruta_guardar = os.path.join(carpeta, nombre_archivo)
    cv2.imwrite(ruta_guardar, imagen)

def restaurar(imagen, sigma):
    imagen=cargar_imagen(imagen)

    if imagen is not None:
        D = calcular_matriz_de_distancias(imagen)
        H = calcular_H(D, sigma)
        G=calcular_transformada_fourier(imagen)
        resultado=calcular_transformada_inversa(G, H)
        return resultado