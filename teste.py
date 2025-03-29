# -*- coding: utf-8 -*-
"""
Created on Fri Mar 21 16:18:18 2025

@author: alexa
"""

import streamlit as st
import numpy as np
from scipy.optimize import minimize
from scipy.optimize import differential_evolution

def parse_input(input_str):
    """Converte a string de entrada em uma lista de números."""
    try:
        return [float(num.strip()) for num in input_str.split(',') if num.strip()]
    except ValueError:
        return []

#definição das funções da distribuição de probabilidade
def f(t,beta,eta):
    return (beta/eta)*((t/eta)**(beta-1))*np.exp(-((t/eta)**beta))

def F(t,beta,eta):
    return 1 - np.exp(-((t/eta)**beta))

def R(t,beta,eta):
    return np.exp(-((t/eta)**beta))

#função de verossimilhança
def V_neg_log(parametros, DCp, DCE, DCD):
    beta = parametros[0]
    eta = parametros[1]
    Vs = 0
    if len(DCp)>0:
        for i in range(0,len(DCp)):
            Vs = Vs + np.log10(f(DCp[i],beta,eta))
    if len(DCE)>0:
        for j in range(0,len(DCE)):
            Vs = Vs + np.log10(F(DCE[j],beta,eta))
    if len(DCD)>0:
        for k in range(0,len(DCD)):
            Vs = Vs + np.log10(R(DCD[k],beta,eta))
    return -Vs

# Configuração da interface
st.title("Ajuste para a Distribuição Weibull")

# Campos de entrada para os vetores
DCp_str = st.text_input("DADOS COMPLETOS - insira os dados de tempo até a falha, separados por vírgula:")
DCE_str = st.text_input("DADOS CENSURADOS À ESQUERDA - insira os dados de quando as falhas foram observadas, separados por vírgula:")
DCD_str = st.text_input("DADOS CENSURADOS À DIREIRA - insira os dados do tempo de funcionamento das unidades que não falharam, separados por vírgula:")


if st.button("Estimar parâmetros:"):
    # Converter entradas para listas numéricas
    DCp = parse_input(DCp_str)
    DCE = parse_input(DCE_str)
    DCD = parse_input(DCD_str)

    #minimizando V_neg (maximizando V)
    vetor_comum = DCp
    vetor_comum = np.append(vetor_comum, DCE)
    vetor_comum = np.append(vetor_comum, DCD)
    eta_max = max(vetor_comum)
    bounds = [(0,10), (0,eta_max)]
    res = differential_evolution(lambda x: V_neg_log(x, DCp, DCE, DCD),bounds)
    
    beta_estimado = res.x[0]
    eta_estimado = res.x[1]
    
    # Exibir resultados
    st.write(f"Parâmetro de forma: {beta_estimado}")
    st.write(f"Parâmetro de escala: {eta_estimado}")
