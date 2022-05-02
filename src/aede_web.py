from autoaede_functions import read_geodata, weights_matrix, plot_moran, map_weighted, plot_lisa, plot_weights, otimizar_k
import streamlit as st

st.title('AUTO AEDE')


file = st.file_uploader('Arquivo')

if file:
    df = read_geodata(file)
    
    k_min, k_max = 1,10
    
    st.table(otimizar_k(df, 'tx_100mil', 1, 10, 0.01))
    
    k_opt = st.slider('Escolha o k', 1, 10)
    
    if k_opt:
    
        pesos = weights_matrix(df, metric = 'knn', k=k_opt)

        st.pyplot(plot_weights(weight=pesos, dados=df))

        st.pyplot(plot_moran(df, 'tx_100mil', weight=pesos))

        st.pyplot(map_weighted(df, 'tx_100mil', 'Título'))

        st.pyplot(plot_lisa(df, 'tx_100mil', weights = pesos, k_opt = k_opt))