

import streamlit as st
import requests
import pandas as pd
import json
import base64  
import os      


#Adicionar a Imagem de Fundo (CSS Injection)
def add_bg_from_local(image_file):
    # Encontra o caminho do arquivo de imagem
    image_path = os.path.join(os.path.dirname(__file__), image_file)
    
    # Abre a imagem, codifica em Base64 e prepara para o CSS
    with open(image_path, "rb") as image_file_content:
        encoded_string = base64.b64encode(image_file_content.read()).decode()
    
    # Injeta o CSS na página para definir o fundo e melhorar a legibilidade
    st.markdown(
    f"""
    <style>
    /* 1. Define a imagem de fundo para toda a aplicação */
    .stApp {{
        background-image: url("data:image/{"png" if image_file.endswith('png') else "jpeg"};base64,{encoded_string}");
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}

    /* 2. Adiciona um fundo semi-transparente à barra lateral */
    .stSidebar > div:first-child {{
        background-color: rgba(245, 245, 245, 0.85); /* Um cinza bem claro */
    }}

    /* 3. Adiciona um fundo semi-transparente ao container principal do conteúdo */
    .main .block-container {{
        background-color: rgba(255, 255, 255, 0.85);
        padding: 2rem;
        border-radius: 10px;
    }}

    
    
    .stAlert {{
        background-color: rgba(230, 246, 255, 0.9); /* Fundo para st.info */
    }}
    .st-emotion-cache-1wivap2 {{ /* Classe CSS específica para st.success */
        background-color: rgba(230, 255, 237, 0.9);
    }}
    .st-emotion-cache-jlqrow {{ /* Classe CSS específica para st.warning */
        background-color: rgba(255, 244, 229, 0.9);
    }}
    .st-emotion-cache-keje6w {{ /* Classe CSS específica para st.error */
        background-color: rgba(255, 235, 235, 0.9);
    }}

    </style>
    """,
    unsafe_allow_html=True
    )

# --- Função Principal do App Streamlit ---

#  Configuração da Página 
st.set_page_config(
    page_title="Predição de Preços de Imóveis",
    page_icon="🏠",
    layout="wide"
)

#  vamos Inicializqr o Estado da Sessão 
# memória para a sessão do usuário.
# Se a variável 'show_background' não existir, ela é criada como False.
if 'show_background' not in st.session_state:
    st.session_state.show_background = False

#  Título e Descrição 
st.title("API de Classificação de Preços de Imóveis 🏠")
st.markdown("""
Esta interface permite interagir com os modelos de Machine Learning (Random Forest e SVM) para prever 
se o valor mediano de um imóvel está **acima ou abaixo da mediana**.

**Instruções:**
1.  **Escolha o modelo** que deseja usar no painel à esquerda.
2.  **Ajuste os valores** das features.
3.  Clique em "Fazer Predição".
""")

#  Painel Lateral com Inputs do Usuário 
st.sidebar.header("Defina os Parâmetros da Predição")
modelo_selecionado = st.sidebar.selectbox('Escolha o Modelo:', ('Random Forest', 'SVM'))
st.sidebar.header("Features do Imóvel")

feature_ranges = {
    'MedInc': (0.0, 20.0), 'HouseAge': (0.0, 55.0), 'AveRooms': (0.9, 58.0), 
    'AveBedrms': (0.5, 11.0), 'Population': (9.0, 30000.0), 'AveOccup': (1.07, 10.0), 
    'Latitude': (32.55, 42.0), 'Longitude': (-124.35, -114.31)
}

def user_input_features():
    inputs = {}
    inputs['MedInc'] = st.sidebar.slider('Renda Mediana (x $10k)', min_value=feature_ranges['MedInc'][0], max_value=feature_ranges['MedInc'][1], value=3.87, step=0.1)
    inputs['AveRooms'] = st.sidebar.slider('Média de Quartos', min_value=feature_ranges['AveRooms'][0], max_value=feature_ranges['AveRooms'][1], value=5.42, step=0.1)
    inputs['AveBedrms'] = st.sidebar.slider('Média de Dormitórios', min_value=feature_ranges['AveBedrms'][0], max_value=feature_ranges['AveBedrms'][1], value=1.1, step=0.05)
    inputs['AveOccup'] = st.sidebar.slider('Média de Ocupantes', min_value=feature_ranges['AveOccup'][0], max_value=feature_ranges['AveOccup'][1], value=3.0, step=0.1)
    inputs['HouseAge'] = st.sidebar.number_input('Idade da Casa (Média)', min_value=feature_ranges['HouseAge'][0], max_value=feature_ranges['HouseAge'][1], value=28.0, step=1.0)
    inputs['Population'] = st.sidebar.number_input('População do Bairro', min_value=feature_ranges['Population'][0], max_value=feature_ranges['Population'][1], value=1425.0, step=1.0)
    inputs['Latitude'] = st.sidebar.number_input('Latitude', min_value=feature_ranges['Latitude'][0], max_value=feature_ranges['Latitude'][1], value=35.63, format="%.2f", step=0.5)
    inputs['Longitude'] = st.sidebar.number_input('Longitude', min_value=feature_ranges['Longitude'][0], max_value=feature_ranges['Longitude'][1], value=-119.57, format="%.2f", step=0.5)
    return inputs

input_data = user_input_features()


#  Exibição dos Dados de Entrada 
st.subheader("Valores de Entrada Selecionados:")
st.info(f"**Modelo Selecionado:** {modelo_selecionado}")
input_df = pd.DataFrame([input_data])
st.write(input_df)

# Botão de Predição e Lógica de Chamada da API (LÓGICA AJUSTADA) v 3.0
if st.button("Fazer Predição", type="primary"):
    # Verifica se esta é a primeira vez que o botão é clicado (o fundo ainda não está visível!!!!)
    needs_rerun_for_background = not st.session_state.show_background
    
    # Define o estado para True para que o fundo apareça na próxima execução
    st.session_state.show_background = True
    
    # Lógica para chamar a API (não muda em relação à versão anterior)
    #definição do endpoint 
    if modelo_selecionado == 'Random Forest':
        endpoint = "predict/rf"
    else:
        endpoint = "predict/svm"
        
    api_url = f"http://api:8000/{endpoint}"
    payload = json.dumps(input_data)
    headers = {'Content-Type': 'application/json'}
    
    #  bloco de exibição de resultado 
    try:
        with st.spinner('Aguardando a resposta da API...'):
            response = requests.post(api_url, data=payload, headers=headers)
        
        if response.status_code == 200:
            prediction_data = response.json()
            predicao = prediction_data.get('predição')
            
            st.subheader("Resultado da Predição:")
            
            if predicao == "Preço acima da mediana":
                st.success(f"📈 Modelo {modelo_selecionado} previu: **{predicao}**")
                st.balloons()
            else:
                st.warning(f"📉 Modelo {modelo_selecionado} previu: **{predicao}**")

            with st.expander("Ver resposta completa da API"):
                st.json(prediction_data)
        else:
            st.error(f"Erro ao chamar a API. Status Code: {response.status_code}")
            st.json(response.json())
    except Exception as e:
        st.error(f"Ocorreu um erro inesperado: {e}")

    # --- LÓGICA DE RERUN CORRIGIDA em relação a versão passada---
    #CONSIDERAÇÕES
        # Só executa o rerun se for a primeira predição, para carregar o fundo.
        # Nas predições seguintes, ele não fará o rerun, e o resultado permanecerá na tela.
    if needs_rerun_for_background:
        st.rerun()