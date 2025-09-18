from fastapi import FastAPI
from joblib import load
from pydantic import BaseModel, Field
import numpy as np
import os
# a primeira coisa a se fazer quando você cria um app do estilo fastapi é criar a instância do app
#imprima na instância do app o título, a descrição e a versão
app= FastAPI(
    title=' API de classificação de preços de imóveis',
    description=' Essa api usa o california housing dataset para uma classificação binária de preços de imóveis. Ela diz se os preços estão acima ou abaixo da mediana. Possui dois modelos: Random Forest e SVM.',
    version='1.0.0'


)

#agora vamos especificar os caminhos para os artefatos
MODEL_DIR= 'models'
RF_MODEL_PATH= os.path.join(MODEL_DIR, 'notebook_model_rf.pkl')
SVM_MODEL_PATH= os.path.join(MODEL_DIR, 'notebook_model_svm.pkl')

#após feito isso, vamos carregar os modelos nesse: 
try: 
    rf_model= load(RF_MODEL_PATH)
    svm_model= load(SVM_MODEL_PATH)
    print(' Modelos carregados com sucesso!')
except Exception as e:
    print(f'Erro ao carregar os modelos: {e}')


#vamos definir o input de cada modelo usando o pydantic

class InputDados(BaseModel):
    MedInc: float =Field(..., gt=0, lt=20, description=' Renda mediana (em milhares de dólares) do bairro, você deve setar para algo no range [0, 20]')
    HouseAge: float =Field(..., gt=0, lt=55, description=' Idade média das casas no bairro, você deve setar para algo no range [0, 55]')
    AveRooms: float= Field(..., gt=0.9, lt=58, description=' Número médio de quartos por casa no bairro, você deve setar para algo no range [0.9, 56.5]')
    AveBedrms: float=Field(..., gt=0.5, lt=11, description=' Número médio de quartos por casa no bairro, você deve setar para algo no range [0.5, 11]')
    Population: float= Field(..., gt=9, lt=30000, description=' População do bairro, você deve setar para algo no range [9, 30000]')
    AveOccup: float= Field(..., gt=1.07, lt=10, description=' Número médio de pessoas por casa no bairro, você deve setar para algo no range [1.07, 10]')
    Latitude: float= Field(..., gt=32.55, lt=42., description=' Latitude do bairro, você deve setar para algo no range [32.55, 42]')
    Longitude: float=Field(..., gt=-124.35, lt=-114.31, description=' Longitude do bairro, você deve setar para algo no range [-124.35, -114.31]')

# vamos definir os edpoints da api

@app.get('/')
async def root():
    return {'message': 'Bem-vindo à API de classificação de preços de imóveis! Faço uso de /predict/rf ou /predict/svm.'}

#vamos falar as métricas dos modelos para o user
@app.get('/metrics')
async def get_metrics():
    ''' Retorna as métricas dos modelos.
    '''
    return {'random forest': { 'AUROC':0.9340, 'AUPRC':0.9331},'svm': {'AUROC': 0.9178, 'AUPRC':0.9092}}


@app.post('/predict/rf')
async def predict_rf(input_data: InputDados):
    ''' Realiza a predição usando o modelo Random Forest.
    Espera um input no formato JSON com os seguintes campos:
    - MedInc: Renda mediana (em milhares de dólares) do bairro
    - HouseAge: Idade média das casas no bairro
    - AveRooms: Número médio de quartos por casa no bairro
    - AveBedrms: Número médio de quartos por casa no bairro
    - Population: População do bairro
    - AveOccup: Número médio de pessoas por casa no bairro
    - Latitude: Latitude do bairro
    - Longitude: Longitude do bairro
    Retorna 0 para preços abaixo da mediana e 1 para preços acima da mediana.
    '''
    #aqui daremos um array com os dados que o user deve inputar
    try:
        data= np.array([[ 
            input_data.MedInc,
            input_data.HouseAge,
            input_data.AveRooms,
            input_data.AveBedrms,
            input_data.Population,
            input_data.AveOccup,
            input_data.Latitude,
            input_data.Longitude
        ]])
        pred= rf_model.predict(data)[0]
        resposta_modelo=['Preço abaixo da mediana', 'Preço acima da mediana']
        if pred==0:
            return { 'modelo': 'random forest',
                    'input_data': input_data,
                    'predição':resposta_modelo[0]

            }
        elif pred==1:
            return { 'modelo': 'random forest',
                    'input_data': input_data,
                    'predição':resposta_modelo[1]

            }
        else:
            return {'message': 'Erro na predição do modelo.'}
    except Exception as e:
        return {'message': f'Erro ao processar a requisição: {e}'}

#vamos colar o endpoint do svm a partir do endpoint do rf
@app.post('/predict/svm')
async def predict_svm(input_data: InputDados):
    ''' Realiza a predição usando o modelo SVM.
    Espera um input no formato JSON com os seguintes campos:
    - MedInc: Renda mediana (em milhares de dólares) do bairro
    - HouseAge: Idade média das casas no bairro
    - AveRooms: Número médio de quartos por casa no bairro
    - AveBedrms: Número médio de quartos por casa no bairro
    - Population: População do bairro
    - AveOccup: Número médio de pessoas por casa no bairro
    - Latitude: Latitude do bairro
    - Longitude: Longitude do bairro
    Retorna 0 para preços abaixo da mediana e 1 para preços acima da mediana.
    '''
    
    try:
        data= np.array([[
            input_data.MedInc,
            input_data.HouseAge,
            input_data.AveRooms,
            input_data.AveBedrms,
            input_data.Population,
            input_data.AveOccup,
            input_data.Latitude,
            input_data.Longitude
        ]])
        pred= svm_model.predict(data)[0]
        resposta_modelo=['Preço abaixo da mediana', 'Preço acima da mediana']
        if pred==0:
            return { 'modelo': 'svm',
                    'input_data': input_data,
                    'predição':resposta_modelo[0]

            }
        elif pred==1:
            return { 'modelo': 'svm',
                    'input_data': input_data,
                    'predição':resposta_modelo[1]

            }
        else:
            return {'message': 'Erro na predição do modelo.'}
    except Exception as e:
        return {'message': f'Erro ao processar a requisição: {e}'}



    
    
    


   

