# vamos começar com a imagem
FROM python:3.9-slim                  
#digo que tipo de kernel do python eu quero
WORKDIR /app 
#define o diretório de trabalho dentro do container
COPY requirements.txt . 
# copia o arquivo das dependências no diretório
RUN pip install --no-cache-dir -r requirements.txt 
#instala as dependências
COPY . . 
#copia o resto dos arquivos para a pasta app (incluindo models)
EXPOSE 8000  
#expõe a porta 8000
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"] 
#comando para rodar o servidor

#uvicorn é o servidor ASGI que roda a aplicação FastAPI
#rode com docker run -d -p 8080:8000 --name meu-container-api api-imoveis