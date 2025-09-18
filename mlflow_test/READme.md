Aqui faremos experimentos com mais hiperparâmetros usando a UI do mlflow.
Os experimentos se dividirão em dois: experimentos com o modelo de Random Forest e experimentos com SVM. Após serem feitos os experimentos, guardarei os modelos que performarem melhor na métrica escolhida (auroc) em relação a aqueles gerados no jupyter notebook. Não automatizei essa parte específica para deixar opcional o uso dos modelos adquiridos nessa etapa -- já que a performance daqueles encontrados no arquivo jupyter notebook  é aceitável. 

Caso queira trocar os modelos originais por aqueles com uma perfomance melhor nos experimentos do mlflow é só substituir os campos em server.py pelos nomes dos novos modelos tanto de random forest como svm.
