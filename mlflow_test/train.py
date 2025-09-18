# train.py (com Threshold e Experimentos Separados)

import os
import pandas as pd
import mlflow
import mlflow.sklearn
import optuna

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
#vamos importar as métricas de classificação
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# --- Configuração --- # --- Configuração do MLflow --- #tenho que configurar o mlflow para dizer sempre a URI do tracking e o nome do experimento
mlflow.set_tracking_uri("file:./mlruns")
#mlflow.set_experiment("Otimizacao California Housing")

AUROC_THRESHOLD = 0.9341 # threshold encontrado no notebook para o random forest
def load_data(file_path):
    df = pd.read_csv(file_path)
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    return train_test_split(X, y, test_size=0.2, random_state=69)

def optimize_random_forest(X_train, X_test, y_train, y_test):
    """
    Função dedicada à otimização e registro de modelos RandomForest.
    """
    # Define o experimento específico para o RandomForest
    mlflow.set_experiment("Otimizacao RandomForest")
    
    def objective(trial):
        # Inicia uma execução aninhada para CADA tentativa do Optuna
        with mlflow.start_run(nested=True):
            # 1. Sugere os hiperparâmetros
            scaler_type = trial.suggest_categorical('scaler', ['StandardScaler', 'MinMaxScaler', 'RobustScaler'])
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 5, 500),
                'max_depth': trial.suggest_int('max_depth', 5, 30),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
                'criterion': trial.suggest_categorical('criterion', ['gini', 'entropy']),

            }
            
            # Loga os parâmetros imediatamente para que você possa ver o que foi tentado,
            # mesmo que a performance seja baixa.
            mlflow.log_param("model_type", "RandomForest")
            mlflow.log_param("scaler", scaler_type)
            mlflow.log_params(params)
            
            # 2. Treina e avalia a pipeline
            if scaler_type == 'StandardScaler': scaler = StandardScaler()
            elif scaler_type == 'RobustScaler': scaler = RobustScaler()
            else: scaler = MinMaxScaler()
            
            pipeline = Pipeline([
                ('scaler', scaler),
                ('model', RandomForestClassifier(**params))
            ])
            pipeline.fit(X_train, y_train)
            y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
            roc_auc = roc_auc_score(y_test, y_pred_proba)
            acuracia= accuracy_score(y_test, pipeline.predict(X_test))
            precisao= precision_score(y_test, pipeline.predict(X_test))
            recall= recall_score(y_test, pipeline.predict(X_test))

            # 3. CONDIÇÃO PARA REGISTRAR O EXPERIMENTO
           

            #podemos registrar todos os experimentos que passaram do threshold 
            if roc_auc > AUROC_THRESHOLD:
                print(f"  --> RF Trial {trial.number}: AUROC = {roc_auc:.4f} (SUPERIOR AO THRESHOLD - REGISTRANDO)")
                print(f"  --> RF Trial {trial.number}: Acurácia = {acuracia:.4f}, 'Precisão = {precisao:.4f}, Recall = {recall:.4f}")
                mlflow.log_metric("roc_auc", roc_auc)
                mlflow.sklearn.log_model(pipeline, "model_pipeline")
            else:
                print(f"  --> RF Trial {trial.number}: AUROC = {roc_auc:.4f} (INFERIOR AO THRESHOLD - DESCARTANDO)")
                  #ogar um status para saber que foi descartado
                mlflow.set_tag("status", "descartado_por_threshold")

        return roc_auc
        
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=50) # Rode 50 trials para o RF
    return study.best_trial


def optimize_svm(X_train, X_test, y_train, y_test):
    """
    Função dedicada à otimização e registro de modelos SVM.
    """
    # Define o experimento específico para o SVM
    mlflow.set_experiment("Otimizacao SVM")

    def objective(trial):
        with mlflow.start_run(nested=True):
            scaler_type = trial.suggest_categorical('scaler', ['StandardScaler', 'MinMaxScaler', 'RobustScaler'])
            params = {
                'C': trial.suggest_float('C', 1e-4, 1e3, log=True),
                'kernel': trial.suggest_categorical('kernel', ['linear', 'rbf', 'poly']),
                'probability': True,
                'gamma': trial.suggest_categorical('gamma', ['scale', 'auto']),
                'degree': trial.suggest_int('degree', 2, 5), # Apenas relevante para kernel 'poly'

                'random_state': 69
            }
            
            mlflow.log_param("model_type", "SVM")
            mlflow.log_param("scaler", scaler_type)
            mlflow.log_params(params)

            if scaler_type == 'StandardScaler': scaler = StandardScaler()
            elif scaler_type == 'RobustScaler': scaler = RobustScaler()
            else: scaler = MinMaxScaler()
            
            pipeline = Pipeline([
                ('scaler', scaler),
                ('model', SVC(**params))
            ])
            pipeline.fit(X_train, y_train)
            
            y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
            roc_auc = roc_auc_score(y_test, y_pred_proba)
            
            if roc_auc > AUROC_THRESHOLD:
                print(f"  --> SVM Trial {trial.number}: AUROC = {roc_auc:.4f} (SUPERIOR AO THRESHOLD - REGISTRANDO)")
                mlflow.log_metric("roc_auc", roc_auc)
                mlflow.sklearn.log_model(pipeline, "model_pipeline")
            else:
                print(f"  --> SVM Trial {trial.number}: AUROC = {roc_auc:.4f} (INFERIOR AO THRESHOLD - DESCARTANDO)")
                mlflow.set_tag("status", "descartado_por_threshold")
                
        return roc_auc

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=50) # Rode 50 trials para o SVM
    return study.best_trial


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_data(os.path.join('californiabin.csv'))
    
    print("--- Iniciando Otimização para RandomForest ---")
    best_rf_trial = optimize_random_forest(X_train, X_test, y_train, y_test)
    
    print("\n--- Iniciando Otimização para SVM ---")
    best_svm_trial = optimize_svm(X_train, X_test, y_train, y_test)
    
    print("\n--- Otimização Concluída ---")
    print("\nMelhor Trial de RandomForest:")
    print(f"  Valor (ROC AUC): {best_rf_trial.value:.4f}")
    print(f"  Hiperparâmetros: {best_rf_trial.params}")

    print("\nMelhor Trial de SVM:")
    print(f"  Valor (ROC AUC): {best_svm_trial.value:.4f}")
    print(f"  Hiperparâmetros: {best_svm_trial.params}")