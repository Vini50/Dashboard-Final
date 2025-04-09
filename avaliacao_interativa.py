import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import joblib
from collections import Counter
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from imblearn.over_sampling import RandomOverSampler


caminho_arquivo = r"pns2019_IA.csv" 
df = pd.read_csv(caminho_arquivo, sep=';', encoding='utf-8')


# Seleção das colunas do dataframe para treino da IA
X = df[["Frequencia_Problemas_Sono", "Frequencia_Problemas_Concentracao", "Frequencia_Problemas_Interesse", "Frequencia_Problemas_Alimentacao", 
        "Frequencia_Sentimento_Deprimido", "Frequencia_Sentimento_Fracasso", "Frequencia_Pensamentos_Suicidio"]]
y = df["Diagnostico_Depressao"]

# Filtro de y para manter apenas valores válidos (1 = Sim, 2 = Não)
valid_values_y = [1, 2]
y = y[y.isin(valid_values_y)]

# Filtro de X para manter apenas valores válidos (1 e 2)
valid_values_x = {col: [1, 2] for col in X.columns}
valid_indices_x = X.apply(lambda col: col.isin(valid_values_x[col.name])).all(axis=1)
X = X[valid_indices_x]
y = y.loc[X.index]  # Garante que y tenha os mesmos índices de X

# Convertendo os valores de X (1 para 0 e 4 para 1)
X = X.apply(lambda col: col.map({1: 0, 2: 1}))

# Divisão de x e y em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Aplicando SMOTE para equilibrar X e y
smote = SMOTE(sampling_strategy='auto', random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Criando e treinando o modelo
pipeline = ImbPipeline([
    ('classifier', DecisionTreeClassifier(random_state=42, max_depth=4))
])
pipeline.fit(X_train_resampled, y_train_resampled)

# Predição
y_pred = pipeline.predict(X_test)

#Utilização de técnica para identificação dos melhores hiperparâmetros
# Ajuste de Hiperparâmetros usando Grid Search
param_grid = {
    'classifier__max_depth': [3, 4, 5, 6, None],
    'classifier__min_samples_split': [2, 5, 10],
    'classifier__min_samples_leaf': [1, 2, 4],
    'classifier__criterion': ['gini', 'entropy']
}

grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
grid_search.fit(X_train_resampled, y_train_resampled)

#Treinamento do modelo com os melhores hiperparâmetros
final_model = ImbPipeline([
    ('smote', SMOTE(random_state=42)),  # Usando SMOTE para balanceamento
    ('classifier', DecisionTreeClassifier(
        random_state=42,
        criterion='entropy',  # Melhor critério
        max_depth= None,  # Melhor profundidade
        min_samples_leaf=1,  # Melhor min_samples_leaf
        min_samples_split=2  # Melhor min_samples_split
    ))
])

# Treino do modelo final
final_model.fit(X_train_resampled, y_train_resampled)

# Previsões no conjunto de teste
y_pred = final_model.predict(X_test)


def carregar_modelo():
    return final_model