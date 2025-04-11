import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score  # Importação adicionada aqui
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

# Configuração inicial do Streamlit
st.set_page_config(page_title="Dashboard Depressão", layout="wide")
st.title("Análise de Depressão - PNS 2019")

@st.cache_data
def load_data():
    caminho_arquivo = r"pns2019_IA.csv" 
    return pd.read_csv(caminho_arquivo, sep=';', encoding='utf-8')

try:
    df = load_data()
    
    # Processamento dos dados (seção 1)
    with st.expander("🔍 Pré-processamento dos Dados"):
        X = df[["Frequencia_Problemas_Sono", "Frequencia_Problemas_Concentracao", 
                "Frequencia_Problemas_Interesse", "Frequencia_Problemas_Alimentacao", 
                "Frequencia_Sentimento_Deprimido", "Frequencia_Sentimento_Fracasso", 
                "Frequencia_Pensamentos_Suicidio"]]
        y = df["Diagnostico_Depressao"]

        # Filtros
        valid_values_y = [1, 2]
        y = y[y.isin(valid_values_y)]
        valid_values_x = {col: [1, 2] for col in X.columns}
        valid_indices_x = X.apply(lambda col: col.isin(valid_values_x[col.name])).all(axis=1)
        X = X[valid_indices_x]
        y = y.loc[X.index]
        X = X.apply(lambda col: col.map({1: 0, 2: 1}))

        st.success(f"Dados pré-processados: {X.shape[0]} amostras válidas")

    # Divisão dos dados (seção 2)
    with st.expander("✂️ Divisão Treino/Teste"):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        st.write(f"Treino: {X_train.shape[0]} amostras | Teste: {X_test.shape[0]} amostras")

    # Modelagem (seção 3)
    with st.expander("🤖 Treinamento do Modelo"):
        pipeline = ImbPipeline([
            ('smote', SMOTE(random_state=42)),
            ('classifier', DecisionTreeClassifier(random_state=42))
        ])

        param_grid = {
            'classifier__max_depth': [3, 4, 5, 6, None],
            'classifier__min_samples_split': [2, 5, 10],
            'classifier__min_samples_leaf': [1, 2, 4],
            'classifier__criterion': ['gini', 'entropy']
        }

        grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
        with st.spinner('Otimizando hiperparâmetros...'):
            grid_search.fit(X_train, y_train)
        
        final_model = grid_search.best_estimator_
        
        # Cálculo da acurácia
        y_pred = final_model.predict(X_test)
        acuracia = accuracy_score(y_test, y_pred)
        
        st.success(f"""
        Modelo treinado com sucesso!
        - Melhores parâmetros: {grid_search.best_params_}
        - Acurácia: {acuracia:.2%}
        """)

    # Função para carregar o modelo
    def carregar_modelo():
        return final_model

except Exception as e:
    st.error(f"🚨 Ocorreu um erro: {str(e)}")
    st.stop()

# Seção de visualização (seção 4)
with st.expander("📊 Métricas de Desempenho"):
    st.subheader("Matriz de Confusão")
    # Adicione aqui a visualização da matriz de confusão
    # Exemplo: plot_confusion_matrix(final_model, X_test, y_test)
    
    st.subheader("Outras Métricas")
    st.write(f"Acurácia no conjunto de teste: {acuracia:.2%}")