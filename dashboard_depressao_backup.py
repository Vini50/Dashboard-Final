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
from avaliacao_interativa import carregar_modelo


# Caminho do arquivo de dados
caminho_arquivo = r"pns2019_IA.csv" 
df = pd.read_csv(caminho_arquivo, sep=';', encoding='utf-8')

modelo = carregar_modelo()

# Renomear os códigos dos estados (caso queira exibir depois)
estados = {
        11: 'Rondônia', 12: 'Acre', 13: 'Amazonas', 14: 'Roraima', 15: 'Pará',
        16: 'Amapá', 17: 'Tocantins', 21: 'Maranhão', 22: 'Piauí', 23: 'Ceará',
        24: 'Rio Grande do Norte', 25: 'Paraíba', 26: 'Pernambuco', 27: 'Alagoas',
        28: 'Sergipe', 29: 'Bahia', 31: 'Minas Gerais', 32: 'Espírito Santo',
        33: 'Rio de Janeiro', 35: 'São Paulo', 41: 'Paraná', 42: 'Santa Catarina',
        43: 'Rio Grande do Sul', 50: 'Mato Grosso do Sul', 51: 'Mato Grosso',
        52: 'Goiás', 53: 'Distrito Federal'
    }
df['Unidade_Federacao'] = df['Unidade_Federacao'].map(estados)

# Map de estado civil
estado_civil_map = {
    1: 'Casado(a)',
    2: 'Divorciado(a) ou desquitado(a) ou separado(a) judicialmente',
    3: 'Viúvo(a)',
    4: 'Solteiro(a)',
}

# Map de raças para um formato legível
raca_map = {
    1: 'Branca',
    2: 'Preta',
    3: 'Parda',
    4: 'Amarela',
    5: 'Indígena',
}

# Map motivos para não visitar o médico
motivo_nao_visitar_map = {
    1: 'Não está mais deprimido',
    2: 'O serviço de saúde é distante ou tem dificuldade de transporte',
    3: 'Não tem ânimo',
    4: 'O tempo de espera no serviço de saúde é muito grande',
    5: 'Tem dificuldades financeiras',
    6: 'O horário de funcionamento do serviço de saúde é incompatível com suas atividades de trabalho ou domésticas',
    7: 'Não conseguiu marcar consulta pelo plano de saúde',
    8: 'Não sabe quem procurar ou aonde ir',
    9: 'Outro'
}

# Filtrar apenas as pessoas com diagnóstico de depressão
df_depressao = df[df['Diagnostico_Depressao'] == 1]
total_depressao = df_depressao.shape[0]

# Configurações iniciais da página
st.set_page_config(page_title="Dashboard Depressão - PNS 2019", layout="wide")

st.image("LOGO_DS.jpg")

# Menu lateral
st.sidebar.title("Navegação")
pagina = st.sidebar.radio("Ir para", [
    "🏠 Introdução",
    "🌎 Panorama Nacional",
    "💡 Estilo de Vida",
    "📝Teste Pessoal"
])
st.write("Página selecionada:", pagina)
# Página: Introdução
if pagina == "🏠 Introdução":
    st.markdown("""
    # 🧠 Dashboard: Depressão no Brasil - PNS 2019

    Bem-vindo ao dashboard interativo com dados da **Pesquisa Nacional de Saúde (PNS) 2019** sobre **depressão** no Brasil.

    ### 🎯 Objetivo
    Apresentar um panorama completo e acessível sobre os principais aspectos relacionados à depressão na população brasileira.

    ### 🧩 Contexto
    A depressão é um transtorno mental comum, que afeta milhões de pessoas no mundo todo. Analisar esses dados pode ajudar a entender padrões e fatores associados, contribuindo para políticas públicas e conscientização.

    ### 🧭 Como navegar
    Use o menu lateral para explorar os dados por diferentes temas:
    - Distribuição Nacional
    - Estilo de Vida
    - Avaliação Interativa

    ---
    """)
   

# Página: Panorama Nacional
elif pagina == "🌎 Panorama Nacional":
    st.header("🌎 Panorama Nacional")
    st.write("Nesta seção, você verá a distribuição da depressão por sexo, raça/cor, região e estado.")
    
    #------------------------------------------------------------------------------------------------------#
    
    # Exibir o número total de pessoas com depressão
    st.metric(label="Total de Pessoas com Diagnóstico de Depressão", value=total_depressao)
    
    #------------------------------------------------------------------------------------------------------#
    
   # Filtrar valores válidos: remover NaN, negativos ou valores fora de faixa
    horas_validas = df_depressao['Horas_Trabalho_Semana']
    horas_validas = horas_validas[(horas_validas.notna()) & (horas_validas > 0) & (horas_validas < 100)]

    # Calcular a média apenas com valores válidos
    media_horas = horas_validas.mean()

    # Exibir a estatística formatada
    st.metric("🕒 Média de horas de trabalho semanal (com depressão)", f"{media_horas:.2f} horas")

    #------------------------------------------------------------------------------------------------------#
    
    # Contar número de pessoas com depressão por estado
    depressao_por_estado = df_depressao['Unidade_Federacao'].value_counts().reset_index()
    depressao_por_estado.columns = ['Estado', 'Quantidade']

    # Ordenar os estados pelo nome (opcional, para facilitar leitura no gráfico)
    depressao_por_estado = depressao_por_estado.sort_values(by='Estado')

    # Criar gráfico de barras com Plotly
    fig = px.bar(
    depressao_por_estado,
    x='Estado',
    y='Quantidade',
    text='Quantidade',
    labels={'Quantidade': 'Número de Pessoas'},
    title="Número de Pessoas com Diagnóstico de Depressão por Estado"
    )
    fig.update_traces(textposition='outside')
    fig.update_layout(xaxis_tickangle=-45)

    st.plotly_chart(fig, use_container_width=True)
    
    #------------------------------------------------------------------------------------------------------#

    # Contar o número de pessoas com depressão por raça
    depressao_por_raca = df_depressao['Cor_Raca'].value_counts().reset_index()
    depressao_por_raca.columns = ['Raça', 'Quantidade']

    depressao_por_raca['Raça'] = depressao_por_raca['Raça'].map(raca_map)
  
    # Criar gráfico de barras com a quantidade de pessoas com depressão por cor
    fig_raca = px.bar(depressao_por_raca, x='Raça', y='Quantidade', text='Quantidade', 
                  labels={'Quantidade': 'Número de Pessoas'}, title="Número de Pessoas com Diagnóstico de Depressão por Cor")
    fig_raca.update_traces(textposition='outside')

    st.plotly_chart(fig_raca)
    
    #------------------------------------------------------------------------------------------------------#
    
    # Contar o número de pessoas com depressão por sexo
    depressao_por_sexo = df_depressao['Sexo'].value_counts().reset_index()
    depressao_por_sexo.columns = ['Sexo', 'Quantidade']
    depressao_por_sexo['Sexo'] = depressao_por_sexo['Sexo'].map({1: 'Masculino', 2: 'Feminino'})
    
    # Criar gráfico de pizza comparando quantidade de homens e mulheres com depressão
    fig_sexo = px.pie(depressao_por_sexo, names='Sexo', values='Quantidade', title="Comparação de Homens e Mulheres com Diagnóstico de Depressão")
    st.plotly_chart(fig_sexo)
    
    #------------------------------------------------------------------------------------------------------#
# Página: Estilo de Vida
elif pagina == "💡 Estilo de Vida":
    st.header("💡 Estilo de Vida")
    st.write("Comportamentos, hábitos e fatores associados à saúde mental.")
    
    #------------------------------------------------------------------------------------------------------#
    
    #Filtrar apenas respostas válidas (1 = Sim, 2 = Não)
    df_remedio = df_depressao[df_depressao['Medicamento_Depressao'].isin([1, 2])]

    # Mapear as respostas para texto
    depressao_com_remedio = df_remedio['Medicamento_Depressao'].map({1: 'Sim', 2: 'Não'}).value_counts().reset_index()
    depressao_com_remedio.columns = ['Resposta', 'Quantidade']

    # Criar gráfico de pizza
    fig_remedio_pizza = px.pie(
    depressao_com_remedio,
    names='Resposta',
    values='Quantidade',
    title="Uso de Medicamento entre Pessoas com Depressão",
    color='Resposta',
    color_discrete_map={'Sim': 'mediumseagreen', 'Não': 'tomato'}
    )
    fig_remedio_pizza.update_traces(textposition='inside', textinfo='percent+label')

    # Exibir o gráfico
    st.plotly_chart(fig_remedio_pizza, use_container_width=True)
    
    #------------------------------------------------------------------------------------------------------#
    # Contar o número de pessoas com depressão por estado civil
    depressao_por_estado_civil = df_depressao['Estado_Civil'].map(estado_civil_map).value_counts().reset_index()
    depressao_por_estado_civil.columns = ['Estado Civil', 'Quantidade']

    # Criar gráfico de barras com a quantidade de pessoas com depressão por estado civil
    fig_estado_civil = px.bar(depressao_por_estado_civil, x='Estado Civil', y='Quantidade', text='Quantidade', 
                          labels={'Quantidade': 'Número de Pessoas'}, title="Número de Pessoas com Diagnóstico de Depressão por Estado Civil")
    fig_estado_civil.update_traces(textposition='outside')

    st.plotly_chart(fig_estado_civil)
    #------------------------------------------------------------------------------------------------------#     
elif pagina == "📝Teste Pessoal":
    st.header("📝Teste Pessoal")
    st.write("Responda algumas perguntas para avaliar seu estado emocional.")

    modelo = carregar_modelo()  # Carrega o modelo treinado

    # Exemplo de perguntas (simples, estilo rádio)
    col1, col2 = st.columns(2)

    with col1:
        sono = st.radio("Você tem tido problemas para dormir?", ["Não", "Sim"])
        concentracao = st.radio("Dificuldade de concentração?", ["Não", "Sim"])
        interesse = st.radio("Perdeu interesse pelas coisas?", ["Não", "Sim"])
        alimentacao = st.radio("Mudança no apetite?", ["Não", "Sim"])

    with col2:
        deprimido = st.radio("Tem se sentido deprimido?", ["Não", "Sim"])
        fracasso = st.radio("Sensação de fracasso?", ["Não", "Sim"])
        suicidio = st.radio("Pensamentos suicidas?", ["Não", "Sim"])

    # Mapeamento de respostas
    respostas = {
        "Frequencia_Problemas_Sono": 1 if sono == "Sim" else 0,
        "Frequencia_Problemas_Concentracao": 1 if concentracao == "Sim" else 0,
        "Frequencia_Problemas_Interesse": 1 if interesse == "Sim" else 0,
        "Frequencia_Problemas_Alimentacao": 1 if alimentacao == "Sim" else 0,
        "Frequencia_Sentimento_Deprimido": 1 if deprimido == "Sim" else 0,
        "Frequencia_Sentimento_Fracasso": 1 if fracasso == "Sim" else 0,
        "Frequencia_Pensamentos_Suicidio": 1 if suicidio == "Sim" else 0
    }

    if st.button("Avaliar"):
        input_df = pd.DataFrame([respostas])
        pred = modelo.predict(input_df)[0]

        if pred == 1:
            st.warning("⚠️ Indícios de depressão foram detectados. Considere buscar apoio profissional.")
        else:
            st.success("✅ Não foram detectados indícios de depressão. Continue cuidando da sua saúde mental.")