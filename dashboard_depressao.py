import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import joblib
import json
import requests
import geopandas as gpd
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

# Carregamento do modelo
modelo = carregar_modelo()

# Configurações iniciais da página
st.set_page_config(page_title="Dashboard Depressão - PNS 2019", layout="wide")

# Logomarca
st.image("LOGO_DS.jpg")

# Navegação no menu lateral
st.sidebar.title("Navegação")
pagina = st.sidebar.radio("Ir para", [
    "🏠 Introdução",
    "🌎 Panorama Nacional",
    "💡 Estilo de Vida",
    "📝Teste Pessoal"
])


# ----------------------- Filtro Regional (válido para todo o dashboard) -----------------------
estados = {
    11: 'Rondônia', 12: 'Acre', 13: 'Amazonas', 14: 'Roraima', 15: 'Pará',
    16: 'Amapá', 17: 'Tocantins', 21: 'Maranhão', 22: 'Piauí', 23: 'Ceará',
    24: 'Rio Grande do Norte', 25: 'Paraíba', 26: 'Pernambuco', 27: 'Alagoas',
    28: 'Sergipe', 29: 'Bahia', 31: 'Minas Gerais', 32: 'Espírito Santo',
    33: 'Rio de Janeiro', 35: 'São Paulo', 41: 'Paraná', 42: 'Santa Catarina',
    43: 'Rio Grande do Sul', 50: 'Mato Grosso do Sul', 51: 'Mato Grosso',
    52: 'Goiás', 53: 'Distrito Federal'
}

df['Nome_Estado'] = df['Unidade_Federacao'].map(estados)

regioes_estados = {
    'Norte': ['Rondônia', 'Acre', 'Amazonas', 'Roraima', 'Pará', 'Amapá', 'Tocantins'],
    'Nordeste': ['Maranhão', 'Piauí', 'Ceará', 'Rio Grande do Norte', 'Paraíba', 'Pernambuco', 'Alagoas', 'Sergipe', 'Bahia'],
    'Centro-Oeste': ['Mato Grosso do Sul', 'Mato Grosso', 'Goiás', 'Distrito Federal'],
    'Sudeste': ['Minas Gerais', 'Espírito Santo', 'Rio de Janeiro', 'São Paulo'],
    'Sul': ['Paraná', 'Santa Catarina', 'Rio Grande do Sul']
}

st.sidebar.markdown("---")
st.sidebar.subheader("🔎 Filtro Regional")
regiao_selecionada = st.sidebar.selectbox("Selecione uma região:", ["Todos"] + list(regioes_estados.keys()))

if regiao_selecionada == "Todos":
    df_filtrado = df.copy()
else:
    estados_filtrados = regioes_estados[regiao_selecionada]
    df_filtrado = df[df['Nome_Estado'].isin(estados_filtrados)]
    
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
   
elif pagina == "🌎 Panorama Nacional":
    st.header("🌎 Panorama Nacional")
    st.write("Nesta seção, você verá a distribuição da depressão por sexo, raça/cor, região e estado.")

    # ----------------------- Mapeamentos ----------------------- #
    cor_map = {
        1: 'Branca',
        2: 'Preta',
        3: 'Parda',
        4: 'Amarela',
        5: 'Indígena',
    }

    map_depressao = {1: 'Com Depressão', 2: 'Sem Depressão'}

    # Aplicar novamente o filtro regional com base no nome dos estados
    if regiao_selecionada == "Todos":
        df_filtrado = df.copy()
    else:
        estados_filtrados = regioes_estados[regiao_selecionada]
        df_filtrado = df[df['Nome_Estado'].isin(estados_filtrados)]

    # Mapear rótulos
    df_filtrado['Depressao_Label'] = df_filtrado['Diagnostico_Depressao'].map(map_depressao)
    df_filtrado['Cor_Label'] = df_filtrado['Cor_Raca'].map(cor_map)

    # Filtrar pessoas com diagnóstico de depressão
    df_depressao = df_filtrado[df_filtrado['Diagnostico_Depressao'] == 1]

    # Mapa - Destaque de estados por região
    if regiao_selecionada == "Todos":
        estados_destaque = [estado for lista in regioes_estados.values() for estado in lista]
    else:
        estados_destaque = regioes_estados[regiao_selecionada]

    #------------------------------------------------------------------------------------------------------#
    todos_estados = [estado for lista in regioes_estados.values() for estado in lista]
    df_mapa = pd.DataFrame({
        'Estado': todos_estados,
        'Destaque': [1 if estado in estados_destaque else 0 for estado in todos_estados]
    })

    # GeoJSON dos estados brasileiros
    geojson_url = "https://raw.githubusercontent.com/codeforamerica/click_that_hood/master/public/data/brazil-states.geojson"
    geojson_data = json.loads(requests.get(geojson_url).text)

    fig_mapa = px.choropleth(
    df_mapa,
    geojson=geojson_data,
    locations='Estado',
    featureidkey="properties.name",
    color='Destaque',
    color_continuous_scale=[[0, "lightgray"], [1, "royalblue"]],
    scope="south america",
    title=f"Mapa do Brasil - Destaque: {regiao_selecionada if regiao_selecionada != 'Todos' else 'Todas as Regiões'}"
    )

    fig_mapa.update_geos(fitbounds="locations", visible=False)
    fig_mapa.update_coloraxes(showscale=False)  # ⬅️ Aqui remove a escala de cor
    fig_mapa.update_layout(margin={"r": 0, "t": 30, "l": 0, "b": 0})

    st.plotly_chart(fig_mapa, use_container_width=True, config={"displayModeBar": False}, key="mapa_regional")

    #------------------------------------------------------------------------------------------------------#
    st.subheader("📊 Indicadores Gerais - Diagnóstico de Depressão e Horas de Trabalho")
    df_filtrado_original = df[df['Diagnostico_Depressao'].isin([1])]
    
    # Indicador estático: Total de pessoas com depressão no Brasil
    df_brasil = df_filtrado_original[df_filtrado_original['Diagnostico_Depressao'] == 1]  # sem filtro regional
    total_brasil = len(df_brasil)

    # Indicador dinâmico: Total de pessoas com depressão na região selecionada
    total_regiao = len(df_depressao)
    titulo_regional = "👥 Total com Depressão na Região Selecionada"
    if regiao_selecionada != "Todos":
        titulo_regional += f" ({regiao_selecionada})"

    # Layout lado a lado
    col1, col2 = st.columns(2)
    with col1:
        st.metric("🇧🇷 Total de Pessoas com Depressão no Brasil", f"{total_brasil:,}".replace(",", "."))

    with col2:
        st.metric(label=titulo_regional, value=f"{total_regiao:,}".replace(",", "."))

    # Médias de horas de trabalho (com e sem depressão)
    horas_com = df_depressao['Horas_Trabalho_Semana']
    horas_com = horas_com[(horas_com.notna()) & (horas_com > 0) & (horas_com < 100)]
    media_horas_com = horas_com.mean()

    df_sem_depressao = df_filtrado[df_filtrado['Diagnostico_Depressao'] == 2]
    horas_sem = df_sem_depressao['Horas_Trabalho_Semana']
    horas_sem = horas_sem[(horas_sem.notna()) & (horas_sem > 0) & (horas_sem < 100)]
    media_horas_sem = horas_sem.mean()

    # Exibe comparativo de horas de trabalho
    col3, col4 = st.columns(2)
    with col3:
        st.metric("🕒 Média de Horas de Trabalho (Com Depressão)", f"{media_horas_com:.2f} h")
    with col4:
        st.metric("🕒 Média de Horas de Trabalho (Sem Depressão)", f"{media_horas_sem:.2f} h")

    #------------------------------------------------------------------------------------------------------#
    # ------------------------ Seletor de agrupamento ------------------------
    opcao = st.selectbox("📊 Visualizar por:", ["Estado", "Cor"], key="filtro_grafico_panorama")

    # ------------------------ Geração dos dados e gráfico dinâmico ------------------------
    if opcao == "Estado":
        dados = df_depressao['Nome_Estado'].value_counts().reset_index()
        dados.columns = ['Categoria', 'Quantidade']
        dados = dados.sort_values(by='Categoria')
        titulo = "Número de Pessoas com Diagnóstico de Depressão por Estado"
        eixo_x = "Categoria"
        labels = {'Categoria': 'Estado', 'Quantidade': 'Número de Pessoas'}

    elif opcao == "Cor":
        dados = df_depressao['Cor_Raca'].value_counts().reset_index()
        dados.columns = ['Codigo_Cor', 'Quantidade']
        dados['Categoria'] = dados['Codigo_Cor'].map(cor_map)
        dados = dados.sort_values(by='Categoria')
        titulo = "Número de Pessoas com Diagnóstico de Depressão por Cor"
        eixo_x = "Categoria"
        labels = {'Categoria': 'Cor', 'Quantidade': 'Número de Pessoas'}

    # ------------------------ Gráfico ------------------------
    fig = px.bar(
        dados,
        x=eixo_x,
        y='Quantidade',
        text='Quantidade',
        labels=labels,
        title=titulo
    )
    fig.update_traces(textposition='outside')
    fig.update_layout(xaxis_tickangle=-45)

    st.plotly_chart(fig, use_container_width=True, key="grafico_dinamico_panorama")
    #------------------------------------------------------------------------------------------------------#
    # Gráfico de pizza - Sexo
    sexo_map = {1: 'Masculino', 2: 'Feminino'}
    depressao_por_sexo = df_depressao['Sexo'].value_counts().reset_index()
    depressao_por_sexo.columns = ['Codigo_Sexo', 'Quantidade']
    depressao_por_sexo['Sexo'] = depressao_por_sexo['Codigo_Sexo'].map(sexo_map)

    fig_sexo = px.pie(
        depressao_por_sexo,
        names='Sexo',
        values='Quantidade',
        title="Comparação de Homens e Mulheres com Diagnóstico de Depressão"
    )
    fig_sexo.update_traces(textposition='inside', textinfo='percent+label')
    st.plotly_chart(fig_sexo, use_container_width=True, key="grafico_sexo")
    
elif pagina == "💡 Estilo de Vida":
    st.header("💡 Estilo de Vida")
    st.write("Comportamentos, hábitos e fatores associados à saúde mental.")
    
    df_stats = df_filtrado.copy()
    # ----------------------- Mapeamentos -----------------------
    map_depressao = {1: 'Com Depressão', 2: 'Sem Depressão'}
    map_bebida = {
        1: 'Nunca', 2: 'Menos de 1 vez/mês', 3: '1 a 3 vezes/mês',
        4: '1 vez/semana', 5: '2 a 3 vezes/semana',
        6: '4 a 6 vezes/semana', 7: 'Todos os dias'
    }
    map_excessiva = {
        1: 'Nunca', 2: 'Menos de 1 vez/mês', 3: '1 a 3 vezes/mês',
        4: '1 vez/semana', 5: '2 a 3 vezes/semana',
        6: '4 a 6 vezes/semana', 7: 'Todos os dias ou quase todos os dias'
    }
    map_esporte = {1: 'Sim', 2: 'Não'}
    map_fumo = {
        1: 'Todos os dias', 2: 'Alguns dias',
        3: 'Não fuma atualmente', 4: 'Nunca fumou'
    }
    estado_civil_map = {
        1: 'Solteiro(a)', 2: 'Casado(a)', 3: 'Separado(a)/Divorciado(a)',
        4: 'Viúvo(a)', 5: 'Outro'
    }

    # ----------------------- ⚖️ Comparativo de Peso----------------------- #
    
    # Filtra dados válidos de peso e diagnóstico
    df_peso = df_filtrado[
        df_filtrado['Peso'].between(1, 599) &
        df_filtrado['Diagnostico_Depressao'].isin([1, 2])
    ].copy()

    # Mapeia diagnóstico
    df_peso['Depressao_Label'] = df_peso['Diagnostico_Depressao'].map({1: 'Com Depressão', 2: 'Sem Depressão'})

    # Estatísticas por grupo
    estatisticas_por_grupo = df_peso.groupby('Depressao_Label')['Peso'].describe().rename(columns={
        'count': 'Total de Pessoas',
        'mean': 'Média (kg)',
        'std': 'Desvio Padrão',
        'min': 'Mínimo (kg)',
        '25%': '1º Quartil (kg)',
        '50%': 'Mediana (kg)',
        '75%': '3º Quartil (kg)',
        'max': 'Máximo (kg)'
    })

    st.subheader("⚖️ Comparativo de Estatísticas de Peso por Diagnóstico de Depressão")
    st.dataframe(estatisticas_por_grupo.style.format("{:.2f}"))

    # ----------------------- 📋 Estatísticas de Atividade Física ----------------------- #
    st.subheader("📋 Estatísticas de Atividade Física por Diagnóstico de Depressão")

    # Filtro com dados ainda numéricos
    df_validos = df_stats[
        df_stats['Frequencia_Esporte_Mes'].isin([1, 2]) &
        df_stats['Frequencia_Esporte_Seman'].between(0, 7) &
        df_stats['Diagnostico_Depressao'].isin([1, 2])
    ].copy()

    # Aplica os mapeamentos APÓS o filtro
    df_validos['Depressao_Label'] = df_validos['Diagnostico_Depressao'].map(map_depressao)
    df_validos['Praticou_3_Meses'] = df_validos['Frequencia_Esporte_Mes'].map(map_esporte)

    # Agrupamento com agregações
    tabela_stats = df_validos.groupby(['Depressao_Label', 'Praticou_3_Meses']).agg({
        'Frequencia_Esporte_Seman': ['mean', 'median', 'std', 'count']
    }).reset_index()

    # Renomeia as colunas corretamente
    tabela_stats.columns = [
        "Diagnóstico de Depressão",
        "Praticou nos Últimos 3 Meses",
        "Média de Dias por Semana",
        "Mediana de Dias por Semana",
        "Desvio Padrão",
        "Total de Pessoas"
    ]

    # Exibe a tabela formatada
    st.dataframe(
        tabela_stats.style.format({
            "Média de Dias por Semana": "{:.2f}",
            "Mediana de Dias por Semana": "{:.0f}",
            "Desvio Padrão": "{:.2f}",
            "Total de Pessoas": "{:.0f}"
        }),
        use_container_width=True
    )

    # ----------------------- 🍷 Consumo de Bebida Alcoólica -----------------------
    st.subheader("🍷 Consumo de Bebida Alcoólica")

    df_bebida = df_filtrado.copy()
    df_bebida['Depressao_Label'] = df_bebida['Diagnostico_Depressao'].map(map_depressao)
    df_bebida['Frequencia_Bebida'] = df_bebida['Frequencia_Bebida'].map(map_bebida)
    df_bebida['Frequencia_Bebida_Excessiva'] = df_bebida['Frequencia_Bebida_Excessiva'].map(map_excessiva)

    tipo_bebida = st.radio(
        "Selecione o tipo de informação:",
        ['Frequência de Consumo', 'Frequência de Bebida em Excesso'],
        horizontal=True
    )

    if tipo_bebida == 'Frequência de Consumo':
        df_consumo = df_bebida[df_bebida['Frequencia_Bebida'] != 'Nunca']  # remove "Nunca"
        fig_bebida = px.histogram(
            df_consumo,
            x='Frequencia_Bebida',
            color='Depressao_Label',
            barmode='group',
            labels={
                'Frequencia_Bebida': 'Frequência',
                'count': 'Número de Pessoas'
            },
            title="Frequência de Consumo de Bebida Alcoólica por Diagnóstico de Depressão",
            color_discrete_map={
                'Com Depressão': 'indianred',
                'Sem Depressão': 'seagreen'
            }
        )
        fig_bebida.update_layout(xaxis_title="Frequência", yaxis_title="Número de Pessoas")
        st.plotly_chart(fig_bebida, use_container_width=True)

    elif tipo_bebida == 'Frequência de Bebida em Excesso':
        df_bebida_excesso = df_bebida[df_bebida['Frequencia_Bebida_Excessiva'].notna()]
        df_bebida_excesso = df_bebida_excesso[df_bebida_excesso['Frequencia_Bebida_Excessiva'] != 'Nunca']  # remove "Nunca"
        fig_bebida_excesso = px.histogram(
            df_bebida_excesso,
            x='Frequencia_Bebida_Excessiva',
            color='Depressao_Label',
            barmode='group',
            labels={
                'Frequencia_Bebida_Excessiva': 'Frequência de Bebida em Excesso',
                'count': 'Número de Pessoas'
            },
            title="Frequência de Consumo Excessivo de Bebida Alcoólica por Diagnóstico de Depressão",
            color_discrete_map={
                'Com Depressão': 'indianred',
                'Sem Depressão': 'seagreen'
            }
        )
        fig_bebida_excesso.update_layout(xaxis_title="Frequência", yaxis_title="Número de Pessoas")
        st.plotly_chart(fig_bebida_excesso, use_container_width=True)

    # ----------------------- 🚬 Frequência de Fumo ----------------------- #
    st.subheader("🚬 Consumo de Cigarros")

    # Mapeamento das frequências de fumo
    mapa_fumo = {
        1: "Um ou mais por dia",
        2: "Um ou mais por semana",
        3: "Menos que uma vez por semana",
        4: "Menos que um por mês",
        5: "Não fuma"
    }

    # Cria cópia e aplica mapeamentos
    df_fumo = df_filtrado.copy()
    df_fumo['Depressao_Label'] = df_fumo['Diagnostico_Depressao'].map(map_depressao)
    df_fumo['Frequencia_Fumo_Label'] = df_fumo['Frequencia_Fumo'].map(mapa_fumo)

    # Filtro interativo
    tipo_fumo = st.radio(
        "Selecione o tipo de informação:",
        ['Frequência de Fumo', 'Quantidade Média por Dia'],
        horizontal=True
    )

    if tipo_fumo == 'Frequência de Fumo':
        df_fumo_freq = df_fumo[df_fumo['Frequencia_Fumo'].isin([1, 2, 3, 4])]  # Exclui "Não fuma"

        fig_fumo_freq = px.histogram(
            df_fumo_freq,
            x='Frequencia_Fumo_Label',
            color='Depressao_Label',
            barmode='group',
            labels={
                'Frequencia_Fumo_Label': 'Frequência de Fumo',
                'count': 'Número de Pessoas'
            },
            title="Frequência de Consumo de Cigarros por Diagnóstico de Depressão",
            color_discrete_map={
                'Com Depressão': 'indianred',
                'Sem Depressão': 'seagreen'
            }
        )
        fig_fumo_freq.update_layout(xaxis_title="Frequência", yaxis_title="Número de Pessoas")
        st.plotly_chart(fig_fumo_freq, use_container_width=True)

    elif tipo_fumo == 'Quantidade Média por Dia':
        df_fumo_dia = df_fumo[df_fumo['Frequencia_Fumo_Dia'].between(1, 98)]

        media_fumo_dia = df_fumo_dia.groupby('Depressao_Label')['Frequencia_Fumo_Dia'].mean().reset_index()
        media_fumo_dia.columns = ['Diagnóstico de Depressão', 'Média de Cigarros por Dia']

        fig_fumo_qtd = px.bar(
            media_fumo_dia,
            x='Diagnóstico de Depressão',
            y='Média de Cigarros por Dia',
            color='Diagnóstico de Depressão',
            text='Média de Cigarros por Dia',
            title="Quantidade Média de Cigarros por Dia por Diagnóstico de Depressão",
            color_discrete_map={
                'Com Depressão': 'indianred',
                'Sem Depressão': 'seagreen'
            }
        )
        fig_fumo_qtd.update_traces(texttemplate='%{text:.2f}', textposition='outside')
        fig_fumo_qtd.update_layout(yaxis_title="Média de Cigarros por Dia")
        st.plotly_chart(fig_fumo_qtd, use_container_width=True)
    
    # ----------------------- 💊 Uso de Medicamento (apenas com depressão) -----------------------

    # Dados
    df_remedio = df_filtrado[df_filtrado['Diagnostico_Depressao'] == 1]
    df_remedio = df_remedio[df_remedio['Medicamento_Depressao'].isin([1, 2])]
    df_remedio['Usa_Medicamento'] = df_remedio['Medicamento_Depressao'].map({1: 'Sim', 2: 'Não'})

    dados = df_remedio['Usa_Medicamento'].value_counts().reset_index()
    dados.columns = ['Uso de Medicamento', 'Quantidade']

    # Gráfico de donut
    fig = px.pie(
        dados,
        names='Uso de Medicamento',
        values='Quantidade',
        color='Uso de Medicamento',
        color_discrete_map={'Sim': 'mediumseagreen', 'Não': 'tomato'},
        hole=0.5,
        title="Distribuição de Uso de Medicamento entre Pessoas com Depressão"
    )
    fig.update_traces(textinfo='percent+label')

    st.plotly_chart(fig, use_container_width=True)

elif pagina == "📝Teste Pessoal":
    st.header("📝Teste Pessoal")
    st.write("Responda algumas perguntas para avaliar seu estado emocional.")

    st.info(
        "**Importante:** Este teste é apenas uma ferramenta educativa e de autopercepção. "
        "Ele **não substitui uma avaliação profissional** realizada por psicólogos ou psiquiatras. "
        "Se você estiver enfrentando dificuldades emocionais, considere buscar ajuda especializada."
    )

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

        st.caption(
            "**Aviso:** Este teste foi desenvolvido com base em dados populacionais e algoritmos estatísticos. "
            "Ele serve apenas como um sinal de alerta inicial e não tem valor diagnóstico. "
            "Para um diagnóstico preciso, consulte um profissional de saúde mental."
        )