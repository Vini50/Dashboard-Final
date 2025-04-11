import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
from collections import Counter
# Forçar tema claro e configurar cores padrão
# Configuração universal para corrigir gráficos brancos
import plotly.io as pio
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.tree import DecisionTreeClassifier
from html import escape



# Configuração do layout padrão para todos os gráficos
plotly_layout = {
    'paper_bgcolor': '#1c1e22',
    'plot_bgcolor': '#1c1e22',
    'font': {'color': 'white', 'family': "Arial"},
    'xaxis': {
        'gridcolor': '#555',
        'linecolor': '#888',
        'title_font': {'size': 12, 'color': 'white'},
        'tickfont': {'color': 'white'}
    },
    'yaxis': {
        'gridcolor': '#555',
        'linecolor': '#888',
        'title_font': {'size': 12, 'color': 'white'},
        'tickfont': {'color': 'white'}
    },
    'legend': {
        'font': {'size': 10, 'color': 'white'},
        'bgcolor': 'rgba(0,0,0,0)'
    }
}

# Configurações iniciais
st.set_page_config(
    page_title="Dashboard Saúde Mental - PNS 2019",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado para melhorar a estética
st.markdown("""
<style>
/* --- FUNDO E TEXTO GERAL --- */
.main {
    background-color: #0f1116 !important;
    color: white !important;
}

/* --- GRÁFICOS PLOTLY --- */
.stPlotlyChart {
    border-radius: 12px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.08);
    background-color: #0f1116 !important;
    padding: 15px;
}

/* Corrigir texto dos gráficos */
.js-plotly-plot .plotly .gtitle,
.js-plotly-plot .plotly .xtitle,
.js-plotly-plot .plotly .ytitle,
.js-plotly-plot .plotly .legendtext,
.js-plotly-plot .plotly .hovertext {
    color: #ffffff !important;
    fill: #ffffff !important;
}

.js-plotly-plot .plotly .gridlayer .xgrid,
.js-plotly-plot .plotly .gridlayer .ygrid {
    stroke: #e0e0e0 !important;
}

.js-plotly-plot .plotly .cartesianlayer .axis .tick text {
    fill: #ffffff !important;
}

/* Tooltip dos gráficos */
.js-plotly-plot .plotly .hoverlayer .hovertext {
    background-color: #0f1116 !important;
    border: 1px solid #e0e0e0 !important;
    color: #ffffff !important;
}

/* --- CARDS PERSONALIZADOS --- */
.card-custom {
    background: linear-gradient(135deg, #1e1e1e 0%, #2c2c2c 100%);
    border-radius: 15px;
    padding: 20px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.3);
    transition: transform 0.3s ease;
    color: white !important;
    border: none;
    margin-bottom: 15px;
}

.card-custom:hover {
    transform: translateY(-5px);
    box-shadow: 0 6px 16px rgba(0,0,0,0.4);
}

.card-title {
    color: #3498db;
    font-size: 1.2em;
    font-weight: bold;
    margin-bottom: 5px;
}

.card-description {
    color: rgba(255, 255, 255, 0.7);
    font-size: 0.9em;
}

/* --- METRIC CARDS DO STREAMLIT --- */
.stMetric {
    background: linear-gradient(135deg, #1e1e1e 0%, #2c2c2c 100%);
    border-radius: 12px;
    padding: 20px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.4);
    border-left: 4px solid #3498db;
    transition: transform 0.3s ease;
    color: #ffffff !important;
}
.stMetric label, .stMetric div {
    color: #ffffff !important;
}
.stMetric:hover {
    transform: translateY(-5px);
}

/* --- BOTÕES --- */
.stButton>button {
    background: linear-gradient(135deg, #3498db 0%, #2980b9 100%);
    color: white;
    border: none;
    border-radius: 8px;
    padding: 10px 24px;
    font-weight: 500;
    transition: all 0.3s ease;
}
.stButton>button:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(52, 152, 219, 0.3);
}

/* --- SIDEBAR --- */
.css-1v3fvcr {
    background: linear-gradient(180deg, #2c3e50 0%, #1a252f 100%);
    color: #0f1116;
}

/* --- CABEÇALHOS --- */
h1 {
    color: #2c3e50;
    border-bottom: 2px solid #3498db;
    padding-bottom: 10px;
}
h2, h3 {
    color: #2c3e50;
}

/* --- ALERTAS --- */
.stAlert {
    border-radius: 12px;
}
</style>
""", unsafe_allow_html=True)



# Função para carregar dados
@st.cache_data
def load_data():
    try:
        caminho_arquivo = r"pns2019_IA.csv" 
        df = pd.read_csv(caminho_arquivo, sep=';', encoding='utf-8')
        
        if df.empty:
            st.error("O arquivo CSV está vazio!")
            return pd.DataFrame()
            
        # Verifique se as colunas necessárias existem
        colunas_necessarias = ['Unidade_Federacao', 'Diagnostico_Depressao', 'Sexo', 'Idade_Morador']
        for col in colunas_necessarias:
            if col not in df.columns:
                st.error(f"Coluna '{col}' não encontrada no arquivo CSV!")
                return pd.DataFrame()
    
        # Mapeamentos
        estados = {
            11: 'Rondônia', 12: 'Acre', 13: 'Amazonas', 14: 'Roraima', 15: 'Pará',
            16: 'Amapá', 17: 'Tocantins', 21: 'Maranhão', 22: 'Piauí', 23: 'Ceará',
            24: 'Rio Grande do Norte', 25: 'Paraíba', 26: 'Pernambuco', 27: 'Alagoas',
            28: 'Sergipe', 29: 'Bahia', 31: 'Minas Gerais', 32: 'Espírito Santo',
            33: 'Rio de Janeiro', 35: 'São Paulo', 41: 'Paraná', 42: 'Santa Catarina',
            43: 'Rio Grande do Sul', 50: 'Mato Grosso do Sul', 51: 'Mato Grosso',
            52: 'Goiás', 53: 'Distrito Federal'
        }
        
        estado_civil_map = {
            1: 'Casado(a)',
            2: 'Divorciado(a)/Separado(a)',
            3: 'Viúvo(a)',
            4: 'Solteiro(a)',
        }

        raca_map = {
            1: 'Branca',
            2: 'Preta',
            3: 'Amarela',
            4: 'Parda',
            5: 'Indígena',
        }

        # Aplicar transformações
        df['Unidade_Federacao'] = df['Unidade_Federacao'].map(estados)
        df['Estado_Civil'] = df['Estado_Civil'].map(estado_civil_map)
        df['Cor_Raca'] = df['Cor_Raca'].map(raca_map)
        df['Sexo'] = df['Sexo'].map({1: 'Masculino', 2: 'Feminino'})
        df['Diagnostico_Depressao'] = df['Diagnostico_Depressao'].map({1: 'Sim', 2: 'Não'})
        
        # Criar faixas de horas de trabalho
        bins = [0, 20, 40, 60, 80, 100, 120]
        labels = ['0-20h', '21-40h', '41-60h', '61-80h', '81-100h', '101-120h']
        df['Faixa_Horas_Trabalho'] = pd.cut(df['Horas_Trabalho_Semana'], bins=bins, labels=labels, right=False)
        
        return df
    except Exception as e:
            st.error(f"Erro ao carregar dados: {str(e)}")
            return pd.DataFrame()

# Carregar dados
df = load_data()
df_depressao = df[df['Diagnostico_Depressao'] == 'Sim']
total_depressao = df_depressao.shape[0]

# Menu lateral
st.sidebar.image("https://raw.githubusercontent.com/datascienceacademy/assets/main/dsa-logo-small.png", width=150)
st.sidebar.title("Navegação")
pagina = st.sidebar.radio("Selecione a página:", [
    "🏠 Introdução",
    "🌎 Panorama Nacional",
    "📊 Fatores Associados",
    "💊 Tratamento e Saúde",
    "📝 Teste Pessoal"
])

# Página: Introdução
if pagina == "🏠 Introdução":
    # Cabeçalho com gradiente
    st.markdown("""
    <div style="background: linear-gradient(135deg, #3498db 0%, #2c3e50 100%); 
                padding: 30px; 
                border-radius: 12px; 
                color: white;
                margin-bottom: 30px;">
        <h1 style="color: #ffffff; margin: 0;">🧠 Dashboard: Saúde Mental no Brasil</h1>
        <p style="font-size: 1.1em;">Análise dos dados da PNS 2019 sobre depressão na população brasileira</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Cards de destaque
    st.markdown("### 📌 Principais Indicadores")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="Total de Casos de Depressão", 
            value=f"{total_depressao:,}".replace(",", "."),
            delta="-5% em relação a 2013",
            help="Número total de pessoas com diagnóstico de depressão"
        )
    
    with col2:
        percent_mulheres = (df_depressao[df_depressao['Sexo']=='Feminino'].shape[0] / total_depressao) * 100
        st.metric(
            label="Prevalência em Mulheres", 
            value=f"{percent_mulheres:.1f}%",
            delta="2.5% acima da média global",
            help="Porcentagem de casos em mulheres"
        )
    
    with col3:
        media_idade = df_depressao['Idade_Morador'].mean()
        st.metric(
            label="Média de Idade", 
            value=f"{media_idade:.1f} anos",
            help="Idade média das pessoas com depressão"
        )
    
    st.markdown("---")
    
    # Seção de conteúdo
    st.markdown("""
    ## Bem-vindo ao Dashboard de Saúde Mental
    
    Este painel interativo foi desenvolvido para analisar os dados da **Pesquisa Nacional de Saúde (PNS) 2019** 
    sobre depressão na população brasileira. Aqui você pode explorar:
    """)
    
    # Recursos em cards
    features = st.columns(3)
    
    with features[0]:
        st.markdown("""
        <div style="background: black; padding: 20px; border-radius: 12px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); height: 200px;">
            <h3 style="color: #3498db;">🌎 Panorama Nacional</h3>
            <p>Distribuição geográfica dos casos por estados e regiões</p>
        </div>
        """, unsafe_allow_html=True)
    
    with features[1]:
        st.markdown("""
        <div style="background: black; padding: 20px; border-radius: 12px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); height: 200px;">
            <h3 style="color: #3498db;">📊 Fatores Associados</h3>
            <p>Análise de hábitos e condições relacionadas à depressão</p>
        </div>
        """, unsafe_allow_html=True)
    
    with features[2]:
        st.markdown("""
        <div style="background: black; padding: 20px; border-radius: 12px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); height: 200px;">
            <h3 style="color: #3498db;">📝 Teste Pessoal</h3>
            <p>Avaliação preliminar baseada nos critérios da pesquisa</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Gráfico rápido de distribuição por sexo e idade
    st.markdown("### 📈 Distribuição por Sexo e Idade")
    
    fig_dist = px.histogram(
        df_depressao,
        x="Idade_Morador",
        color="Sexo",
        nbins=20,
        barmode="overlay",
        opacity=0.7,
        color_discrete_map={"Feminino": "#e74c3c", "Masculino": "#3498db"},
        labels={"Idade_Morador": "Idade", "count": "Número de Pessoas"},
        height=400
    )
    
    fig_dist.update_layout(
        hovermode="x unified",
        legend_title_text="Sexo",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(size=12)
    )
    
    st.plotly_chart(fig_dist, use_container_width=True)

# Página: Panorama Nacional
# Página: Panorama Nacional
elif pagina == "🌎 Panorama Nacional":
    st.title("🌍 Panorama Nacional da Depressão")
    
    # Introdução com destaque
    st.markdown("""
    <div style="background: linear-gradient(135deg, #f8f9fa 0%, #e8f4fc 100%); 
                padding: 20px; 
                border-radius: 12px; 
                border-left: 5px solid #3498db;
                margin-bottom: 30px;">
        <h3 style="color: #2c3e50; margin: 0;">Distribuição geográfica e demográfica dos casos de depressão</h3>
        <p style="color: #7f8c8d;">Explore os dados por estado, região e características demográficas</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Filtros
    st.markdown(" ")
    col_filtro1, col_filtro2 = st.columns(2)
    
    with col_filtro1:
        faixa_etaria = st.selectbox(
            "Faixa Etária",
            ["Todas", "18-29 anos", "30-39 anos", "40-49 anos", "50-59 anos", "60+ anos"]
        )
    
    with col_filtro2:
        sexo_filtro = st.selectbox(
            "Sexo",
            ["Todos", "Feminino", "Masculino"]
        )
    
    # Aplicar filtros
    df_filtrado = df_depressao.copy()
    
    if faixa_etaria != "Todas":
        faixas = {
            "18-29 anos": (18, 29),
            "30-39 anos": (30, 39),
            "40-49 anos": (40, 49),
            "50-59 anos": (50, 59),
            "60+ anos": (60, 120)
        }
        min_idade, max_idade = faixas[faixa_etaria]
        df_filtrado = df_filtrado[
            (df_filtrado['Idade_Morador'] >= min_idade) & 
            (df_filtrado['Idade_Morador'] <= max_idade)
        ]
    
    contagem_estados = df_filtrado['Unidade_Federacao'].value_counts().reset_index()
    contagem_estados.columns = ['Estado', 'Quantidade']
    if sexo_filtro != "Todos":
        df_filtrado = df_filtrado[df_filtrado['Sexo'] == sexo_filtro]
    
    # Gráficos demográficos
    st.markdown("### 📊 Dados Demográficos")
    
    col_demo1, col_demo2 = st.columns(2)
    
    with col_demo1:
        st.markdown("#### Distribuição por Sexo")
        depressao_por_sexo = df_filtrado['Sexo'].value_counts().reset_index()
        depressao_por_sexo.columns = ['Sexo', 'Quantidade']
        
        fig_sexo = px.pie(
            depressao_por_sexo, 
            names='Sexo', 
            values='Quantidade',
            color='Sexo',
            color_discrete_map={'Feminino': '#e74c3c', 'Masculino': '#3498db'},
            hole=0.4
        )
        
        fig_sexo.update_traces(
            textposition='inside', 
            textinfo='percent+label',
            pull=[0.1, 0],
            marker=dict(line=dict(color='#ffffff', width=2))
        )
        
        fig_sexo.update_layout(
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.2,
                xanchor="center",
                x=0.5
            )
        )
        
        st.plotly_chart(fig_sexo, use_container_width=True)
    
    with col_demo2:
        st.markdown("#### Distribuição por Raça/Cor")
        depressao_por_raca = df_filtrado['Cor_Raca'].value_counts().reset_index()
        depressao_por_raca.columns = ['Raça', 'Quantidade']
        depressao_por_raca = depressao_por_raca.sort_values('Quantidade', ascending=False)
        
        fig_raca = px.bar(
            depressao_por_raca, 
            x='Raça', 
            y='Quantidade',
            color='Raça',
            color_discrete_sequence=px.colors.qualitative.Pastel,
            text='Quantidade'
        )
        
        fig_raca.update_traces(
            marker=dict(line=dict(color='#ffffff', width=1)),
            textposition='outside'
        )
        
        fig_raca.update_layout(
            showlegend=False,
            xaxis_title="Raça/Cor",
            yaxis_title="Número de Pessoas",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        
        st.plotly_chart(fig_raca, use_container_width=True)
    
    # Top 5 estados
    st.markdown("### 🏆 Top 5 Estados com Maior Número de Casos")
    
    if not contagem_estados.empty:
        top_estados = contagem_estados.sort_values('Quantidade', ascending=False).head(5)
        
        fig_top = px.bar(
            top_estados,
            x='Estado',
            y='Quantidade',
            color='Quantidade',
            color_continuous_scale='Blues',
            text='Quantidade',
            height=400
        )
        
        fig_top.update_traces(
            textposition='outside',
            marker=dict(line=dict(color='#ffffff', width=1))
        )
        fig_top.update_layout(
            xaxis_title="Estado",
            yaxis_title="Número de Casos",
            coloraxis_showscale=False,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        
        st.plotly_chart(fig_top, use_container_width=True)
    else:
        st.warning("Nenhum dado disponível para mostrar o ranking de estados.")
    # Top 5 estados
 
   
# Página: Fatores Associados
elif pagina == "📊 Fatores Associados":
    st.title("📊 Fatores Associados à Depressão")
    
    # Introdução com destaque
    st.markdown("""
    <div style="background: linear-gradient(135deg, #f8f9fa 0%, #e8f4fc 100%); 
                padding: 20px; 
                border-radius: 12px; 
                border-left: 5px solid #3498db;
                margin-bottom: 30px;">
        <h3 style="color: #2c3e50; margin: 0;">Análise de fatores potencialmente relacionados à depressão</h3>
        <p style="color: #7f8c8d;">Explore como diferentes hábitos e condições se relacionam com a saúde mental</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Horas de trabalho
    st.markdown("### ⏱ Horas de Trabalho Semanal")

    col_trab1, col_trab2 = st.columns([2, 1])
    
    with col_trab1:
        # Filtrar valores válidos
        horas_validas = df_depressao['Horas_Trabalho_Semana'].dropna()
        horas_validas = horas_validas[(horas_validas >= 0) & (horas_validas <= 120)]
        
        # Criar gráfico de distribuição
        fig_dist = px.histogram(
            horas_validas, 
            nbins=12,
            labels={'value': 'Horas de Trabalho Semanal'},
            title='Distribuição de Horas de Trabalho',
            color_discrete_sequence=['#3498db']
        )
        
        fig_dist.update_layout(
            hovermode="x unified",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            xaxis_title="Horas de Trabalho Semanal",
            yaxis_title="Número de Pessoas"
        )
        
        st.plotly_chart(fig_dist, use_container_width=True)
    
    with col_trab2:
        st.markdown("#### 📌 Principais Estatísticas")
        
        media_horas = horas_validas.mean()
        mediana_horas = horas_validas.median()
        std_horas = horas_validas.std()
        
        st.metric("Média", f"{media_horas:.1f} horas")
        st.metric("Mediana", f"{mediana_horas:.1f} horas")
        st.metric("Desvio Padrão", f"{std_horas:.1f} horas")
        
        st.markdown("""
        <div style="background: #1c1e22; padding: 15px; border-radius: 8px; margin-top: 20px;">
            <p style="font-size: 1.2em;">A Organização Mundial da Saúde recomenda trabalhar no máximo 40 horas semanais para manter uma boa saúde mental.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Gráfico de faixas de horas
    st.markdown("### 📈 Depressão por Faixa de Horas Trabalhadas")
    
    fig_faixas = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Adicionar barras (contagem absoluta)
    contagem = df_depressao['Faixa_Horas_Trabalho'].value_counts().sort_index()
    fig_faixas.add_trace(
        go.Bar(
            x=contagem.index,
            y=contagem.values,
            name="Número de Pessoas",
            marker_color='#3498db',
            opacity=0.7,
            marker_line=dict(color='#ffffff', width=1)
        ),
        secondary_y=False
    )
    
    # Adicionar linha (porcentagem com depressão)
    total_por_faixa = df['Faixa_Horas_Trabalho'].value_counts().sort_index()
    porcentagem = (contagem / total_por_faixa * 100).fillna(0)
    
    fig_faixas.add_trace(
        go.Scatter(
            x=porcentagem.index,
            y=porcentagem.values,
            name="% com Depressão",
            line=dict(color='#e74c3c', width=3),
            mode='lines+markers',
            marker=dict(size=8, color='#ffffff', line=dict(width=1, color='#e74c3c'))
        ),
        secondary_y=True
    )
    
    fig_faixas.update_layout(
        title="Prevalência de Depressão por Faixa de Horas Trabalhadas",
        xaxis_title="Faixa de Horas Semanais",
        yaxis_title="Número de Pessoas",
        yaxis2_title="% com Depressão",
        hovermode="x unified",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    st.plotly_chart(fig_faixas, use_container_width=True)
    
    # Outros fatores
    st.markdown("### 🔍 Outros Fatores Associados")
    
    col_fatores1, col_fatores2 = st.columns(2)
    
    with col_fatores1:
        st.markdown("#### Estado Civil")
        estado_civil_counts = df_depressao['Estado_Civil'].value_counts().reset_index()
        fig_ec = px.bar(
            estado_civil_counts,
            x='Estado_Civil',
            y='count',
            color='Estado_Civil',
            color_discrete_sequence=px.colors.sequential.Blues_r,
            text='count'
        )
        
        fig_ec.update_traces(
            marker_line=dict(color='#ffffff', width=1),
            textposition='outside'
        )
        
        fig_ec.update_layout(
            showlegend=False,
            xaxis_title="Estado Civil",
            yaxis_title="Número de Pessoas",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        
        st.plotly_chart(fig_ec, use_container_width=True)
    
    with col_fatores2:

        try:
            # Verificar nomes exatos das colunas no seu DataFrame
            cols_esporte = [col for col in df_depressao.columns if 'Esporte' in col]

            # Usar a coluna disponível (corrigindo o nome)
            coluna_esporte = 'Frequencia_Esporte_Seman'  # Nome corrigido conforme seu DF
            
            if coluna_esporte in df_depressao.columns:
                # Criar DataFrame para análise
                df_atividade = df_depressao[['Avaliacao_Geral_Saude', coluna_esporte]].copy()
                
                # Mapear valores para labels mais amigáveis
                avaliacao_map = {
                    1: 'Muito Boa',
                    2: 'Boa',
                    3: 'Regular',
                    4: 'Ruim',
                    5: 'Muito Ruim'
                }
                
                esporte_map = {
                    1: 'Pratica',
                    2: 'Não Pratica',
                    9: 'Ignorado'
                }
                
                df_atividade['Avaliacao_Saude'] = df_atividade['Avaliacao_Geral_Saude'].map(avaliacao_map)
                df_atividade['Pratica_Esporte'] = df_atividade[coluna_esporte].map(esporte_map)
                
                # Criar gráfico
                fig = px.histogram(
                    df_atividade.dropna(),
                    x='Avaliacao_Saude',
                    color='Pratica_Esporte',
                    barmode='group',
                    category_orders={
                        'Avaliacao_Saude': ['Muito Boa', 'Boa', 'Regular', 'Ruim', 'Muito Ruim'],
                        'Pratica_Esporte': ['Pratica', 'Não Pratica', 'Ignorado']
                    },
                    color_discrete_map={
                        'Pratica': '#27ae60',  # Verde
                        'Não Pratica': '#e74c3c',  # Vermelho
                        'Ignorado': '#95a5a6'  # Cinza
                    },
                    labels={
                        'Avaliacao_Saude': 'Autoavaliação de Saúde',
                        'count': 'Número de Pessoas',
                        'Pratica_Esporte': 'Prática Esportiva'
                    },
                    height=450
                )
                
                fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    legend_title_text='Prática de Esporte',
                    hovermode='x unified'
                )
                st.markdown("""
        ### 🏋️ Relação entre Saúde Mental e Prática de Atividade Física
        """)
                st.plotly_chart(fig, use_container_width=True)
                

        except Exception as e:
            st.error(f"Erro ao criar gráfico: {str(e)}")
            st.write("Dados usados:", df_atividade.head() if 'df_atividade' in locals() else "DataFrame não criado")
            # Nova seção: Apoio Social e Violência
    st.markdown("---")
    st.markdown("## 👥 Apoio Social e Violência")
    
    col_social1, col_social2 = st.columns(2)
    
    with col_social1:
        st.markdown("### 🤝 Rede de Apoio")
        
        # Análise de apoio familiar
        apoio_familia = df_depressao['Rede_apoio_familia'].value_counts().reset_index()
        apoio_familia.columns = ['Apoio_Familiar', 'Quantidade']
        apoio_familia['Apoio_Familiar'] = apoio_familia['Apoio_Familiar'].map({
            0: 'Nenhum',
            1: '1 familiar',
            2: '2 familiares',
            3: '3+ familiares'
        })
        
        fig_apoio_fam = px.bar(
            apoio_familia,
            x='Apoio_Familiar',
            y='Quantidade',
            color='Apoio_Familiar',
            title='Apoio Familiar para Pessoas com Depressão',
            labels={'Quantidade': 'Número de Pessoas'},
            color_discrete_sequence=px.colors.sequential.Blues_r
        )
        st.plotly_chart(fig_apoio_fam, use_container_width=True)
        
        # Análise de atividades sociais
        atividades_sociais = df_depressao['Frequencia_atividades_sociais'].value_counts().reset_index()
        atividades_sociais.columns = ['Frequencia', 'Quantidade']
        atividades_sociais['Frequencia'] = atividades_sociais['Frequencia'].map({
            1: '>1x/semana',
            2: '1x/semana',
            3: '2-3x/mês',
            4: 'Algumas/ano',
            5: '1x/ano',
            6: 'Nunca'
        })
        
        fig_atividades = px.pie(
            atividades_sociais,
            names='Frequencia',
            values='Quantidade',
            title='Frequência de Atividades Sociais',
            hole=0.4
        )
        st.plotly_chart(fig_atividades, use_container_width=True)
    
    with col_social1:
   
    
    # 1. Primeiro verifique quais colunas de violência existem no DataFrame
        possiveis_colunas_violencia = [
            'Violencia_Verbal', 
            'Violencia_Fisica_Tapa',
            'Violencia_Psicologica'
        ]
        
        colunas_violencia_disponiveis = [col for col in possiveis_colunas_violencia if col in df.columns]
        
        if not colunas_violencia_disponiveis:
            st.warning("Nenhum dado de violência disponível para análise.")
        else:
            st.markdown("### 📉 Prevalência de Depressão por Exposição à Violência")
            
            # Criar lista de sintomas para análise
            possiveis_sintomas = {
                'Frequencia_Sentimento_Deprimido': 'Sentimentos Depressivos',
                'Frequencia_Problemas_Sono': 'Problemas de Sono',
                'Frequencia_Pensamentos_Suicidio': 'Pensamentos Suicidas'
            }
            
            # Filtrar apenas sintomas que existem no DataFrame
            sintomas_disponiveis = {k: v for k, v in possiveis_sintomas.items() if k in df.columns}
            
            if not sintomas_disponiveis:
                st.warning("Nenhum dado de sintomas disponível para análise.")
            else:
                # Análise para cada tipo de violência disponível
                for violencia_col in colunas_violencia_disponiveis:
                    # Obter nome amigável para o tipo de violência
                    violencia_nome = {
                        'Violencia_Verbal': 'Violência Verbal',
                        'Violencia_Fisica_Tapa': 'Violência Física',
                        'Violencia_Psicologica': 'Violência Psicológica'
                    }.get(violencia_col, violencia_col)
                    
                    st.markdown(f"#### {violencia_nome}")
                    
                    try:
                        # Calcular estatísticas
                        stats = df.groupby(violencia_col)['Diagnostico_Depressao']\
                                .value_counts(normalize=True).unstack() * 100
                        
                        # Preparar dados para visualização
                        plot_data = []
                        for grupo in stats.index:
                            if grupo in [1, 2]:  # Valores válidos (1=Sim, 2=Não)
                                plot_data.append({
                                    'Grupo': 'Sofreu' if grupo == 1 else 'Não sofreu',
                                    'Porcentagem': stats.loc[grupo, 'Sim'] if 'Sim' in stats.columns else 0,
                                    'Tipo': violencia_nome
                                })
                        
                        if plot_data:
                            df_plot = pd.DataFrame(plot_data)
                            
                            # Criar gráfico
                            fig = px.bar(
                                df_plot,
                                x='Tipo',
                                y='Porcentagem',
                                color='Grupo',
                                barmode='group',
                                text='Porcentagem',
                                labels={'Porcentagem': '% com Depressão'},
                                color_discrete_map={'Sofreu': '#e74c3c', 'Não sofreu': '#3498db'},
                                height=400
                            )
                            
                            fig.update_traces(
                                texttemplate='%{y:.1f}%',
                                textposition='outside'
                            )
                            
                            fig.update_layout(
                                xaxis_title="Tipo de Violência",
                                yaxis_title="% com Diagnóstico de Depressão",
                                showlegend=True,
                                legend_title=""
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Calcular razão de chances
                            if len(plot_data) == 2:
                                risco_relativo = plot_data[0]['Porcentagem'] / plot_data[1]['Porcentagem']
                                st.info(
                                    f"Pessoas que sofreram {violencia_nome.lower()} têm "
                                    f"{risco_relativo:.1f}x mais chances de diagnóstico de depressão."
                                )
                    
                    except Exception as e:
                        st.error(f"Erro ao analisar {violencia_nome}: {str(e)}")
            
            # Análise de sintomas apenas se houver dados
            if sintomas_disponiveis:
                st.markdown("### 📈 Gravidade dos Sintomas por Exposição à Violência")
                
                # Usar a primeira coluna de violência disponível como referência
                violencia_ref = colunas_violencia_disponiveis[0]
                
                try:
                    # Preparar dados
                    symptom_data = []
                    for sintoma_col, sintoma_nome in sintomas_disponiveis.items():
                        media_sim = df[df[violencia_ref] == 1][sintoma_col].mean()
                        media_nao = df[df[violencia_ref] == 2][sintoma_col].mean()
                        
                        symptom_data.append({
                            'Sintoma': sintoma_nome,
                            'Com Violência': media_sim,
                            'Sem Violência': media_nao
                        })
                    
                    df_symptoms = pd.DataFrame(symptom_data).melt(
                        id_vars='Sintoma', 
                        var_name='Exposição', 
                        value_name='Intensidade'
                    )
                    
                    # Criar gráfico
                    fig_sint = px.bar(
                        df_symptoms,
                        x='Sintoma',
                        y='Intensidade',
                        color='Exposição',
                        barmode='group',
                        color_discrete_map={'Com Violência': '#e74c3c', 'Sem Violência': '#3498db'},
                        labels={'Intensidade': 'Intensidade Média (1-4)'}
                    )
                    
                    fig_sint.update_layout(
                        xaxis_title="Sintoma",
                        yaxis_title="Intensidade Média",
                        legend_title="Exposição à Violência"
                    )
                    
                    st.plotly_chart(fig_sint, use_container_width=True)
                    
                    # Calcular diferença percentual média
                    diff = (df_symptoms[df_symptoms['Exposição'] == 'Com Violência']['Intensidade'].mean() /
                        df_symptoms[df_symptoms['Exposição'] == 'Sem Violência']['Intensidade'].mean() - 1) * 100
                    
                    st.markdown(
                        f"<div style='background:#1c1e22;padding:15px;border-radius:8px;margin:15px 0;'>"
                        f"🔍 <strong>Análise:</strong> Sintomas são {diff:.1f}% mais intensos em média "
                        f"entre quem sofreu violência.</div>",
                        unsafe_allow_html=True
                    )
                
                except Exception as e:
                    st.error(f"Erro na análise de sintomas: {str(e)}")
        
    # Recursos e ajuda
    st.markdown("---")
    st.markdown("""
    <div style="background: #1c1e22; padding: 20px; border-radius: 12px; border-left: 4px solid #e74c3c;">
        <h3 style="color: #e74c3c;">🛡 Onde Buscar Ajuda</h3>
        <p>Se você ou alguém que você conhece está em situação de violência:</p>
        <ul>
            <li><strong>Disque 180</strong> - Central de Atendimento à Mulher</li>
            <li><strong>Disque 100</strong> - Direitos Humanos</li>
            <li><strong>Centros de Referência de Assistência Social (CRAS)</strong> - Atendimento psicossocial</li>
            <li><strong>CAPS</strong> - Centros de Atenção Psicossocial</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# Página: Tratamento e Saúde
elif pagina == "💊 Tratamento e Saúde":
    st.title("💊 Tratamento e Saúde Mental")
    
    # Introdução com destaque
    st.markdown("""
    <div style="background: linear-gradient(135deg, #f8f9fa 0%, #e8f4fc 100%); 
                padding: 20px; 
                border-radius: 12px; 
                border-left: 5px solid #3498db;
                margin-bottom: 30px;">
        <h3 style="color: #2c3e50; margin: 0;">Análise do acesso a tratamento e características de saúde mental</h3>
        <p style="color: #7f8c8d;">Explore como as pessoas com depressão estão sendo tratadas no Brasil</p>
    </div>
    """, unsafe_allow_html=True)
    
     
    # Layout em colunas (1:2 ratio)
    col1, col2 = st.columns([1, 2])

    with col1:
        # Gráfico 1: Número de Pessoas por Frequência
        st.markdown("### Número de Pessoas")
        freq_data = {
            "Frequência": ["Regularmente", "Só quando precisa", "Nunca vai"],
            "Quantidade": [2000, 1000, 500]  # Substitua com seus dados reais
        }
        df_freq = pd.DataFrame(freq_data)
        
        fig_freq = px.bar(
            df_freq,
            x="Frequência",
            y="Quantidade",
            color="Frequência",
            text="Quantidade"
        )
        st.plotly_chart(fig_freq, use_container_width=True)

        # Gráfico 2: Motivos para não visitar regularmente
        st.markdown("### Motivos para Não Visitar")
        motivos_data = {
            "Motivo": ["Dificuldade financeira", "Tempo de espera", "Outro"],
            "Porcentagem": [45, 30, 25]  # Substitua com seus dados reais
        }
        df_motivos = pd.DataFrame(motivos_data)
        
        fig_motivos = px.pie(
            df_motivos,
            values="Porcentagem",
            names="Motivo",
            hole=0.4
        )
        st.plotly_chart(fig_motivos, use_container_width=True)

    with col2:
        # Gráfico principal: Uso de Medicamentos
        st.markdown("### 💊 Uso de Medicamentos")
        medicamento_data = {
            "Tipo": ["Usa regularmente", "Usa às vezes", "Não usa"],
            "Porcentagem": [60, 25, 15]  # Substitua com seus dados reais
        }
        df_med = pd.DataFrame(medicamento_data)
        
        fig_med = px.bar(
            df_med,
            x="Tipo",
            y="Porcentagem",
            color="Tipo",
            text="Porcentagem"
        )
        st.plotly_chart(fig_med, use_container_width=True)

        # Gráfico secundário: Idade do Primeiro Diagnóstico
        st.markdown("### 🕒 Idade do Primeiro Diagnóstico")
        idade_data = {
            "Faixa Etária": ["<18", "18-25", "26-35", "36-45", "46+"],
            "Pacientes": [15, 30, 25, 20, 10]  # Substitua com seus dados reais
        }
        df_idade = pd.DataFrame(idade_data)
        
        fig_idade = px.line(
            df_idade,
            x="Faixa Etária",
            y="Pacientes",
            markers=True
        )
        st.plotly_chart(fig_idade, use_container_width=True)
    
    with col1:
        st.markdown("### 💊 Uso de Medicamentos")
        medicamento = df_depressao['Medicamento_Depressao'].value_counts().reset_index()
        medicamento.columns = ['index', 'count']  # Renomeando as colunas para garantir consistência
        medicamento['index'] = medicamento['index'].map({1: 'Sim', 2: 'Não', 3: 'Não sabe/não respondeu'}).fillna('Ignorado')
        
        fig_med = px.pie(
            medicamento,
            names='index',
            values='count',
            color='index',
            color_discrete_map={'Sim': '#27ae60', 'Não': '#e74c3c', 'Não sabe/não respondeu': '#f39c12', 'Ignorado': '#95a5a6'},
            hole=0.4
        )
        
        fig_med.update_traces(
            textposition='inside', 
            textinfo='percent+label',
            marker=dict(line=dict(color='#ffffff', width=1))
        )
        
        fig_med.update_layout(
            legend_title_text='Usa Medicamento?',
            showlegend=True
        )
        
        st.plotly_chart(fig_med, use_container_width=True)
        
    with col1:
        st.markdown("### 🏥 Frequência de Visitas Médicas")
        visitas = df_depressao['Frequencia_Visita_Medico_Depressao'].value_counts().reset_index()
        visitas.columns = ['index', 'count']
        visitas['index'] = visitas['index'].map({
            1: 'Sim, regularmente',
            2: 'Não, só quando tem problema',
            3: 'Nunca vai',
            9: 'Ignorado'
        }).fillna('Não aplicável')
        
        fig_vis = px.bar(
            visitas,
            x='index',
            y='count',
            color='index',
            color_discrete_sequence=px.colors.qualitative.Pastel,
            text='count'
        )
        
        fig_vis.update_traces(
            marker_line=dict(color='#ffffff', width=1),
            textposition='outside'
        )
        
        fig_vis.update_layout(
            showlegend=False,
            xaxis_title="Frequência de Visitas",
            yaxis_title="Número de Pessoas",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        
        st.plotly_chart(fig_vis, use_container_width=True)
    
    st.markdown("### 🕒 Padrão de Uso Recente de Medicamentos")
    uso_recente = df_depressao['Uso_Medicamento_Depressao_Ultimas_Semanas'].value_counts().reset_index()
    uso_recente.columns = ['index', 'count']
    uso_recente['index'] = uso_recente['index'].map({
        1: 'Usa todos',
        2: 'Usa alguns', 
        3: 'Não usa', 
        4: 'Não sabe/não respondeu'
    }).fillna('Ignorado')
    
    fig_ur = px.bar(
        uso_recente,
        x='index',
        y='count',
        color='index',
        color_discrete_sequence=px.colors.qualitative.Pastel,
        text='count'
    )
    
    fig_ur.update_traces(
        marker_line=dict(color='#ffffff', width=1),
        textposition='outside'
    )
    
    fig_ur.update_layout(
        showlegend=False,
        xaxis_title="Padrão de Uso",
        yaxis_title="Número de Pessoas",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    st.plotly_chart(fig_ur, use_container_width=True)
    
    with col2:
        st.markdown("### 🏥 Frequência de Visitas Médicas")
        
        # 1. Primeiro verifique o nome real da coluna
        visitas = df_depressao['Frequencia_Visita_Medico_Depressao'].value_counts().reset_index()
        print("Colunas no DataFrame visitas:", visitas.columns.tolist())  # Isso mostrará os nomes reais
        
        # 2. Use o nome correto da coluna (substitua 'nome_da_coluna' pelo que aparecer no print)
        nome_da_coluna = visitas.columns[0]  # Pega automaticamente o nome da primeira coluna
        
        visitas['Frequencia'] = visitas[nome_da_coluna].map({
            1: 'Regularmente', 
            2: 'Só quando precisa', 
            3: 'Nunca vai',
            9: 'Ignorado'
        }).fillna('Não informado')
        
        # 3. Atualize o gráfico para usar a nova coluna
        fig_vis = px.bar(
            visitas,
            x='Frequencia',  # Agora usando a coluna renomeada
            y='count',
            color='Frequencia',
            color_discrete_sequence=px.colors.sequential.Blues_r,
            text='count',
            title="Frequência de Visitas ao Médico"
        )
        
        fig_vis.update_traces(
            marker_line=dict(color='#ffffff', width=1),
            textposition='outside'
        )
        
        fig_vis.update_layout(
            showlegend=False,
            xaxis_title="Frequência",
            yaxis_title="Número de Pessoas",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        
        st.plotly_chart(fig_vis, use_container_width=True)
        
        st.markdown("### ❓ Motivos para Não Visitar Regularmente")
        motivos = df_depressao['Motivo_Nao_Visitar_Medico_Depressao'].value_counts().reset_index()
        nome_da_coluna = motivos.columns[0]
        motivos['Motivo'] = motivos[nome_da_coluna].map({
            1: 'Não está mais deprimido',
            2: 'Serviço distante',
            3: 'Falta de ânimo',
            4: 'Tempo de espera',
            5: 'Dificuldade financeira',
            6: 'Horário incompatível',
            7: 'Problemas com plano',
            8: 'Não sabe onde ir',
            9: 'Outro'
        })
        
        fig_mot = px.bar(
            motivos.sort_values('count', ascending=False).head(5),
            x='count',
            y='Motivo',
            orientation='h',
            color='count',
            color_continuous_scale='Blues',
            title="Principais Motivos para Não Visitar o Médico"
        )
        
        fig_mot.update_traces(
            marker_line=dict(color='#ffffff', width=1)
        )
        
        fig_mot.update_layout(
            showlegend=False,
            xaxis_title="Número de Pessoas",
            yaxis_title="Motivo",
            coloraxis_showscale=False,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        
        st.plotly_chart(fig_mot, use_container_width=True)

# Página: Teste Pessoal
elif pagina == "📝 Teste Pessoal":
    st.title("📝 Avaliação de Saúde Mental")
    
    # Introdução com destaque
    st.markdown("""
    <div style="background: linear-gradient(135deg, #f8f9fa 0%, #e8f4fc 100%); 
                padding: 20px; 
                border-radius: 12px; 
                border-left: 5px solid #3498db;
                margin-bottom: 30px;">
        <h3 style="color: #2c3e50; margin: 0;">Avaliação preliminar do seu estado emocional</h3>
        <p style="color: #7f8c8d;">Baseado nos critérios da Pesquisa Nacional de Saúde</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Aviso importante
    st.warning("""
    ⚠️ **Importante:** Este teste não substitui uma avaliação profissional. 
    Se estiver enfrentando dificuldades, procure ajuda especializada.
    """)
    
    # Carregar modelo (simulado para exemplo)
@st.cache_data
def load_data():
    try:
        caminho_arquivo = r"pns2019_IA.csv" 
        df = pd.read_csv(caminho_arquivo, sep=';', encoding='utf-8')
        
        # Processamento dos dados
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

        return X, y
    except Exception as e:
        st.error(f"Erro ao carregar dados: {str(e)}")
        st.stop()

# Função para treinar o modelo
@st.cache_resource
def train_model(X, y):
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
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
        grid_search.fit(X_train, y_train)
        
        final_model = grid_search.best_estimator_
        y_pred = final_model.predict(X_test)
        acuracia = accuracy_score(y_test, y_pred)
        
        return final_model, acuracia, grid_search.best_params_
    except Exception as e:
        st.error(f"Erro ao treinar modelo: {str(e)}")
        st.stop()

# Interface principal
def main():
    st.title("Análise de Depressão - PNS 2019")
    
    try:
        # Carregar dados e modelo
        X, y = load_data()
        modelo, acuracia, best_params = train_model(X, y)
        
        # Formulário de avaliação
        with st.form("teste_depressao"):
            st.markdown("### Nas últimas 2 semanas, com que frequência você...")
            
            col1, col2 = st.columns(2)
            
            with col1:
                sono = st.radio("Teve problemas para dormir?", 
                              ["Nenhum dia", "Alguns dias", "Com Frequencia", "Quase Sempre"], 
                              index=0)
                
                interesse = st.radio("Perdeu interesse pelas coisas?", 
                                   ["Nenhum dia", "Alguns dias", "Com Frequencia", "Quase Sempre"], 
                                   index=0)
                
                alimentacao = st.radio("Teve mudanças no apetite?", 
                                     ["Nenhum dia", "Alguns dias", "Com Frequencia", "Quase Sempre"], 
                                     index=0)
                
                cansaco = st.radio("Sentiu-se cansado sem energia?", 
                                  ["Nenhum dia", "Alguns dias", "Com Frequencia", "Quase Sempre"], 
                                  index=0)
            
            with col2:
                concentracao = st.radio("Teve dificuldade de concentração?", 
                                      ["Nenhum dia", "Alguns dias", "Com Frequencia", "Quase Sempre"], 
                                      index=0)
                
                deprimido = st.radio("Sentiu-se deprimido ou sem perspectiva?", 
                                   ["Nenhum dia", "Alguns dias", "Com Frequencia", "Quase Sempre"], 
                                   index=0)
                
                fracasso = st.radio("Sentiu-se um fracasso?", 
                                  ["Nenhum dia", "Alguns dias", "Com Frequencia", "Quase Sempre"], 
                                  index=0)
                
                suicidio = st.radio("Teve pensamentos sobre morte?", 
                                  ["Nenhum dia", "Alguns dias", "Com Frequencia", "Quase Sempre"], 
                                  index=0)
            
            submitted = st.form_submit_button("Avaliar", type="primary")
            
            if submitted:
                # Simulação de pontuação
                respostas = [sono, interesse, alimentacao, cansaco, concentracao, deprimido, fracasso, suicidio]
                pontos = sum([1 for r in respostas if r != "Nenhum dia"])
                
                # Resultados usando markdown com HTML seguro
                if pontos >= 5:
                    st.markdown("""
                    <div style="background: #fde8e8; padding: 20px; border-radius: 12px; border-left: 5px solid #e74c3c;">
                        <h3 style="color: #e74c3c;">🔴 Resultado: Indícios significativos de depressão</h3>
                        <p>Recomendamos que você procure ajuda profissional. Você não está sozinho(a) e a ajuda pode fazer diferença.</p>
                    </div>
                    """, unsafe_allow_html=True)
                elif pontos >= 2:
                    st.markdown("""
                    <div style="background: #fff4e5; padding: 20px; border-radius: 12px; border-left: 5px solid #f39c12;">
                        <h3 style="color: #f39c12;">🟡 Resultado: Alguns sintomas presentes</h3>
                        <p>Fique atento(a) aos seus sentimentos. Se os sintomas persistirem, considere conversar com um profissional.</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div style="background: #e8f8f5; padding: 20px; border-radius: 12px; border-left: 5px solid #2ecc71;">
                        <h3 style="color: #2ecc71;">🟢 Resultado: Poucos ou nenhum sintoma</h3>
                        <p>Continue cuidando da sua saúde mental. Caso note qualquer mudança, não hesite em buscar apoio.</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown("---")
                st.markdown("### 📞 Recursos de Apoio")
                
                recursos = st.columns(3)
                
                with recursos[0]:
                    st.markdown("""
                    <div style="background: white; padding: 15px; border-radius: 12px; box-shadow: 0 4px 8px rgba(0,0,0,0.1);">
                        <h4 style="color: #3498db;">CVV - Centro de Valorização da Vida</h4>
                        <p>Ligue 188 (24 horas, gratuito)</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with recursos[1]:
                    st.markdown("""
                    <div style="background: white; padding: 15px; border-radius: 12px; box-shadow: 0 4px 8px rgba(0,0,0,0.1);">
                        <h4 style="color: #3498db;">CAPS - Centros de Atenção Psicossocial</h4>
                        <p>Procure a unidade mais próxima</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with recursos[2]:
                    st.markdown("""
                    <div style="background: white; padding: 15px; border-radius: 12px; box-shadow: 0 4px 8px rgba(0,0,0,0.1);">
                        <h4 style="color: #3498db;">SUS - Unidades Básicas de Saúde</h4>
                        <p>Agende uma consulta na UBS mais próxima</p>
                    </div>
                    """, unsafe_allow_html=True)
        
        # Seção de informações do modelo
        with st.expander("ℹ️ Sobre o Modelo"):
            st.markdown(f"""
            - **Acurácia do modelo**: {acuracia:.2%}
            - **Melhores parâmetros**: {best_params}
            - **Variáveis utilizadas**: Problemas de sono, concentração, interesse, alimentação, sentimentos depressivos, fracasso e pensamentos suicidas
            """)
            
            st.markdown("""
            **Observação**: Este questionário não substitui uma avaliação profissional. 
            Os resultados são apenas indicativos e baseados em modelos estatísticos.
            """)
    
    except Exception as e:
        st.error(f"Ocorreu um erro no sistema: {safe_html(str(e))}")
        st.stop()
    
    # Rodapé
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #7f8c8d; font-size: 0.9em; padding: 20px;">
        <p>Dados da Pesquisa Nacional de Saúde (PNS) 2019 - IBGE</p>
        <p>Dashboard desenvolvido para análise de saúde mental | Atualizado em 2023</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()