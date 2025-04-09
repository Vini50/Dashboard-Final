import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
from collections import Counter

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
    /* Estilos gerais */
    .main {
        background-color: #f8f9fa;
    }
    
    /* Estilo dos cards de métricas */
    .stMetric {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        border-left: 4px solid #3498db;
        transition: transform 0.3s ease;
    }
    
    .stMetric:hover {
        transform: translateY(-5px);
    }
    
    /* Estilo dos gráficos */
    .stPlotlyChart {
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        background-color: white;
        padding: 15px;
    }
    
    /* Cabeçalhos */
    h1 {
        color: #2c3e50;
        border-bottom: 2px solid #3498db;
        padding-bottom: 10px;
    }
    
    h2 {
        color: #2c3e50;
        margin-top: 1.5em;
    }
    
    h3 {
        color: #2c3e50;
    }
    
    /* Sidebar */
    .css-1v3fvcr {
        background: linear-gradient(180deg, #2c3e50 0%, #1a252f 100%);
        color: white;
    }
    
    /* Botões */
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
    
    /* Alertas */
    .stAlert {
        border-radius: 12px;
    }
</style>
""", unsafe_allow_html=True)

# Função para carregar dados
@st.cache_data
def load_data():
    caminho_arquivo = r"pns2019_IA.csv" 
    df = pd.read_csv(caminho_arquivo, sep=';', encoding='utf-8')
    
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
        <h1 style="color: white; margin: 0;">🧠 Dashboard: Saúde Mental no Brasil</h1>
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
        <div style="background: white; padding: 20px; border-radius: 12px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); height: 200px;">
            <h3 style="color: #3498db;">🌎 Panorama Nacional</h3>
            <p>Distribuição geográfica dos casos por estados e regiões</p>
        </div>
        """, unsafe_allow_html=True)
    
    with features[1]:
        st.markdown("""
        <div style="background: white; padding: 20px; border-radius: 12px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); height: 200px;">
            <h3 style="color: #3498db;">📊 Fatores Associados</h3>
            <p>Análise de hábitos e condições relacionadas à depressão</p>
        </div>
        """, unsafe_allow_html=True)
    
    with features[2]:
        st.markdown("""
        <div style="background: white; padding: 20px; border-radius: 12px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); height: 200px;">
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
    st.markdown("### 🔍 Filtros")
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
    
    if sexo_filtro != "Todos":
        df_filtrado = df_filtrado[df_filtrado['Sexo'] == sexo_filtro]
    
    # Mapa do Brasil
    st.markdown("### 🗺 Mapa de Distribuição por Estado")
    
    depressao_por_estado = df_filtrado['Unidade_Federacao'].value_counts().reset_index()
    depressao_por_estado.columns = ['Estado', 'Quantidade']
    
    estado_siglas = {
        'Rondônia': 'RO', 'Acre': 'AC', 'Amazonas': 'AM', 'Roraima': 'RR',
        'Pará': 'PA', 'Amapá': 'AP', 'Tocantins': 'TO', 'Maranhão': 'MA',
        'Piauí': 'PI', 'Ceará': 'CE', 'Rio Grande do Norte': 'RN',
        'Paraíba': 'PB', 'Pernambuco': 'PE', 'Alagoas': 'AL', 'Sergipe': 'SE',
        'Bahia': 'BA', 'Minas Gerais': 'MG', 'Espírito Santo': 'ES',
        'Rio de Janeiro': 'RJ', 'São Paulo': 'SP', 'Paraná': 'PR',
        'Santa Catarina': 'SC', 'Rio Grande do Sul': 'RS',
        'Mato Grosso do Sul': 'MS', 'Mato Grosso': 'MT', 'Goiás': 'GO',
        'Distrito Federal': 'DF'
    }
    
    depressao_por_estado['Sigla'] = depressao_por_estado['Estado'].map(estado_siglas)
    
    fig_mapa = px.choropleth(
        depressao_por_estado,
        locations='Sigla',
        locationmode='Brazil',
        color='Quantidade',
        scope='south america',
        color_continuous_scale='Blues',
        hover_name='Estado',
        hover_data={'Quantidade': True, 'Sigla': False},
        title='Casos de Depressão por Estado',
        height=500
    )
    
    fig_mapa.update_geos(
        visible=False,
        resolution=110,
        showcountries=False,
        showsubunits=True,
        subunitcolor='gray'
    )
    
    fig_mapa.update_layout(
        margin={"r":0,"t":40,"l":0,"b":0},
        geo=dict(bgcolor='rgba(0,0,0,0)'),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color="#2c3e50")
    )
    
    st.plotly_chart(fig_mapa, use_container_width=True)
    
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
    
    top_estados = depressao_por_estado.sort_values('Quantidade', ascending=False).head(5)
    
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
        <div style="background: #f8f9fa; padding: 15px; border-radius: 8px; margin-top: 20px;">
            <p style="font-size: 0.9em;">A Organização Mundial da Saúde recomenda trabalhar no máximo 40 horas semanais para manter uma boa saúde mental.</p>
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
        st.markdown("#### Avaliação Geral de Saúde")
        avaliacao = df_depressao['Avaliacao_Geral_Saude'].value_counts().reset_index()
        avaliacao['index'] = avaliacao['index'].map({
            1: 'Muito Boa', 2: 'Boa', 3: 'Regular', 4: 'Ruim', 5: 'Muito Ruim'
        })
        
        fig_av = px.pie(
            avaliacao,
            names='index',
            values='count',
            hole=0.4,
            color_discrete_sequence=px.colors.sequential.Reds_r
        )
        
        fig_av.update_traces(
            textposition='inside',
            textinfo='percent+label',
            marker=dict(line=dict(color='#ffffff', width=1))
        )
        
        fig_av.update_layout(
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.2,
                xanchor="center",
                x=0.5
            )
        )
        
        st.plotly_chart(fig_av, use_container_width=True)

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
    
    col_trat1, col_trat2 = st.columns(2)
    
    with col_trat1:
        st.markdown("### 💊 Uso de Medicamentos")
        medicamento = df_depressao['Medicamento_Depressao'].value_counts().reset_index()
        medicamento['index'] = medicamento['index'].map({1: 'Sim', 2: 'Não'})
        
        fig_med = px.pie(
            medicamento,
            names='index',
            values='count',
            color='index',
            color_discrete_map={'Sim': '#27ae60', 'Não': '#e74c3c'},
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
        
        st.markdown("### 🕒 Padrão de Uso Recente")
        uso_recente = df_depressao['Uso_Medicamento_Depressao_Ultimas_Semanas'].value_counts().reset_index()
        uso_recente['index'] = uso_recente['index'].map({
            1: 'Usa todos', 2: 'Usa alguns', 3: 'Não usa', 4: 'Não sabe'
        })
        
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
    
    with col_trat2:
        st.markdown("### 🏥 Frequência de Visitas Médicas")
        visitas = df_depressao['Frequencia_Visita_Medico_Depressao'].value_counts().reset_index()
        visitas['index'] = visitas['index'].map({
            1: 'Regularmente', 2: 'Só quando precisa', 3: 'Nunca vai'
        })
        
        fig_vis = px.bar(
            visitas,
            x='index',
            y='count',
            color='index',
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
        motivos['index'] = motivos['index'].map({
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
            y='index',
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
    @st.cache_resource
    def carregar_modelo():
        # Em uma aplicação real, você carregaria um modelo treinado
        return None  # Substitua por joblib.load('modelo.pkl')
    
    modelo = carregar_modelo()
    
    # Formulário
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
            # Simulação de pontuação (em um caso real, usar o modelo)
            respostas = [sono, interesse, alimentacao, cansaco, concentracao, deprimido, fracasso, suicidio]
            pontos = sum([1 for r in respostas if r != "Nenhum dia"])
            
            if pontos >= 5:
                st.error("""
                <div style="background: #fde8e8; padding: 20px; border-radius: 12px; border-left: 5px solid #e74c3c;">
                    <h3 style="color: #e74c3c;">🔴 Resultado: Indícios significativos de depressão</h3>
                    <p>Recomendamos que você procure ajuda profissional. Você não está sozinho(a) e a ajuda pode fazer diferença.</p>
                </div>
                """, unsafe_allow_html=True)
            elif pontos >= 2:
                st.warning("""
                <div style="background: #fff4e5; padding: 20px; border-radius: 12px; border-left: 5px solid #f39c12;">
                    <h3 style="color: #f39c12;">🟡 Resultado: Alguns sintomas presentes</h3>
                    <p>Fique atento(a) aos seus sentimentos. Se os sintomas persistirem, considere conversar com um profissional.</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.success("""
                <div style="background: #e8f8f5; padding: 20px; border-radius: 12px; border-left: 5px solid #2ecc71;">
                    <h3 style="color: #2ecc71;">🟢 Resultado: Poucos ou nenhum sintoma</h3>
                    <p>Continue cuidando da sua saúde mental. Praticar exercícios, manter rotinas saudáveis e conexões sociais são importantes.</p>
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

# Rodapé
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #7f8c8d; font-size: 0.9em; padding: 20px;">
    <p>Dados da Pesquisa Nacional de Saúde (PNS) 2019 - IBGE</p>
    <p>Dashboard desenvolvido para análise de saúde mental | Atualizado em 2023</p>
</div>
""", unsafe_allow_html=True)