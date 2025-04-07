import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import joblib
from collections import Counter

# Configurações da página
st.set_page_config(
    page_title="Dashboard Depressão - PNS 2019", 
    layout="wide",
    page_icon="🧠"
)

# CSS personalizado
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .stMetric {
        background-color: white;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .stPlotlyChart {
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .css-1v3fvcr {
        padding: 2rem 1rem;
    }
    h1, h2, h3 {
        color: #2c3e50;
    }
</style>
""", unsafe_allow_html=True)

# Carregar dados
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
    
    # Criar faixas de horas de trabalho
    bins = [0, 20, 40, 60, 80, 100, 120]
    labels = ['0-20h', '21-40h', '41-60h', '61-80h', '81-100h', '101-120h']
    df['Faixa_Horas_Trabalho'] = pd.cut(df['Horas_Trabalho_Semana'], bins=bins, labels=labels, right=False)
    
    return df

df = load_data()
df_depressao = df[df['Diagnostico_Depressao'] == 1]
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
    st.title("🧠 Dashboard: Depressão no Brasil - PNS 2019")
    st.markdown("""
    <div style="background-color:#e8f4fc;padding:20px;border-radius:10px;">
        <h3 style="color:#2c3e50;">Bem-vindo ao dashboard interativo sobre depressão no Brasil</h3>
        <p>Este painel utiliza dados da <b>Pesquisa Nacional de Saúde (PNS) 2019</b> para analisar a distribuição 
        e fatores associados à depressão na população brasileira.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric(label="Total de Pessoas com Depressão", value=f"{total_depressao:,}".replace(",", "."))
        
        st.markdown("""
        ### 📌 Principais Recursos:
        - Visualização geográfica dos casos
        - Análise por características demográficas
        - Fatores de risco associados
        - Teste interativo de avaliação
        """)
    
    with col2:
        st.image("https://img.freepik.com/free-vector/mental-health-concept-illustration_114360-9963.jpg", 
                caption="Saúde Mental é Fundamental")
        
    st.markdown("---")
    
    st.markdown("""
    ### 📊 Destaques Iniciais
    """)
    
    cols = st.columns(3)
    
    with cols[0]:
        st.metric("Mulheres com Depressão", 
                value=f"{df_depressao[df_depressao['Sexo']=='Feminino'].shape[0]:,}".replace(",", "."),
                help="Total de mulheres com diagnóstico de depressão")
    
    with cols[1]:
        media_idade = df_depressao['Idade_Morador'].mean()
        st.metric("Média de Idade", 
                value=f"{media_idade:.1f} anos",
                help="Média de idade das pessoas com depressão")
    
    with cols[2]:
        st.metric("Uso de Medicamentos", 
                value=f"{df_depressao[df_depressao['Medicamento_Depressao']==1].shape[0]:,}".replace(",", "."),
                help="Pessoas que usam medicamentos para depressão")

# Página: Panorama Nacional
elif pagina == "🌎 Panorama Nacional":
    st.title("🌍 Panorama Nacional da Depressão")
    st.markdown("""
    <div style="background-color:#e8f4fc;padding:15px;border-radius:10px;">
        Distribuição geográfica e demográfica dos casos de depressão no Brasil.
    </div>
    """, unsafe_allow_html=True)
    
    # Mapa do Brasil
    st.markdown("### Mapa de Distribuição por Estado")
    
    depressao_por_estado = df_depressao['Unidade_Federacao'].value_counts().reset_index()
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
        title='Casos de Depressão por Estado'
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
        height=500
    )
    
    st.plotly_chart(fig_mapa, use_container_width=True)
    
    # Gráficos demográficos
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Distribuição por Sexo")
        depressao_por_sexo = df_depressao['Sexo'].value_counts().reset_index()
        depressao_por_sexo.columns = ['Sexo', 'Quantidade']
        
        fig_sexo = px.pie(
            depressao_por_sexo, 
            names='Sexo', 
            values='Quantidade',
            color='Sexo',
            color_discrete_map={'Feminino': '#3498db', 'Masculino': '#2ecc71'},
            hole=0.4
        )
        fig_sexo.update_traces(
            textposition='inside', 
            textinfo='percent+label',
            pull=[0.1, 0]
        )
        st.plotly_chart(fig_sexo, use_container_width=True)
    
    with col2:
        st.markdown("### Distribuição por Raça/Cor")
        depressao_por_raca = df_depressao['Cor_Raca'].value_counts().reset_index()
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
        fig_raca.update_layout(showlegend=False)
        st.plotly_chart(fig_raca, use_container_width=True)

# Página: Fatores Associados
elif pagina == "📊 Fatores Associados":
    st.title("📊 Fatores Associados à Depressão")
    st.markdown("""
    <div style="background-color:#e8f4fc;padding:15px;border-radius:10px;">
        Análise de fatores potencialmente relacionados à depressão na população brasileira.
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### Horas de Trabalho Semanal")
    
    # Filtrar valores válidos
    horas_validas = df_depressao['Horas_Trabalho_Semana'].dropna()
    horas_validas = horas_validas[(horas_validas >= 0) & (horas_validas <= 120)]
    
    # Criar gráfico de distribuição
    fig_dist = px.histogram(
        horas_validas, 
        nbins=12,
        labels={'value': 'Horas de Trabalho Semanal'},
        title='Distribuição de Horas de Trabalho entre Pessoas com Depressão'
    )
    st.plotly_chart(fig_dist, use_container_width=True)
    
    # Gráfico de faixas de horas
    st.markdown("### Depressão por Faixa de Horas Trabalhadas")
    
    fig_faixas = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Adicionar barras (contagem absoluta)
    contagem = df_depressao['Faixa_Horas_Trabalho'].value_counts().sort_index()
    fig_faixas.add_trace(
        go.Bar(
            x=contagem.index,
            y=contagem.values,
            name="Número de Pessoas",
            marker_color='#3498db',
            opacity=0.6
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
            mode='lines+markers'
        ),
        secondary_y=True
    )
    
    fig_faixas.update_layout(
        title="Prevalência de Depressão por Faixa de Horas Trabalhadas",
        xaxis_title="Faixa de Horas Semanais",
        yaxis_title="Número de Pessoas",
        yaxis2_title="% com Depressão",
        hovermode="x unified"
    )
    
    st.plotly_chart(fig_faixas, use_container_width=True)
    
    # Outros fatores
    st.markdown("### Outros Fatores Associados")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Estado Civil**")
        estado_civil_counts = df_depressao['Estado_Civil'].value_counts().reset_index()
        fig_ec = px.bar(
            estado_civil_counts,
            x='Estado_Civil',
            y='count',
            color='Estado_Civil',
            color_discrete_sequence=px.colors.sequential.Blues_r
        )
        st.plotly_chart(fig_ec, use_container_width=True)
    
    with col2:
        st.markdown("**Avaliação Geral de Saúde**")
        avaliacao = df_depressao['Avaliacao_Geral_Saude'].value_counts().reset_index()
        avaliacao['index'] = avaliacao['index'].map({
            1: 'Muito Boa', 2: 'Boa', 3: 'Regular', 4: 'Ruim', 5: 'Muito Ruim'
        })
        fig_av = px.pie(
            avaliacao,
            names='index',
            values='count',
            hole=0.4
        )
        st.plotly_chart(fig_av, use_container_width=True)

# Página: Tratamento e Saúde
elif pagina == "💊 Tratamento e Saúde":
    st.title("💊 Tratamento e Saúde Mental")
    st.markdown("""
    <div style="background-color:#e8f4fc;padding:15px;border-radius:10px;">
        Análise do acesso a tratamento e características de saúde mental.
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Uso de Medicamentos")
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
        fig_med.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_med, use_container_width=True)
        
        st.markdown("### Padrão de Uso Recente")
        uso_recente = df_depressao['Uso_Medicamento_Depressao_Ultimas_Semanas'].value_counts().reset_index()
        uso_recente['index'] = uso_recente['index'].map({
            1: 'Usa todos', 2: 'Usa alguns', 3: 'Não usa', 4: 'Não sabe'
        })
        
        fig_ur = px.bar(
            uso_recente,
            x='index',
            y='count',
            color='index',
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        st.plotly_chart(fig_ur, use_container_width=True)
    
    with col2:
        st.markdown("### Frequência de Visitas Médicas")
        visitas = df_depressao['Frequencia_Visita_Medico_Depressao'].value_counts().reset_index()
        visitas['index'] = visitas['index'].map({
            1: 'Regularmente', 2: 'Só quando precisa', 3: 'Nunca vai'
        })
        
        fig_vis = px.bar(
            visitas,
            x='index',
            y='count',
            color='index',
            title="Frequência de Visitas ao Médico"
        )
        st.plotly_chart(fig_vis, use_container_width=True)
        
        st.markdown("### Motivos para Não Visitar Regularmente")
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
            title="Principais Motivos para Não Visitar o Médico"
        )
        st.plotly_chart(fig_mot, use_container_width=True)

# Página: Teste Pessoal
elif pagina == "📝 Teste Pessoal":
    st.title("📝 Avaliação de Saúde Mental")
    st.markdown("""
    <div style="background-color:#e8f4fc;padding:15px;border-radius:10px;">
        Responda às perguntas abaixo para uma avaliação preliminar do seu estado emocional.
    </div>
    """, unsafe_allow_html=True)
    
    st.warning("""
    ⚠️ Este teste não substitui uma avaliação profissional. 
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
                          ["Nenhum dia", "Alguns dias", "Mais da metade", "Quase todos"], 
                          index=0)
            
            interesse = st.radio("Perdeu interesse pelas coisas?", 
                               ["Nenhum dia", "Alguns dias", "Mais da metade", "Quase todos"], 
                               index=0)
            
            alimentacao = st.radio("Teve mudanças no apetite?", 
                                 ["Nenhum dia", "Alguns dias", "Mais da metade", "Quase todos"], 
                                 index=0)
            
            cansaco = st.radio("Sentiu-se cansado sem energia?", 
                              ["Nenhum dia", "Alguns dias", "Mais da metade", "Quase todos"], 
                              index=0)
        
        with col2:
            concentracao = st.radio("Teve dificuldade de concentração?", 
                                  ["Nenhum dia", "Alguns dias", "Mais da metade", "Quase todos"], 
                                  index=0)
            
            deprimido = st.radio("Sentiu-se deprimido ou sem perspectiva?", 
                               ["Nenhum dia", "Alguns dias", "Mais da metade", "Quase todos"], 
                               index=0)
            
            fracasso = st.radio("Sentiu-se um fracasso?", 
                              ["Nenhum dia", "Alguns dias", "Mais da metade", "Quase todos"], 
                              index=0)
            
            suicidio = st.radio("Teve pensamentos sobre morte?", 
                              ["Nenhum dia", "Alguns dias", "Mais da metade", "Quase todos"], 
                              index=0)
        
        submitted = st.form_submit_button("Avaliar")
        
        if submitted:
            # Simulação de pontuação (em um caso real, usar o modelo)
            respostas = [sono, interesse, alimentacao, cansaco, concentracao, deprimido, fracasso, suicidio]
            pontos = sum([1 for r in respostas if r != "Nenhum dia"])
            
            if pontos >= 5:
                st.error("""
                🔴 Resultado: Indícios significativos de depressão
                
                Recomendamos que você procure ajuda profissional. Você não está sozinho(a) e a ajuda pode fazer diferença.
                """)
            elif pontos >= 2:
                st.warning("""
                🟡 Resultado: Alguns sintomas presentes
                
                Fique atento(a) aos seus sentimentos. Se os sintomas persistirem, considere conversar com um profissional.
                """)
            else:
                st.success("""
                🟢 Resultado: Poucos ou nenhum sintoma
                
                Continue cuidando da sua saúde mental. Praticar exercícios, manter rotinas saudáveis e conexões sociais são importantes.
                """)
            
            st.markdown("---")
            st.markdown("### 📞 Recursos de Apoio")
            st.markdown("""
            - **CVV (Centro de Valorização da Vida)**: 188 (ligação gratuita)
            - **CAPS (Centros de Atenção Psicossocial)**: Procure a unidade mais próxima
            - **SUS**: Agende uma consulta na UBS mais próxima
            """)

# Rodapé
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: gray; font-size: 0.9em;">
    Dados da Pesquisa Nacional de Saúde (PNS) 2019 - IBGE<br>
    Dashboard desenvolvido para análise de saúde mental
</div>
""", unsafe_allow_html=True)