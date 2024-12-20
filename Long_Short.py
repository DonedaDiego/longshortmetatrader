#principal
import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller, coint
from statsmodels.regression.linear_model import OLS
from statsmodels.regression.rolling import RollingOLS
import plotly.graph_objects as go
from datetime import datetime, timedelta
from st_aggrid import AgGrid, GridOptionsBuilder
import requests

BASE_URL = "https://api.cumecasadeanalises.com.br"

# Inicialização do MT5
def inicializar_mt5():
    try:
        resposta = requests.get(f'{BASE_URL}/initialize')
        return resposta.json().get('success', False)
    except:
        st.error("Não foi possível conectar com a API do MetaTrader5")
        return False

st.set_page_config(page_title="Long&Short - Por Cointegração", layout="wide",page_icon="📊")
#st.sidebar.image(r"C:\Users\usuario\Desktop\Vscode\longshortmetatrader\assets\Logo.png", width=100)

@st.cache_data(ttl=24*3600)
def obter_top_50_acoes_brasileiras():
    try:
        resposta = requests.get(f'{BASE_URL}/top50')
        if resposta.status_code == 200:
            return resposta.json()
        return []
    except:
        st.error("Não foi possível obter a lista de ações")
        return []

SETORES = {
   "ABEV3": "Bebidas",
   "ALOS3": "Exploração de Imóveis",
   "ALUP3": "Energia Elétrica",
   "ALUP4": "Energia Elétrica",
   "AURE3": "Energia Elétrica",
   "AZZA3": "Comércio Varejista", 
   "B3SA3": "Serviços Financeiros",
   "BBAS3": "Intermediários Financeiros",
   "BBDC3": "Intermediários Financeiros",
   "BBDC4": "Intermediários Financeiros",
   "BBSE3": "Previdência e Seguros",
   "BPAC11": "Intermediários Financeiros",
   "BPAN4": "Intermediários Financeiros",
   "BRAP3": "Mineração",
   "BRAP4": "Mineração",
   "BRAV3": "Petróleo, Gás e Biocombustíveis",
   "BRFS3": "Alimentos Processados",
   "CCRO3": "Transporte",
   "CMIG3": "Energia Elétrica",
   "CMIG4": "Energia Elétrica",
   "CMIN3": "Mineração",
   "CPFE3": "Energia Elétrica",
   "CPLE3": "Energia Elétrica",
   "CPLE5": "Energia Elétrica",
   "CPLE6": "Energia Elétrica",
   "CRFB3": "Comércio e Distribuição",
   "CSAN3": "Petróleo, Gás e Biocombustíveis",
   "CSMG3": "Água e Saneamento",
   "CSNA3": "Siderurgia e Metalurgia",
   "CXSE3": "Previdência e Seguros",
   "CYRE3": "Construção Civil",
   "ELET3": "Energia Elétrica",
   "ELET6": "Energia Elétrica",
   "EGIE3": "Energia Elétrica",
   "EMBR3": "Material de Transporte",
   "ENEV3": "Energia Elétrica",
   "ENGI11": "Energia Elétrica",
   "ENGI3": "Energia Elétrica",
   "ENGI4": "Energia Elétrica",
   "EQTL3": "Energia Elétrica",
   "GGBR3": "Siderurgia e Metalurgia",
   "GGBR4": "Siderurgia e Metalurgia",
   "GOAU4": "Siderurgia e Metalurgia",
   "HAPV3": "Serviços Médicos",
   "HYPE3": "Comércio e Distribuição",
   "ITSA3": "Holdings Diversificadas",
   "ITSA4": "Holdings Diversificadas",
   "ITUB3": "Intermediários Financeiros",
   "ITUB4": "Intermediários Financeiros",
   "JBSS3": "Alimentos Processados",
   "KLBN11": "Madeira e Papel",
   "KLBN3": "Madeira e Papel",
   "KLBN4": "Madeira e Papel",
   "LREN3": "Comércio Varejista",
   "MDIA3": "Alimentos Processados",
   "NEOE3": "Energia Elétrica",
   "NTCO3": "Produtos de Cuidado Pessoal",
   "PETR3": "Petróleo, Gás e Biocombustíveis",
   "PETR4": "Petróleo, Gás e Biocombustíveis",
   "PRIO3": "Petróleo, Gás e Biocombustíveis",
   "PSSA3": "Previdência e Seguros",
   "RAIL3": "Transporte",
   "RAIZ4": "Petróleo, Gás e Biocombustíveis",
   "RDOR3": "Serviços Médicos",
   "RENT3": "Diversos",
   "SANB11": "Intermediários Financeiros",
   "SANB4": "Intermediários Financeiros",
   "SBSP3": "Água e Saneamento",
   "SUZB3": "Madeira e Papel",
   "VBBR3": "Petróleo, Gás e Biocombustíveis",
   "VALE3": "Mineração",
   "VIVT3": "Telecomunicações",
   "WEGE3": "Máquinas e Equipamentos",
   "UGPA3": "Petróleo, Gás e Biocombustíveis"
}

@st.cache_data(ttl=24*3600)
def obter_dados(tickers, data_inicio, data_fim):
    try:
        dados_requisicao = {
            'tickers': tickers,
            'start_date': data_inicio.strftime('%Y-%m-%d'),
            'end_date': data_fim.strftime('%Y-%m-%d')
        }
        
        resposta = requests.post(
            f'{BASE_URL}/historical', 
            json=dados_requisicao
        )
        
        if resposta.status_code == 200:
            dados = resposta.json()
            return pd.read_json(dados['data'])
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Erro ao buscar dados da API: {str(e)}")
        return pd.DataFrame()
        
def calcular_meia_vida(spread):
    spread_lag = spread.shift(1)
    spread_diff = spread.diff()
    spread_lag = spread_lag.iloc[1:]
    spread_diff = spread_diff.iloc[1:]
    modelo = OLS(spread_diff, spread_lag).fit()
    if modelo.params[0] < 0:
        return -np.log(2) / modelo.params[0]
    return None

def teste_adf(serie_temporal):
    return adfuller(serie_temporal)[1]

def calcular_zscore(spread):
    return (spread - spread.mean()) / spread.std()

@st.cache_data(ttl=24*3600)
def analisar_pares(dados, max_meia_vida=30, min_meia_vida=1, max_pvalor_adf=0.05, min_correlacao=0.5, max_pvalor_coint=0.05):
    n = dados['Close'].shape[1]
    resultados = []
    total_pares = 0
    for i in range(n):
        for j in range(i+1, n):
            total_pares += 1
            acao1, acao2 = dados['Close'].columns[i], dados['Close'].columns[j]
            if len(dados['Close'][acao1].dropna()) > 1 and len(dados['Close'][acao2].dropna()) > 1:
                try:
                    _, pvalor, _ = coint(dados['Close'][acao1], dados['Close'][acao2])
                    if pvalor <= max_pvalor_coint:
                        modelo = OLS(dados['Close'][acao1], dados['Close'][acao2]).fit()
                        spread = dados.loc[:, ('Close', acao1)] - modelo.params[0] * dados.loc[:, ('Close', acao2)]
                        meia_vida = calcular_meia_vida(spread)
                        if meia_vida is not None:
                            pvalor_adf = teste_adf(spread)
                            correlacao = dados['Close'][acao1].corr(dados['Close'][acao2])
                            if (min_meia_vida <= meia_vida <= max_meia_vida and 
                                pvalor_adf <= max_pvalor_adf and 
                                correlacao >= min_correlacao):
                                resultados.append({
                                    'Ação1': acao1,
                                    'Ação2': acao2,
                                    'P-valor Cointegração': pvalor,
                                    'Meia-vida': meia_vida,
                                    'P-valor ADF': pvalor_adf,
                                    'Correlação': correlacao,
                                    'Beta': float(modelo.params[0])
                                })
                except:
                    continue
    return pd.DataFrame(resultados), total_pares

@st.cache_data(ttl=24*3600)
def filtrar_pares_por_zscore(dados, df_resultados, zscore_minimo=2):
    pares_validos = []
    for _, row in df_resultados.iterrows():
        acao1, acao2 = row['Ação1'], row['Ação2']
        
        # Calcular Beta Rotativo
        beta_rotativo = calcular_beta_rotativo(dados, acao1, acao2, 60)  
        if not beta_rotativo.empty:
            beta_atual = beta_rotativo.iloc[-1]
            beta_medio = beta_rotativo.mean()
            desvio_padrao = beta_rotativo.std()
            
            # Análise do status
            distancia_media = abs(beta_atual - beta_medio)
            if distancia_media <= desvio_padrao:
                status = 'Favorável'
            elif distancia_media <= 1.5 * desvio_padrao:
                status = 'Cautela'
            else:
                status = 'Não Recomendado'
        else:
            status = 'Indisponível'

                
        modelo = OLS(dados['Close'][acao1], dados['Close'][acao2]).fit()
        spread = dados['Close', acao1] - modelo.params[0] * dados['Close', acao2]
        zscore_atual = calcular_zscore(spread).iloc[-1]

        if abs(zscore_atual) >= zscore_minimo:
            direcao_acao1 = "↓ Venda" if zscore_atual > 0 else "↑ Compra"
            direcao_acao2 = "↑ Compra" if zscore_atual > 0 else "↓ Venda"
            pares_validos.append({
                'Ação1': acao1,
                'Ação2': acao2,
                'P-valor Cointegração': row['P-valor Cointegração'],
                'Meia-vida': row['Meia-vida'],
                'P-valor ADF': row['P-valor ADF'],
                'Correlação': row['Correlação'],
                'Beta': row['Beta'],  
                'Status': status,
                'Z-score Atual': zscore_atual,
                'Direção Ação1': direcao_acao1,
                'Direção Ação2': direcao_acao2
            })
    return pd.DataFrame(pares_validos)

def analisar_cointegration_stability(dados, acao1, acao2, periodos):
    resultados = []
    periodos = [60, 90, 120, 140, 180, 200, 240]  # Períodos mais granulares
    
    for periodo in periodos:
        dados_recentes = dados.tail(periodo)
        _, pvalor, _ = coint(dados_recentes['Close'][acao1], dados_recentes['Close'][acao2])
        spread = dados_recentes['Close'][acao1] - dados_recentes['Close'][acao2]
        meia_vida = calcular_meia_vida(spread) if len(spread.dropna()) > 1 else None
        modelo = OLS(dados_recentes['Close'][acao1], dados_recentes['Close'][acao2]).fit()
        
        resultados.append({
            'Período': periodo,
            'P-valor Coint.': round(pvalor, 4),
            'Meia-vida': round(meia_vida, 2) if meia_vida else None,
            'R²': round(modelo.rsquared, 4),
            'Status': 'Cointegrado' if pvalor < 0.05 else 'Não Cointegrado'
        })

    df = pd.DataFrame(resultados)
    
  
    return df

@st.cache_data(ttl=24*3600)
def calcular_atr(dados, acao, periodo=14):
    high = dados['High'][acao]
    low = dados['Low'][acao]
    close = dados['Close'][acao].shift(1)
    tr = pd.DataFrame({
        'hl': high - low,
        'hc': abs(high - close),
        'lc': abs(low - close)
    }).max(axis=1)
    return tr.rolling(window=periodo).mean()

def calcular_valor_stop_atr(dados, acao1, acao2, qtd1, qtd2, direcao_acao1, direcao_acao2, multiplicador_atr=2):
    atr1 = calcular_atr(dados, acao1)
    atr2 = calcular_atr(dados, acao2)
    ultimo_preco1 = dados['Close'][acao1].iloc[-1]
    ultimo_preco2 = dados['Close'][acao2].iloc[-1]

    if direcao_acao1 == "Compra":
        stop_acao1 = ultimo_preco1 - (multiplicador_atr * atr1.iloc[-1])
        stop_acao2 = ultimo_preco2 + (multiplicador_atr * atr2.iloc[-1])
    else:
        stop_acao1 = ultimo_preco1 + (multiplicador_atr * atr1.iloc[-1])
        stop_acao2 = ultimo_preco2 - (multiplicador_atr * atr2.iloc[-1])
    return abs(qtd1 * (ultimo_preco1 - stop_acao1) + qtd2 * (stop_acao2 - ultimo_preco2))

def calcular_valor_gain_atr(dados, acao1, acao2, qtd1, qtd2, direcao_acao1, direcao_acao2, multiplicador_atr=3):
    atr1 = calcular_atr(dados, acao1)
    atr2 = calcular_atr(dados, acao2)
    ultimo_preco1 = dados['Close'][acao1].iloc[-1]
    ultimo_preco2 = dados['Close'][acao2].iloc[-1]

    if direcao_acao1 == "Compra":
        gain_acao1 = ultimo_preco1 + (multiplicador_atr * atr1.iloc[-1])
        gain_acao2 = ultimo_preco2 - (multiplicador_atr * atr2.iloc[-1])
    else:
        gain_acao1 = ultimo_preco1 - (multiplicador_atr * atr1.iloc[-1])
        gain_acao2 = ultimo_preco2 + (multiplicador_atr * atr2.iloc[-1])
    return abs(qtd1 * (gain_acao1 - ultimo_preco1) + qtd2 * (ultimo_preco2 - gain_acao2))

def plotar_zscore(dados, acao1, acao2, zscore_entrada=2, zscore_saida=0, zscore_stop_loss=3):
    modelo = OLS(dados['Close'][acao1], dados['Close'][acao2]).fit()
    spread = dados['Close'][acao1] - modelo.params[0] * dados['Close'][acao2]
    zscore = calcular_zscore(spread)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=zscore.index, y=zscore, mode='lines', name='Z-score'))

    fig.add_hline(y=0, line_dash="dash", line_color="red", name="Média")
    fig.add_hline(y=zscore_entrada, line_dash="dash", line_color="green", name="Entrada Superior")
    fig.add_hline(y=-zscore_entrada, line_dash="dash", line_color="green", name="Entrada Inferior")
    fig.add_hline(y=zscore_saida, line_dash="dash", line_color="yellow", name="Take-Profit Superior")
    fig.add_hline(y=-zscore_saida, line_dash="dash", line_color="yellow", name="Take-Profit Inferior")
    fig.add_hline(y=zscore_stop_loss, line_dash="dash", line_color="red", name="Stop-Loss Superior")
    fig.add_hline(y=-zscore_stop_loss, line_dash="dash", line_color="red", name="Stop-Loss Inferior")

    fig.update_layout(
        title=f'Z-score para {acao1} vs {acao2}',
        xaxis_title='Data',
        yaxis_title='Z-score',
        showlegend=True
    )
    return fig

def plotar_spread(dados, acao1, acao2):
    spread = dados['Close'][acao1] / dados['Close'][acao2]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dados.index, y=spread, mode='lines', name='Spread'))

    fig.update_layout(
        title=f'Spread entre {acao1} e {acao2}',
        xaxis_title='Data',
        yaxis_title='Spread (Preço Relativo)',
        showlegend=True,
        height=400
    )
    return fig

@st.cache_data(ttl=3600)
def calcular_beta_rotativo(dados, acao1, acao2, janela):
    log_returns1 = np.log(dados['Close'][acao1] / dados['Close'][acao1].shift(1))
    log_returns2 = np.log(dados['Close'][acao2] / dados['Close'][acao2].shift(1))

    if len(log_returns1.dropna()) < janela or len(log_returns2.dropna()) < janela:
        return pd.Series()

    model = RollingOLS(log_returns1, log_returns2, window=janela)
    rolling_res = model.fit()

    return pd.Series(rolling_res.params.iloc[:, 0], index=dados.index[janela:])

def analisar_beta_rotativo(beta_rotativo, beta_atual, beta_medio, desvio_padrao):

    analise = {
        'status': 'Normal',
        'nivel_risco': 'Baixo',
        'alertas': [],
        'sugestoes': [],
        'interpretacao': '',
        'explicacao_status': ''
    }
    
    # Verifica a distância da média
    distancia_media = abs(beta_atual - beta_medio)
    
    # Análise da situação atual
    if distancia_media <= desvio_padrao:
        analise['status'] = 'Favorável'
        analise['nivel_risco'] = 'Baixo'
        analise['explicacao_status'] = """
        ✅ Momento Ideal:
        - Beta próximo da média histórica
        - Relação entre os ativos está estável
        - Boa oportunidade para operar
        """
        analise['sugestoes'].extend([
            "Tamanho normal de posição",
            "Stops podem ser mais relaxados",
            "Monitoramento normal"
        ])
        
    elif distancia_media <= 1.5 * desvio_padrao:
        analise['status'] = 'Cautela'
        analise['nivel_risco'] = 'Médio'
        analise['explicacao_status'] = """
        ⚠️ Momento de Atenção:
        - Beta se afastando da média
        - Relação ainda estável, mas requer atenção
        - Operar com mais cuidado
        """
        analise['sugestoes'].extend([
            "Reduzir tamanho da posição",
            "Stops mais próximos",
            "Monitoramento mais frequente"
        ])
        
    else:
        analise['status'] = 'Não Recomendado'
        analise['nivel_risco'] = 'Alto'
        analise['explicacao_status'] = """
        🚫 Momento Desfavorável:
        - Beta muito distante da média
        - Possível mudança na relação dos ativos
        - Risco de desintegração do par
        """
        analise['sugestoes'].extend([
            "Evitar novas entradas",
            "Se já estiver posicionado, considerar sair",
            "Aguardar estabilização do beta"
        ])

    # Adiciona interpretação da sensibilidade
    if beta_atual > beta_medio:
        analise['interpretacao'] = f"🔍 Os ativos estão {((beta_atual/beta_medio - 1) * 100):.1f}% mais correlacionados que o normal"
    else:
        analise['interpretacao'] = f"🔍 Os ativos estão {((1 - beta_atual/beta_medio) * 100):.1f}% menos correlacionados que o normal"
        
    return analise

def exibir_analise_beta(dados, acao1, acao2, janela_beta):
    """
    Exibe a análise do beta de forma didática
    """
    fig_beta, media_beta, desvio_padrao_beta, beta_rotativo = plotar_beta_rotativo(dados, acao1, acao2, janela_beta)
    
    if beta_rotativo is not None and not beta_rotativo.empty:
        beta_atual = beta_rotativo.iloc[-1]
        analise = analisar_beta_rotativo(beta_rotativo, beta_atual, media_beta, desvio_padrao_beta)
        
        st.write("## 📊 Análise do Beta Rotation")
        
        # Métricas principais com explicações
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Status da Operação", analise['status'])
        with col2:
            st.metric("Nível de Risco", analise['nivel_risco'])
        with col3:
            st.metric("Beta Atual", f"{beta_atual:.4f}")
        
        # Explicação do status atual
        st.markdown(analise['explicacao_status'])
            
        # Interpretação da correlação
        st.info(analise['interpretacao'])
        
        # Sugestões práticas
        if analise['sugestoes']:
            st.success("**💡 Sugestões Práticas:**\n" + "\n".join(f"• {sugestao}" for sugestao in analise['sugestoes']))
            
        # Explicação didática do beta
        st.markdown("""
        ### 📖 Entenda o Beta:
        - **Próximo da média (linha vermelha)** = Relação normal entre os ativos
        - **Dentro das bandas (linhas verdes)** = Comportamento previsível
        - **Movimento suave** = Maior estabilidade na relação
        - **Movimento brusco** = Possível mudança na relação dos ativos
        """)

def plotar_beta_rotativo(dados, acao1, acao2, janela):
    beta_rotativo = calcular_beta_rotativo(dados, acao1, acao2, janela)
    if beta_rotativo.empty:
        return None, None, None, None

    media_beta = beta_rotativo.mean()
    desvio_padrao_beta = beta_rotativo.std()
    limite_superior = media_beta + 2 * desvio_padrao_beta
    limite_inferior = media_beta - 2 * desvio_padrao_beta

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=beta_rotativo.index, y=beta_rotativo, mode='lines', name='Beta Rotation'))
    fig.add_trace(go.Scatter(x=beta_rotativo.index, y=[media_beta]*len(beta_rotativo), 
                             mode='lines', name='Média', line=dict(color='red', dash='dash')))
    fig.add_trace(go.Scatter(x=beta_rotativo.index, y=[limite_superior]*len(beta_rotativo), 
                             mode='lines', name='+2 Desvios Padrão', line=dict(color='green', dash='dot')))
    fig.add_trace(go.Scatter(x=beta_rotativo.index, y=[limite_inferior]*len(beta_rotativo), 
                             mode='lines', name='-2 Desvios Padrão', line=dict(color='green', dash='dot')))

    fig.update_layout(
        title=f'Beta Rotation entre {acao1} e {acao2} (Janela: {janela} dias)',
        xaxis_title='Data',
        yaxis_title='Beta',
        showlegend=True
    )

    return fig, media_beta, desvio_padrao_beta, beta_rotativo

def ajustar_quantidade(qtd):
    if qtd >= 100:
        resto = qtd % 100
        if resto >= 51:
            return round((qtd + (100 - resto)) / 100) * 100
        return round((qtd - resto) / 100) * 100
    return round(qtd)

def main():
    st.title("Long&Short - Learn Ai & Machine Learning Geminii")
    st.sidebar.markdown("---")
    st.sidebar.markdown("[www.geminii.com.br](https://www.geminii.com.br)")

    investimento = st.sidebar.number_input("Valor a ser investido (R$)", min_value=1000.0, value=10000.0, step=1000.0)

    # Aqui substituímos o número de dias por um selectbox com valores pré-definidos
    dias = st.sidebar.selectbox("Número de dias para análise", [60, 90, 120, 140, 180, 200, 240], index=6)

    janela_beta = st.sidebar.number_input("Janela para cálculo do Beta Rotation (dias)", min_value=30, value=min(60, dias-1), max_value=dias-1)

    zscore_minimo = st.sidebar.number_input("Z-Score Mínimo para Filtro", min_value=1.0, max_value=3.0, value=2.0, step=0.1)
    max_meia_vida = 50
    #max_pvalor_adf = st.sidebar.number_input("P-valor ADF Máximo", min_value=0.01, max_value=0.1, value=0.05, format="%.2f")
    min_correlacao = 0.05
    max_pvalor_adf = 0.05
    max_pvalor_coint = 0.05
    #max_pvalor_coint = st.sidebar.number_input("P-valor de Cointegração Máximo", min_value=0.01, max_value=0.1, value=0.05, format="%.2f")

    multiplicador_atr_stop = 2
    multiplicador_atr_gain = 3
    
    # Inicialização das variáveis de sessão
    if 'dados' not in st.session_state:
        st.session_state.dados = None
    if 'df_resultados_filtrados' not in st.session_state:
        st.session_state.df_resultados_filtrados = None
    if 'total_pares' not in st.session_state:
        st.session_state.total_pares = 0
    
    # Botão de atualização
    st.sidebar.markdown("---")
    atualizar = st.sidebar.button("🔄 Analisar Pares", use_container_width=True)

    if atualizar:
        tickers = obter_top_50_acoes_brasileiras()
        data_fim = datetime.now()
        data_inicio = data_fim - timedelta(days=dias)

        with st.spinner("Obtendo dados..."):
            st.session_state.dados = obter_dados(tickers, data_inicio, data_fim)

        with st.spinner("Analisando pares..."):
            df_resultados, st.session_state.total_pares = analisar_pares(st.session_state.dados, max_meia_vida, 1, 
                                                                        max_pvalor_adf, min_correlacao, max_pvalor_coint)

        with st.spinner("Filtrando pares com base no Z-score mínimo..."):
            st.session_state.df_resultados_filtrados = filtrar_pares_por_zscore(st.session_state.dados, 
                                                                              df_resultados, zscore_minimo)

    # Exibição dos resultados
    if st.session_state.df_resultados_filtrados is not None:
        st.write("### 🔍 Resultado da Análise de Pares")
        st.write(f"**Volume Total Processado:** {st.session_state.total_pares} pares analisados")

        # Exibição de métricas resumidas
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            <div style='background-color:#1F3D33; padding:10px; border-radius:5px;'>
            ✅ {len(st.session_state.df_resultados_filtrados)} pares identificados com oportunidade de trading (Z-score > {zscore_minimo})
            </div>
            """, unsafe_allow_html=True)
        with col2:
            setores_unicos = set()
            for _, row in st.session_state.df_resultados_filtrados.iterrows():
                setores_unicos.add(SETORES[row['Ação1']])
                setores_unicos.add(SETORES[row['Ação2']])
            st.markdown(f"""
            <div style='background-color:#1F2D3D; padding:10px; border-radius:5px;'>
            📊 {len(setores_unicos)} setores diferentes identificados nas oportunidades
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

        # Exibição da tabela interativa
        if not st.session_state.df_resultados_filtrados.empty:
            df_display = st.session_state.df_resultados_filtrados.copy()
            df_display = df_display[['Ação1', 'Ação2', 'Meia-vida', 'Status', 'Z-score Atual', 'Direção Ação1', 'Direção Ação2']]
            df_display = df_display.rename(columns={'Status': 'Beta Rotation'})

            # Configuração do Grid
            gb = GridOptionsBuilder.from_dataframe(df_display)
            gb.configure_column('Meia-vida', type=["numericColumn"], precision=2)
            gb.configure_column('Z-score Atual', type=["numericColumn"], precision=2)
            
            gb.configure_column("Direção Ação1", width=100)
            gb.configure_column("Direção Ação2", width=100)
            grid_options = gb.build()

            # Renderização do AgGrid
            response = AgGrid(
                df_display,
                gridOptions=grid_options,
                theme="alpine",
                update_mode="MODEL_CHANGED",
                allow_unsafe_jscode=True,
                height=400
            )

            st.subheader("Análise Detalhada de Par")
            opcoes_pares = [f"{row['Ação1']} - {row['Ação2']}" for _, row in df_display.iterrows()]
            par_selecionado = st.selectbox("Selecione um par para análise detalhada:", opcoes_pares)
            if par_selecionado:
                acao1, acao2 = par_selecionado.split(' - ')
                col1, col2 = st.columns(2)
                with col1:
                    st.info(f"**{acao1}**: {SETORES[acao1]}")
                with col2:
                    st.info(f"**{acao2}**: {SETORES[acao2]}")    
            if par_selecionado:
                acao1, acao2 = par_selecionado.split(' - ')
                acao1_calc = acao1
                acao2_calc = acao2

                linha_selecionada = st.session_state.df_resultados_filtrados[
                    (st.session_state.df_resultados_filtrados['Ação1'] == acao1_calc) & 
                    (st.session_state.df_resultados_filtrados['Ação2'] == acao2_calc)
                ].iloc[0]

                direcao_acao1 = linha_selecionada['Direção Ação1']
                direcao_acao2 = linha_selecionada['Direção Ação2']

                ultimo_preco1 = st.session_state.dados['Close'][acao1_calc].iloc[-1]
                ultimo_preco2 = st.session_state.dados['Close'][acao2_calc].iloc[-1]

                # Cálculo das quantidades e valores
                if linha_selecionada['Beta'] > 0:
                    valor_por_lado = investimento / 2
                    qtd1 = ajustar_quantidade(round(valor_por_lado / ultimo_preco1))
                    qtd2 = ajustar_quantidade(round(valor_por_lado / ultimo_preco2))
                    
                    valor_acao1 = qtd1 * ultimo_preco1
                    valor_acao2 = qtd2 * ultimo_preco2
                    
                    valor_total = abs(qtd1 * ultimo_preco1) + abs(qtd2 * ultimo_preco2)
                    if valor_total > investimento:
                        fator_ajuste = investimento / valor_total
                        qtd1 = round(qtd1 * fator_ajuste)
                        qtd2 = round(qtd2 * fator_ajuste)
                        qtd1 = ajustar_quantidade(qtd1)
                        qtd2 = ajustar_quantidade(qtd2)
                            
                    valor_acao1 = qtd1 * ultimo_preco1
                    valor_acao2 = qtd2 * ultimo_preco2

                    if abs(valor_acao1 - valor_acao2) > min(ultimo_preco1, ultimo_preco2):
                        if valor_acao1 > valor_acao2:
                            qtd1 = ajustar_quantidade(round(qtd1 * (valor_acao2 / valor_acao1)))
                        else:
                            qtd2 = ajustar_quantidade(round(qtd2 * (valor_acao1 / valor_acao2)))
                else:
                    valor_por_lado = investimento / 2
                    qtd1 = ajustar_quantidade(round(valor_por_lado / ultimo_preco1))
                    qtd2 = ajustar_quantidade(round(valor_por_lado / ultimo_preco2))
                                
                    valor_total = abs(qtd1 * ultimo_preco1) + abs(qtd2 * ultimo_preco2)
                    if valor_total > investimento:
                        fator_ajuste = investimento / valor_total
                        qtd1 = round(qtd1 * fator_ajuste)
                        qtd2 = round(qtd2 * fator_ajuste)
                        qtd1 = ajustar_quantidade(qtd1)
                        qtd2 = ajustar_quantidade(qtd2)

                valor_total_acao1 = (qtd1 * ultimo_preco1) * (1 if direcao_acao1 == "Venda" else -1)
                valor_total_acao2 = (qtd2 * ultimo_preco2) * (-1 if direcao_acao2 == "Venda" else 1)

                impacto_liquido = valor_total_acao1 + valor_total_acao2

                valor_stop = calcular_valor_stop_atr(st.session_state.dados, acao1_calc, acao2_calc, qtd1, qtd2, 
                                                   direcao_acao1, direcao_acao2, multiplicador_atr=multiplicador_atr_stop)
                valor_gain = calcular_valor_gain_atr(st.session_state.dados, acao1_calc, acao2_calc, qtd1, qtd2, 
                                                   direcao_acao1, direcao_acao2, multiplicador_atr=multiplicador_atr_gain)

                valor_total_operacao_abs = (abs(qtd1) * ultimo_preco1 + abs(qtd2) * ultimo_preco2)
                percentual_stop = (valor_stop / valor_total_operacao_abs) * 100 if valor_total_operacao_abs != 0 else 0
                percentual_gain = (valor_gain / valor_total_operacao_abs) * 100 if valor_total_operacao_abs != 0 else 0

                # Gráficos e análises
                fig_zscore = plotar_zscore(st.session_state.dados, acao1_calc, acao2_calc)
                fig_spread = plotar_spread(st.session_state.dados, acao1_calc, acao2_calc)

                spread = st.session_state.dados['Close'][acao1_calc] / st.session_state.dados['Close'][acao2_calc]
                df_spread_tab = pd.DataFrame({'Data': spread.index, 'Spread': spread.values})
                df_spread_tab['Data'] = df_spread_tab['Data'].dt.strftime('%Y-%m-%d')
                df_spread_tab = df_spread_tab.sort_values('Data', ascending=False).head(3)

                fig_beta_rotativo, media_beta, desvio_padrao_beta, beta_rotativo = plotar_beta_rotativo(
                    st.session_state.dados, acao1_calc, acao2_calc, janela_beta)

                # Detalhes do par
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Detalhes do Par:**")
                    st.write(f"Correlação: {linha_selecionada['Correlação']:.4f}")
                    st.write(f"Beta: {linha_selecionada['Beta']:.4f}")
                    st.write(f"Meia-vida: {linha_selecionada['Meia-vida']:.2f} dias")
                with col2:
                    st.write("**Testes Estatísticos:**")
                    st.write(f"P-valor de Cointegração: {linha_selecionada['P-valor Cointegração']:.4f}")
                    st.write(f"P-valor ADF: {linha_selecionada['P-valor ADF']:.4f}")
                    st.write(f"Z-score atual: {linha_selecionada['Z-score Atual']:.4f}")

                st.subheader(f"Financeiro a ser Investido R$ {investimento:_.2f}".replace(".", ",").replace("_", "."))

                df_qtd = pd.DataFrame({
                    "Ação": [acao1, acao2],
                    "Direção": [direcao_acao1, direcao_acao2],
                    "Quantidade": [qtd1, qtd2],
                    "Último Preço do ativo": [f"R$ {ultimo_preco1:.2f}".replace(".", ",").replace("_", "."), f"R$ {ultimo_preco2:.2f}".replace(".", ",").replace("_", ".")],
                    "Valor Total": [
                        f"R$ {valor_total_acao1:_.2f}".replace(".", ",").replace("_", "."),
                        f"R$ {valor_total_acao2:_.2f}".replace(".", ",").replace("_", ".")
                    ]
                })

                
                tabs = st.tabs(["Z-score", "Resumo Financeiro", "Stop/Gain ATR", "Beta Rotation", "Spread", "Margem e Custos", "Análise Períodos"])

                with tabs[0]:
                    st.subheader(f"Z-score para {acao1} vs {acao2}")
                    st.plotly_chart(fig_zscore)

                with tabs[1]:
                    st.subheader("Resumo Financeiro")
                    st.table(df_qtd)
                    st.write(f"**Impacto financeiro líquido na conta:** R$ {impacto_liquido:,.2f}")
                    st.write("* As quantidades podem não coincidir exatamente com os valores financeiros, pois são arredondadas para manter a neutralidade da operação.")

                with tabs[2]:
                    st.subheader("Stop/Gain Sugestão")
                    st.write(f"**Valor estimado de Stop-Loss:** R$ {valor_stop:_.2f} ({percentual_stop:.2f}%)")
                    st.write(f"**Valor estimado de Gain:** R$ {valor_gain:.2f} ({percentual_gain:.2f}%)")
                    st.write("A análise sugere gain e stop considerando a volatilidade dos ativos, além de avaliar o retorno à média.")    

                with tabs[3]:
                    st.subheader(f"Beta Rotation entre {acao1} e {acao2}")
                    if fig_beta_rotativo is not None and beta_rotativo is not None and not beta_rotativo.empty:
                        st.plotly_chart(fig_beta_rotativo)
                        exibir_analise_beta(st.session_state.dados, acao1_calc, acao2_calc, janela_beta)
                    else:
                        st.warning("Não foi possível calcular o Beta Rotation com os dados disponíveis.")

                with tabs[4]:
                    st.subheader(f"Spread entre {acao1} e {acao2}")
                    st.plotly_chart(fig_spread)
                    st.subheader(f"Últimos Valores do Spread ({acao1}/{acao2})")
                    st.table(df_spread_tab.set_index('Data'))

                with tabs[5]:
                    st.subheader("Análise de Margem e Custos Operacionais")
                    c1, c2 = st.columns(2)
                    with c1:
                        percentual_garantia = st.number_input("Percentual de Garantia (%)", value=25.0, min_value=0.0, max_value=100.0)
                        taxa_btc_anual = st.number_input("Taxa BTC Anual (%)", value=2.0, min_value=0.0, max_value=50.0)
                    with c2:
                        dias_operacao = st.number_input("Dias da Operação", value=10, min_value=1)
                        taxa_corretora_btc = st.number_input("Taxa Corretora sobre BTC (%)", value=35.0, min_value=0.0, max_value=100.0)

                    valor_vendido = 0.0
                    valor_comprado = 0.0
                    for idx, row in df_qtd.iterrows():
                        val = float(row['Valor Total'].replace('R$ ', '').replace(',', ''))
                        if row['Direção'] == "Venda":
                            valor_vendido += val
                        else:
                            valor_comprado += val
                
                    garantia_venda = valor_vendido * (1 + percentual_garantia/100)
                    garantia_compra = abs(valor_comprado) * (1 + percentual_garantia/100)
                    margem_necessaria = garantia_venda - garantia_compra

                    volume_total = abs(valor_vendido) + abs(valor_comprado)
                    emolumentos = (volume_total / 10000) * 3.25
                    taxa_btc_diaria = (1 + taxa_btc_anual/100) ** (1/252) - 1
                    custo_btc_periodo = valor_vendido * taxa_btc_diaria * dias_operacao
                    taxa_corretora = custo_btc_periodo * (taxa_corretora_btc/100)
                    custo_total = emolumentos + custo_btc_periodo + taxa_corretora

                    c3, c4 = st.columns(2)
                    with c3:
                        st.write("Valores da Operação:")
                        st.write(f"Total Vendido: R$ {valor_vendido:,.2f}")
                        st.write(f"Total Comprado: R$ {valor_comprado:,.2f}")
                        st.write(f"Garantia sobre Vendas: R$ {garantia_venda:,.2f}")
                        st.write(f"Contragarantia sobre Compras: R$ {garantia_compra:,.2f}")

                    with c4:
                        st.write("Margem e Custos:")
                        st.write(f"Margem Necessária: R$ {margem_necessaria:,.2f}")
                        st.write(f"Emolumentos B3: R$ {emolumentos:,.2f}")
                        st.write(f"Custo BTC: R$ {custo_btc_periodo:,.2f}")
                        st.write(f"Taxa Corretora BTC: R$ {taxa_corretora:,.2f}")

                    st.write(f"**Custo Total Estimado: R$ {custo_total:,.2f}**")

                    st.markdown("---")
                    st.info("""
                    **Notas importantes:**
                    - Os valores são parte de uma média bem próxima a realidade mas cada corretora tem a sua. (Consulte) 
                    - Emolumentos B3: 3,25 a cada 10.000 negociados aproximadamente
                    - BTC (aluguel) é calculado sobre o valor vendido 
                    - A margem pode variar conforme a volatilidade dos ativos
                    - Recomenda-se manter folga na margem para ajustes
                    """)
                with tabs[6]:
                    st.subheader("Análise de Estabilidade por Períodos")
                    
                    periodos = [60, 90, 120, 140, 180, 200, 240]
                    df_estabilidade = analisar_cointegration_stability(st.session_state.dados, acao1_calc, acao2_calc, periodos)
                    df_display = df_estabilidade[['Período', 'P-valor Coint.', 'R²', 'Status']]
                    col1, col2 = st.columns(2)
                    with col1:
                        st.dataframe(df_display)
                        
                    with col2:
                        # Análise da estabilidade
                        periodos_cointegrados = df_estabilidade['Status'].eq('Cointegrado').sum()
                        total_periodos = len(df_estabilidade)
                        meia_vida_validos = df_estabilidade['Meia-vida'].dropna()
                        var_meia_vida = meia_vida_validos.std() / meia_vida_validos.mean() if not meia_vida_validos.empty else None
                        
                        if periodos_cointegrados == total_periodos:
                            status = "✅ Cointegração estável em todos os períodos"
                            nivel_risco = "Baixo"
                        elif periodos_cointegrados >= total_periodos * 0.7:
                            status = "⚠️ Cointegração presente na maioria dos períodos"
                            nivel_risco = "Médio"
                        else:
                            status = "❌ Períodos sem cointegração"
                            nivel_risco = "Alto"

                        st.markdown("### Diagnóstico")
                        st.markdown(f"**{status}**")
                        st.markdown(f"""
                        • {periodos_cointegrados} de {total_periodos} períodos cointegrados
                        • Variação Meia-vida: {var_meia_vida*100:.2f}% {"(Estável)" if var_meia_vida < 0.3 else "(Instável)"}
                        • R² médio: {df_estabilidade['R²'].mean():.4f} (>{0.95:.2f} indica boa aderência)
                        """)   
                   
                                      
        else:
            st.warning("Nenhum par encontrado que atenda aos critérios. Tente ajustar os parâmetros.")

        st.markdown("---")
        st.subheader('Dinâmica de Mercado:')
        st.markdown("""
        É importante lembrar que a eficácia dessa estratégia depende da **dinâmica do mercado**. Nem sempre os pares se comportam como esperado, 
        pois eventos inesperados, mudanças no cenário econômico, notícias ou fundamentos das empresas podem afetar a relação entre os preços.

        Portanto, o sucesso da estratégia depende de **monitorar constantemente os trades** e adaptar os parâmetros conforme as condições do mercado mudam.
        """)

    else:
        st.info("👆 Configure os parâmetros no menu lateral e clique em 'Analisar Pares' para iniciar a análise.")

    st.markdown("---")
    st.subheader('Disclaimer:')
    st.markdown("""
    O conteúdo deste material é destinado exclusivamente a fins informativos e educacionais. 
    As análises, opiniões e informações apresentadas não constituem recomendação de investimento. 
    Cada investidor deve fazer sua própria análise e tomar suas decisões de forma independente, considerando seu perfil, objetivos e tolerância a risco.

    Diego Doneda, Analista CNPI - 9668
    Geminii Learn Ai & Machine Learning
    """)

if __name__ == "__main__":
    main()