import streamlit as st
import pandas as pd
import numpy as np

from statsmodels.tsa.stattools import adfuller, coint
from statsmodels.regression.linear_model import OLS
from statsmodels.regression.rolling import RollingOLS
import plotly.graph_objects as go
from datetime import datetime, timedelta
import requests

API_URL = "https://b6a7-2001-1284-f514-4e92-dda9-fc26-6bc2-a8dd.ngrok-free.app"


st.set_page_config(page_title="Long&Short - Learn Ai & Machine Learning Geminii", layout="wide")
st.sidebar.image(r"C:\Users\usuario\Desktop\Vscode\longshortmetatrader\assets\Logo.png", width=100)

@st.cache_data(ttl=24*3600)
def obter_top_50_acoes_brasileiras():
    return [acao for acao in [
        "ABEV3", "ALOS3","AMBP3", "AZUL4", "B3SA3", "BBAS3", "BBDC3", "BBDC4", "BHIA3", "BRAP4", "BRFS3", "BRKM5", 
        "CCRO3", "CMIG4", "CPLE6", "CRFB3", "CSAN3", "CSNA3", "EGIE3", "ELET3", "ELET6", "EQTL3", "GGBR4", 
        "GOAU4", "IRBR3", "ITSA4", "JBSS3", "KLBN11", "LREN3", "MGLU3", "MRFG3", "MULT3", "NTCO3", "PCAR3", 
        "PETR3", "PETR4", "PETZ3", "PRIO3", "RADL3", "RAIL3", "RAIZ4", "RENT3", "SANB11", "SAPR4", "SUZB3",
        "TAEE11", "TIMS3", "TRAD3", "UGPA3", "USIM5", "VALE3", "VIVT3", "VBBR3","WEGE3", "YDUQ3",
    ]]


@st.cache_data(ttl=24*3600)
def obter_dados(tickers, data_inicio, data_fim):
    try:
        # Converte as datas para strings
        start_date = data_inicio.strftime("%Y-%m-%d")
        end_date = data_fim.strftime("%Y-%m-%d")
        
        # Faz a requisição ao backend para obter os dados históricos
        response = requests.get(
            f"{API_URL}/historical",
            params={"tickers": ",".join(tickers), "start_date": start_date, "end_date": end_date}
        )
        
        # Verifica se a resposta foi bem-sucedida
        if response.status_code == 200:
            data = response.json()
            # Converte o JSON retornado pela API em um DataFrame
            dados = pd.DataFrame(data)
            dados['time'] = pd.to_datetime(dados['time'])
            return dados.set_index('time')
        else:
            st.error(f"Erro ao obter dados: {response.text}")
            return pd.DataFrame()
    except Exception as e:
        st.error(f"Erro ao conectar à API: {str(e)}")
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
                        spread = dados['Close'][acao1] - modelo.params[0] * dados['Close'][acao2]
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
        modelo = OLS(dados['Close'][acao1], dados['Close'][acao2]).fit()
        spread = dados['Close'][acao1] - modelo.params[0] * dados['Close'][acao2]
        zscore_atual = calcular_zscore(spread).iloc[-1]

        if abs(zscore_atual) >= zscore_minimo:
            direcao_acao1 = "Venda" if zscore_atual > 0 else "Compra"
            direcao_acao2 = "Compra" if zscore_atual > 0 else "Venda"
            pares_validos.append({
                'Ação1': acao1,
                'Ação2': acao2,
                'P-valor Cointegração': row['P-valor Cointegração'],
                'Meia-vida': row['Meia-vida'],
                'P-valor ADF': row['P-valor ADF'],
                'Correlação': row['Correlação'],
                'Beta': row['Beta'],
                'Z-score Atual': zscore_atual,
                'Direção Ação1': direcao_acao1,
                'Direção Ação2': direcao_acao2
            })
    return pd.DataFrame(pares_validos)

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
    """
    Análise didática do beta rotativo para decisões de trading
    """
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
        
        st.write("## 📊 Análise do Beta Rotativo")
        
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
    fig.add_trace(go.Scatter(x=beta_rotativo.index, y=beta_rotativo, mode='lines', name='Beta Rotativo'))
    fig.add_trace(go.Scatter(x=beta_rotativo.index, y=[media_beta]*len(beta_rotativo), 
                             mode='lines', name='Média', line=dict(color='red', dash='dash')))
    fig.add_trace(go.Scatter(x=beta_rotativo.index, y=[limite_superior]*len(beta_rotativo), 
                             mode='lines', name='+2 Desvios Padrão', line=dict(color='green', dash='dot')))
    fig.add_trace(go.Scatter(x=beta_rotativo.index, y=[limite_inferior]*len(beta_rotativo), 
                             mode='lines', name='-2 Desvios Padrão', line=dict(color='green', dash='dot')))

    fig.update_layout(
        title=f'Beta Rotativo entre {acao1} e {acao2} (Janela: {janela} dias)',
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

    janela_beta = st.sidebar.number_input("Janela para cálculo do Beta Rotativo (dias)", min_value=30, value=min(60, dias-1), max_value=dias-1)

    zscore_minimo = st.sidebar.number_input("Z-Score Mínimo para Filtro", min_value=1.0, max_value=3.0, value=2.0, step=0.1)
    max_meia_vida = 50
    max_pvalor_adf = st.sidebar.number_input("P-valor ADF Máximo", min_value=0.01, max_value=0.1, value=0.05, format="%.2f")
    min_correlacao = 0.05
    max_pvalor_coint = st.sidebar.number_input("P-valor de Cointegração Máximo", min_value=0.01, max_value=0.1, value=0.05, format="%.2f")

    multiplicador_atr_stop = 2
    multiplicador_atr_gain = 3

    tickers = obter_top_50_acoes_brasileiras()
    data_fim = datetime.now()
    data_inicio = data_fim - timedelta(days=dias)

    with st.spinner("Obtendo dados..."):
        dados = obter_dados(tickers, data_inicio, data_fim)

    with st.spinner("Analisando pares..."):
        df_resultados, total_pares = analisar_pares(dados, max_meia_vida, 1, max_pvalor_adf, min_correlacao, max_pvalor_coint)

    with st.spinner("Filtrando pares com base no Z-score mínimo..."):
        df_resultados_filtrados = filtrar_pares_por_zscore(dados, df_resultados, zscore_minimo)

    st.subheader("Resultados da Análise de Pares")
    st.write(f"Total de pares analisados: {total_pares}")
    st.write(f"Pares que passaram nos filtros estatísticos: {len(df_resultados)}")
    st.write(f"Pares que também atendem ao Z-score mínimo: {len(df_resultados_filtrados)}")

    if not df_resultados_filtrados.empty:
        df_display = df_resultados_filtrados.copy()
        df_display['Ação1'] = df_display['Ação1'].str.replace('.SA', '')
        df_display['Ação2'] = df_display['Ação2'].str.replace('.SA', '')
        st.dataframe(df_display)

        st.info("""
        **Nota sobre os valores de Beta:**
        O Beta na tabela acima é calculado usando todo o período de análise e representa uma visão histórica geral.
        O Beta Rotativo no gráfico abaixo usa uma janela móvel e mostra como a relação entre os ativos muda ao longo do tempo.
        """)

        st.subheader("Análise Detalhada de Par")
        opcoes_pares = [f"{row['Ação1']} - {row['Ação2']}" for _, row in df_display.iterrows()]
        par_selecionado = st.selectbox("Selecione um par para análise detalhada:", opcoes_pares)

        if par_selecionado:
            acao1, acao2 = par_selecionado.split(' - ')
            acao1_calc = acao1
            acao2_calc = acao2

            linha_selecionada = df_resultados_filtrados[
                (df_resultados_filtrados['Ação1'] == acao1_calc) & 
                (df_resultados_filtrados['Ação2'] == acao2_calc)
            ].iloc[0]

            direcao_acao1 = linha_selecionada['Direção Ação1']
            direcao_acao2 = linha_selecionada['Direção Ação2']

            ultimo_preco1 = dados['Close'][acao1_calc].iloc[-1]
            ultimo_preco2 = dados['Close'][acao2_calc].iloc[-1]

            if linha_selecionada['Beta'] > 0:
                valor_por_lado = investimento / 2
                qtd1 = ajustar_quantidade(round(valor_por_lado / ultimo_preco1))
                qtd2 = ajustar_quantidade(round(valor_por_lado / ultimo_preco2))
                
                valor_acao1 = qtd1 * ultimo_preco1
                valor_acao2 = qtd2 * ultimo_preco2
                
                # Verifica valor total e ajusta se necessário
                valor_total = abs(qtd1 * ultimo_preco1) + abs(qtd2 * ultimo_preco2)
                if valor_total > investimento:
                    fator_ajuste = investimento / valor_total
                    qtd1 = round(qtd1 * fator_ajuste)
                    qtd2 = round(qtd2 * fator_ajuste)
                    qtd1 = ajustar_quantidade(qtd1)
                    qtd2 = ajustar_quantidade(qtd2)
                        
                # Verifica se o valor financeiro está balanceado
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
            valor_total_acao2 = (qtd2 * ultimo_preco2) * (1 if direcao_acao2 == "Venda" else -1)

            impacto_liquido = valor_total_acao1 + valor_total_acao2

            valor_stop = calcular_valor_stop_atr(dados, acao1_calc, acao2_calc, qtd1, qtd2, direcao_acao1, direcao_acao2, multiplicador_atr=multiplicador_atr_stop)
            valor_gain = calcular_valor_gain_atr(dados, acao1_calc, acao2_calc, qtd1, qtd2, direcao_acao1, direcao_acao2, multiplicador_atr=multiplicador_atr_gain)

            valor_total_operacao_abs = (abs(qtd1) * ultimo_preco1 + abs(qtd2) * ultimo_preco2)
            percentual_stop = (valor_stop / valor_total_operacao_abs) * 100 if valor_total_operacao_abs != 0 else 0
            percentual_gain = (valor_gain / valor_total_operacao_abs) * 100 if valor_total_operacao_abs != 0 else 0

            fig_zscore = plotar_zscore(dados, acao1_calc, acao2_calc)
            fig_spread = plotar_spread(dados, acao1_calc, acao2_calc)

            spread = dados['Close'][acao1_calc] / dados['Close'][acao2_calc]
            df_spread_tab = pd.DataFrame({'Data': spread.index, 'Spread': spread.values})
            df_spread_tab['Data'] = df_spread_tab['Data'].dt.strftime('%Y-%m-%d')
            df_spread_tab = df_spread_tab.sort_values('Data', ascending=False).head(3)

            fig_beta_rotativo, media_beta, desvio_padrao_beta, beta_rotativo = plotar_beta_rotativo(dados, acao1_calc, acao2_calc, janela_beta)

            # Mostrar detalhes cruciais do par fora das abas
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

            st.subheader(f"Financeiro a ser Investido  R$ {investimento:,.2f}")

            df_qtd = pd.DataFrame({
                "Ação": [acao1, acao2],
                "Direção": [direcao_acao1, direcao_acao2],
                "Quantidade": [qtd1, qtd2],
                "Último Preço do ativo": [f"R$ {ultimo_preco1:.2f}", f"R$ {ultimo_preco2:.2f}"],
                "Valor Total": [
                    f"R$ {valor_total_acao1:.2f}",
                    f"R$ {valor_total_acao2:.2f}"
                ]
            })

            # Criação das abas (sem duplicar o resumo financeiro)
            tabs = st.tabs(["Z-score", "Resumo Financeiro", "Stop/Gain ATR", "Beta Rotativo", "Spread", "Margem e Custos"])

            with tabs[0]:
                st.subheader(f"Z-score para {acao1} vs {acao2}")
                st.plotly_chart(fig_zscore)

            with tabs[1]:
                st.subheader("Resumo Financeiro")
                st.table(df_qtd)
                st.write(f"**Impacto financeiro líquido na conta:** R$ {impacto_liquido:,.2f}")
                st.write("* As quantidades podem não coincidir exatamente com os valores financeiros, pois são arredondadas para manter a neutralidade da operação. Fique atento a isso.")

            with tabs[2]:
                st.subheader("Stop/Gain Sugestão")
                st.write(f"**Valor estimado de Stop-Loss:** R$ {valor_stop:.2f} ({percentual_stop:.2f}%)")
                st.write(f"**Valor estimado de Gain:** R$ {valor_gain:.2f} ({percentual_gain:.2f}%)")
                st.write("A analíse é uma sugestão de gain e stop levando em conta a volatilidade dos ativos, outra media é o retorno a média")    
            with tabs[3]:
                st.subheader(f"Beta Rotativo entre {acao1} e {acao2}")
                if fig_beta_rotativo is not None and beta_rotativo is not None and not beta_rotativo.empty:
                    st.plotly_chart(fig_beta_rotativo)
                    exibir_analise_beta(dados, acao1_calc, acao2_calc, janela_beta)
                else:
                    st.warning("Não foi possível calcular o beta rotativo com os dados disponíveis.")

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
                - Emolumentos B3:  3,25 a cada 10.000 negociados aproximadamente
                - BTC (aluguel) é calculado sobre o valor vendido 
                - A margem pode variar conforme a volatilidade dos ativos
                - Recomenda-se manter folga na margem para ajustes
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
