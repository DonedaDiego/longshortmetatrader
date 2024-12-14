import streamlit as st
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller, coint
from statsmodels.regression.linear_model import OLS
from datetime import datetime, timedelta
import plotly.graph_objects as go

def initialize_mt5():
    if not mt5.initialize():
        st.error("Falha ao inicializar o MetaTrader5")
        return False
    return True

st.set_page_config(page_title="Long&Short - Learn Ai & Machine Learning Geminii", layout="wide")
st.sidebar.image(r"C:\Users\usuario\Desktop\Vscode\longshortmetatrader\assets\Logo.png", width=100)

@st.cache_data
def obter_top_50_acoes_brasileiras():
    return [acao for acao in ["ABEV3", "ALOS3","AMBP3", "AZUL4", "B3SA3", "BBAS3", "BBDC3", "BBDC4", "BHIA3", "BRAP4", "BRFS3", "BRKM5", 
        "CCRO3", "CMIG4", "CPLE6", "CRFB3", "CSAN3", "CSNA3", "EGIE3", "ELET3", "ELET6", "EQTL3", "GGBR4", 
        "GOAU4", "IRBR3", "ITSA4", "JBSS3", "KLBN11", "LREN3", "MGLU3", "MRFG3", "MULT3", "NTCO3", "PCAR3", 
        "PETR3", "PETR4", "PETZ3", "PRIO3", "RADL3", "RAIL3", "RAIZ4", "RENT3", "SANB11", "SAPR4", "SUZB3",
        "TAEE11", "TIMS3", "TRAD3", "UGPA3", "USIM5", "VALE3", "VIVT3", "VBBR3","WEGE3", "YDUQ3"]]

@st.cache_data
def obter_dados(tickers, data_inicio, data_fim):
    try:
        if not initialize_mt5():
            return pd.DataFrame()
        
        start_ts = int(data_inicio.timestamp())
        end_ts = int(data_fim.timestamp())
        
        dados_close = pd.DataFrame()
        
        for ticker in tickers:
            rates = mt5.copy_rates_range(ticker, mt5.TIMEFRAME_D1, start_ts, end_ts)
            if rates is not None and len(rates) > 0:
                df = pd.DataFrame(rates)
                df['time'] = pd.to_datetime(df['time'], unit='s')
                df.set_index('time', inplace=True)
                dados_close[ticker] = df['close']
        
        mt5.shutdown()
        return dados_close.ffill().dropna(how='all')
        
    except Exception as e:
        st.error(f"Erro ao obter dados: {e}")
        mt5.shutdown()
        return pd.DataFrame()

def calcular_metricas_par(dados, acao1, acao2):
    if len(dados[acao1]) < 2 or len(dados[acao2]) < 2:
        return None

    try:
        _, pvalor_coint, _ = coint(dados[acao1], dados[acao2])
        modelo = OLS(dados[acao1], dados[acao2]).fit()
        spread = dados[acao1] - modelo.params[0] * dados[acao2]
        
        correlacao = dados[acao1].corr(dados[acao2])
        half_life = calcular_meia_vida(spread)
        pvalor_adf = adfuller(spread)[1]
        
        preco_atual_1 = float(dados[acao1].iloc[-1])
        preco_atual_2 = float(dados[acao2].iloc[-1])

        return {
            'Correlação': float(correlacao),
            'Beta': float(modelo.params[0]),
            'Meia-vida': float(half_life) if half_life is not None else None,
            'P-valor Cointegração': float(pvalor_coint),
            'P-valor ADF': float(pvalor_adf),
            'Preço_Acao1': preco_atual_1,
            'Preço_Acao2': preco_atual_2
        }
    except Exception as e:
        print(f"Erro ao calcular métricas para {acao1}-{acao2}: {e}")
        return None

def calcular_meia_vida(spread):
    spread_lag = spread.shift(1)
    spread_diff = spread.diff()
    spread_lag, spread_diff = spread_lag[1:], spread_diff[1:]
    modelo_hl = OLS(spread_diff, spread_lag).fit()
    if modelo_hl.params[0] < 0:
        return -np.log(2) / modelo_hl.params[0]
    return None

def plotar_zscore(dados, acao1, acao2, data_entrada=None):
    if acao1 not in dados.columns or acao2 not in dados.columns:
        st.error(f"Dados não disponíveis para {acao1} ou {acao2}.")
        return None
    
    try:
        modelo = OLS(dados[acao1], dados[acao2]).fit()
        spread = dados[acao1] - modelo.params[0] * dados[acao2]
        zscore = (spread - spread.mean()) / spread.std()
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=zscore.index,
            y=zscore,
            mode='lines',
            name='Z-score'
        ))
        
        fig.add_hline(y=0, line_dash="dash", line_color="yellow", name="Média")
        fig.add_hline(y=2, line_dash="dot", line_color="green", name="+2")
        fig.add_hline(y=-2, line_dash="dot", line_color="green", name="-2")
        
        if data_entrada:
            data_str = data_entrada.strftime("%Y-%m-%d")
            fig.add_vline(
                x=data_str,
                line_dash="solid",
                line_color="blue",
                name="Data de Entrada"
            )
        
        fig.update_layout(
            title=f'Z-score para {acao1} vs {acao2}',
            xaxis_title='Data',
            yaxis_title='Z-score',
            showlegend=True
        )
        
        return fig
    except Exception as e:
        st.error(f"Erro ao calcular o Z-score: {e}")
        return None

def main():
    st.title("Long&Short - Learn Ai & Machine Learning Geminii")

    st.sidebar.header("Configurações")
    dias = st.sidebar.number_input("Número de dias para análise", min_value=30, max_value=240, value=60, step=1)
    
    tickers = obter_top_50_acoes_brasileiras()
    acao1 = st.sidebar.selectbox("Selecione a primeira ação:", tickers)
    acao2 = st.sidebar.selectbox("Selecione a segunda ação:", [t for t in tickers if t != acao1])
    
    data_entrada = st.sidebar.date_input(
        "Data de entrada na operação",
        value=None,
        min_value=datetime.now() - timedelta(days=240),
        max_value=datetime.now()
    )

    if st.sidebar.button("Realizar Análise"):
        data_fim = datetime.now()
        data_inicio = data_fim - timedelta(days=dias)
        dados = obter_dados([acao1, acao2], data_inicio, data_fim)

        if not dados.empty:
            metricas = calcular_metricas_par(dados, acao1, acao2)
            if metricas:
                st.subheader(f"Análise do Par {acao1} - {acao2}")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"Data da análise: {datetime.now().strftime('%Y-%m-%d')}")
                    st.write(f"Período analisado: {dias} dias")
                    st.write(f"Correlação: {metricas['Correlação']:.4f}")
                    st.write(f"Beta: {metricas['Beta']:.4f}")
                    st.write(f"Preço {acao1}: R$ {metricas['Preço_Acao1']:.2f}")
                
                with col2:
                    st.write(f"Meia-vida: {metricas['Meia-vida']:.2f} dias")
                    st.write(f"P-valor Cointegração: {metricas['P-valor Cointegração']:.4f}")
                    st.write(f"P-valor ADF: {metricas['P-valor ADF']:.4f}")
                    st.write(f"Preço {acao2}: R$ {metricas['Preço_Acao2']:.2f}")

                st.plotly_chart(plotar_zscore(dados, acao1, acao2, data_entrada))

    st.markdown("---")
    st.subheader('Disclaimer:')
    st.markdown("""
    O conteúdo deste material é destinado exclusivamente a fins informativos e educacionais. 
    As análises, opiniões e informações apresentadas não constituem, em nenhuma hipótese, recomendação de investimento. 
    Cada investidor deve fazer sua própria análise e tomar suas decisões de forma independente, considerando seu perfil, objetivos e tolerância a risco.
    
    Diego Doneda, Analista CNPI - 9668
    
    Geminii Learn Ai & Machine Learning 
    """)

if __name__ == "__main__":
    main()
