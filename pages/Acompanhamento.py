import streamlit as st
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import coint
from statsmodels.regression.linear_model import OLS
from datetime import datetime, timedelta, time
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from db_longshort import inicializar_db, listar_operacoes

def initialize_mt5():
    """Inicializa a conexão com o MetaTrader5"""
    if not mt5.initialize():
        st.error("❌ Falha ao inicializar o MetaTrader5")
        return False
    return True


def obter_top_50_acoes_brasileiras():
    """Retorna lista das principais ações brasileiras"""
    return ["ABEV3", "ALOS3", "ALUP3", "ALUP4", "AURE3", "AZZA3", "B3SA3", "BBAS3", "BBDC3", "BBDC4", "BBSE3", "BPAC11",  
      "BPAN4", "BRAP3", "BRAP4", "BRAV3", "BRFS3",  "CCRO3", "CMIG3", "CMIG4", "CMIN3", 
      "CPFE3", "CPLE3", "CPLE5", "CPLE6", "CRFB3", "CSAN3", "CSMG3", "CSNA3", "CXSE3", "CYRE3",  "ELET3", "ELET6", 
      "EGIE3", "EMBR3", "ENEV3", "ENGI11", "FLRY3","ENGI3", "ENGI4", "EQTL3", "GGBR3", "GGBR4", "GOAU4", "HAPV3", "HYPE3", 
      "ITSA3", "ITSA4", "ITUB3", "ITUB4", "JBSS3", "KLBN11", "KLBN3", "KLBN4", "LREN3", "MDIA3",  "NEOE3", 
      "NTCO3", "PETR3", "PETR4", "PRIO3", "PSSA3", "RAIL3", "RAIZ4", "RDOR3", "RENT3", "SANB11", "SANB4", "SBSP3", "SUZB3", 
      "VBBR3", "VALE3", "VIVT3", "WEGE3", "UGPA3"]
    

 
 
# "ABEV3", "ASAI3", "ALOS3", "AMBP3", "AZUL4", "B3SA3", "BBAS3", "BBDC3", "BBDC4", "BBSE3", "BRAP4", "BRFS3",
#             "BRKM5", "BPAN4", "CCRO3", "CMIG4", "CPLE6", "CRFB3", "CSAN3", "CSNA3", "EGIE3", "ELET3", "ELET6", "EQTL3", 
#             "FLRY3", "GGBR4", "GOAU4", "IRBR3", "ITSA4", "JBSS3", "KLBN11", "LREN3", "MGLU3", "MOVI3", "MRFG3", "MULT3", 
#             "NTCO3", "PCAR3", "PETR3", "PETR4", "PETZ3", "PRIO3", "RADL3", "RAIL3", "RAIZ4", "RENT3", "SANB11", "SAPR4", 
#             "SUZB3", "TAEE11", "TIMS3", "UGPA3", "USIM5", "VALE3", "VIVT3", "VBBR3", "WEGE3", "YDUQ3"      
      
def obter_dados(tickers, data_inicio, data_fim):
    """Obtém dados históricos do MetaTrader5"""
    if not initialize_mt5():
        return pd.DataFrame()
    
    # Sempre pegar 240 dias (período máximo) de dados
    data_inicio_max = data_fim - timedelta(days=240)
    
    start_ts = int(datetime.combine(data_inicio_max, time.min).timestamp())
    end_ts = int(datetime.combine(data_fim, time.max).timestamp())
    
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

def calcular_zscore(dados, acao1, acao2, periodo_cointegracao):

    dados_periodo = dados.tail(periodo_cointegracao)
    
    modelo = OLS(dados_periodo[acao1], dados_periodo[acao2]).fit()
    spread = dados[acao1] - modelo.params[0] * dados[acao2]
    return (spread - spread.mean()) / spread.std()

def calcular_stop_gain_zscore(dados, acao1, acao2, qtd1, qtd2, direcao1, direcao2, zscore_atual, 
                             stop_desvios, gain_desvios):
    preco_atual1 = dados[acao1].iloc[-1]
    preco_atual2 = dados[acao2].iloc[-1]
    
    modelo = OLS(dados[acao1], dados[acao2]).fit()
    beta = modelo.params[0]
    spread = dados[acao1] - beta * dados[acao2]
    spread_std = spread.std()
    
    # Variação nos preços por desvio
    delta_stop = abs(stop_desvios - zscore_atual) * spread_std / zscore_atual
    delta_gain = abs(gain_desvios - zscore_atual) * spread_std / zscore_atual
    
    # Ajuste direcional
    stop_mult = -1 if direcao1 == "LONG" else 1
    gain_mult = 1 if direcao1 == "LONG" else -1
    
    # Cálculo do valor financeiro
    stop_value = (delta_stop * qtd1 * preco_atual1 * stop_mult) + (delta_stop * qtd2 * preco_atual2 * -stop_mult)
    gain_value = (delta_gain * qtd1 * preco_atual1 * gain_mult) + (delta_gain * qtd2 * preco_atual2 * -gain_mult)
    
    return abs(stop_value), abs(gain_value)

def plotar_analise_completa(dados, acao1, acao2, periodo_cointegracao, data_entrada=None):
    dados_coint = dados.tail(periodo_cointegracao).copy()
    
    # Criar subplots
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=(
            f'Preços Normalizados: {acao1} vs {acao2} (Período: {periodo_cointegracao} dias)',
            f'Z-Score da Operação'
        ),
        vertical_spacing=0.15,
        row_heights=[0.6, 0.4]
    )
    
    # Normalizar preços para base 100
    preco_norm1 = dados_coint[acao1] / dados_coint[acao1].iloc[0] * 100
    preco_norm2 = dados_coint[acao2] / dados_coint[acao2].iloc[0] * 100
    
    # Plot preços normalizados
    fig.add_trace(
        go.Scatter(
            x=dados_coint.index,
            y=preco_norm1,
            name=acao1,
            line=dict(color='blue'),
            showlegend=True
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=dados_coint.index,
            y=preco_norm2,
            name=acao2,
            line=dict(color='red'),
            showlegend=True
        ),
        row=1, col=1
    )
    
    # Calcular e plotar z-score
    zscore = calcular_zscore(dados, acao1, acao2, periodo_cointegracao)
    zscore_plot = zscore.tail(periodo_cointegracao)
    
    fig.add_trace(
        go.Scatter(
            x=zscore_plot.index,
            y=zscore_plot,
            name='Z-Score',
            line=dict(color='purple'),
            showlegend=True
        ),
        row=2, col=1
    )
    
    # Adicionar linhas de referência para o Z-Score
    fig.add_hline(y=0, line_dash="dash", line_color="yellow", row=2, col=1)
    fig.add_hline(y=3, line_dash="dot", line_color="red", row=2, col=1)
    fig.add_hline(y=2, line_dash="dot", line_color="green", row=2, col=1)
    fig.add_hline(y=-2, line_dash="dot", line_color="green", row=2, col=1)
    fig.add_hline(y=-3, line_dash="dot", line_color="red", row=2, col=1)
    
    # Adicionar linha vertical da data de entrada se fornecida
    if data_entrada:
        fig.add_vline(
            x=data_entrada,
            line_dash="solid",
            line_color="green",
            line_width=1,
            row='all'
        )
    
    # Configurar layout
    fig.update_layout(
        height=700,
        showlegend=True,
        title_text=f"Análise Completa da Operação - {acao1} vs {acao2}",
        title_x=0.5,
        template="plotly_dark",
        margin=dict(t=100),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # Configurar eixos
    fig.update_yaxes(title_text="Preço Base 100", row=1, col=1)
    fig.update_yaxes(title_text="Z-Score", row=2, col=1)
    fig.update_xaxes(title_text="Data", row=2, col=1)
    
    # Adicionar anotações para as bandas do Z-Score
    fig.add_annotation(
        text="Zona de Venda",
        xref="paper", yref="y2",
        x=1.02, y=2.5,
        showarrow=False,
        font=dict(size=10, color="red")
    )
    
    fig.add_annotation(
        text="Zona de Compra",
        xref="paper", yref="y2",
        x=1.02, y=-2.5,
        showarrow=False,
        font=dict(size=10, color="green")
    )
    
    return fig

def calcular_lucro(dados, acao1, acao2, preco_entrada1, preco_entrada2, qtd1, qtd2, direcao1, direcao2):
    """Calcula o resultado da operação"""
    preco_atual1 = dados[acao1].iloc[-1]
    preco_atual2 = dados[acao2].iloc[-1]
    lucro_acao1 = (preco_atual1 - preco_entrada1) * qtd1 * (1 if direcao1 == "LONG" else -1)
    lucro_acao2 = (preco_atual2 - preco_entrada2) * qtd2 * (1 if direcao2 == "LONG" else -1)
    return lucro_acao1 + lucro_acao2, lucro_acao1, lucro_acao2, preco_atual1, preco_atual2

def criar_card_metrica(titulo, valor, delta=None, prefix="R$"):
    """Cria card personalizado para métricas"""
    with st.container():
        st.markdown(
            f"""
            <div style='padding: 1rem; background-color: #1E1E1E; border-radius: 0.5rem; margin: 0.5rem 0;'>
                <h3 style='margin: 0; color: #CCCCCC;'>{titulo}</h3>
                <p style='font-size: 1.5rem; margin: 0.5rem 0; color: white;'>
                    {prefix} {valor:,.2f}
                </p>
                {f"<p style='margin: 0; color: {'green' if delta >= 0 else 'red'};'>{delta:+.2f}%</p>" if delta is not None else ""}
            </div>
            """,
            unsafe_allow_html=True
        )

def main():
   st.set_page_config(
       page_title="Long&Short Tracker",
       layout="wide",
       initial_sidebar_state="expanded"
   )
   
   st.markdown("""
       <h1 style='text-align: center; color: #ffffff;'>
           📈 Long&Short - Acompanhamento de Posições
       </h1>
       <hr style='margin: 1rem 0;'>
   """, unsafe_allow_html=True)

   tab_acompanhamento, tab_simulacao = st.tabs(["📊 Acompanhamento", "🎯 Simulação"])

   with st.sidebar:
       st.header("🔧 Configurações")
       
       tickers = obter_top_50_acoes_brasileiras()
       col1, col2 = st.columns(2)
       with col1:
           acao1 = st.selectbox("Ação 1:", tickers)
       with col2:
           acao2 = st.selectbox("Ação 2:", [t for t in tickers if t != acao1])
       
       dias = st.selectbox(
           "Período de Cointegração (dias)",
           [60, 90, 120, 140, 180, 200, 240],
           index=0
       )
       
       st.markdown("### 📊 Configuração das Posições")
       
       st.subheader(f"🔵 {acao1}")
       qtd1 = st.number_input(f"Quantidade", min_value=1, value=100, key="qtd1")
       preco_entrada1 = st.number_input(f"Preço de entrada", min_value=0.01, value=10.0, key="preco1")
       direcao1 = st.radio(f"Direção", ["LONG", "SHORT"], key="dir1")
       
       st.subheader(f"🔴 {acao2}")
       qtd2 = st.number_input(f"Quantidade", min_value=1, value=100, key="qtd2")
       preco_entrada2 = st.number_input(f"Preço de entrada", min_value=0.01, value=10.0, key="preco2")
       direcao2 = st.radio(f"Direção", ["LONG", "SHORT"], key="dir2")
       
       st.markdown("### 🎯 Stops e Gains")
       stop_loss = st.number_input("Stop-Loss (R$)", min_value=0.01, value=500.0)
       gain = st.number_input("Gain (R$)", min_value=0.01, value=1000.0)
       
       data_entrada = st.date_input("Data de Entrada", datetime.now().date())

       st.markdown("### 📏 Simulação por Desvios")
       stop_desvios = st.slider("Stop Loss (Desvios)", min_value=-5.0, max_value=5.0, value=-2.5, step=0.1)
       gain_desvios = st.slider("Take Profit (Desvios)", min_value=-5.0, max_value=5.0, value=0.0, step=0.1)
       
       with st.sidebar:
    
        st.markdown("### 📂 Carregar Operação")
        usar_banco = st.checkbox("Carregar dados do banco", value=False)
        
        if usar_banco:
            operacoes = listar_operacoes(status="Aberta")
            if not operacoes.empty:
                op_selecionada = st.selectbox(
                    "Selecione uma operação",
                    options=operacoes.apply(lambda x: f"{x['acao1']} vs {x['acao2']} - {x['data_entrada']}", axis=1)
                )
                
                if op_selecionada:
                    idx = operacoes.apply(lambda x: f"{x['acao1']} vs {x['acao2']} - {x['data_entrada']}", axis=1) == op_selecionada
                    op = operacoes[idx].iloc[0]
                    acao1 = op['acao1']
                    acao2 = op['acao2']
                    qtd1 = op['qtd1']
                    qtd2 = op['qtd2']
                    preco_entrada1 = op['preco1']
                    preco_entrada2 = op['preco2']
                    direcao1 = op['direcao1']
                    direcao2 = op['direcao2']
                    dias = op['periodo']
            else:
                # Campos de entrada manual existentes
                acao1 = st.selectbox("Ação 1:", tickers)
                acao2 = st.selectbox("Ação 2:", [t for t in tickers if t != acao1])           
   
   if st.sidebar.button("📊 Atualizar Análise", use_container_width=True):
       data_fim = datetime.now().date()
       data_inicio = data_fim - timedelta(days=240)
       dados = obter_dados([acao1, acao2], data_inicio, data_fim)

       if not dados.empty:
           zscore = calcular_zscore(dados, acao1, acao2, dias)
           lucro_total, lucro_acao1, lucro_acao2, preco_atual1, preco_atual2 = calcular_lucro(
               dados, acao1, acao2, preco_entrada1, preco_entrada2, qtd1, qtd2, direcao1, direcao2
           )

           with tab_acompanhamento:
               col1, col2, col3 = st.columns(3)
               with col1:
                   criar_card_metrica(f"{acao1} - Resultado", lucro_acao1)
               with col2:
                   criar_card_metrica(f"{acao2} - Resultado", lucro_acao2)
               with col3:
                   criar_card_metrica("Resultado Total", lucro_total)
               
               if lucro_total >= gain:
                   st.success("🎯 Take Profit Atingido!")
               elif lucro_total <= -stop_loss:
                   st.error("⚠️ Stop Loss Atingido!")
               else:
                   st.info("🔄 Operação em Andamento")
               
               fig = plotar_analise_completa(dados, acao1, acao2, dias, data_entrada)
               st.plotly_chart(fig, use_container_width=True)
               
               with st.expander("📊 Detalhes da Operação"):
                   col1, col2 = st.columns(2)
                   with col1:
                       st.metric(
                           f"{acao1} - Preço Atual",
                           f"R$ {preco_atual1:.2f}",
                           f"{((preco_atual1 - preco_entrada1) / preco_entrada1 * 100):.2f}%"
                       )
                       st.metric("Quantidade", qtd1)
                       st.metric("Direção", direcao1)
                   with col2:
                       st.metric(
                           f"{acao2} - Preço Atual",
                           f"R$ {preco_atual2:.2f}",
                           f"{((preco_atual2 - preco_entrada2) / preco_entrada2 * 100):.2f}%"
                       )
                       st.metric("Quantidade", qtd2)
                       st.metric("Direção", direcao2)
                       
                       

           with tab_simulacao:
               stop_valor, gain_valor = calcular_stop_gain_zscore(
                   dados, acao1, acao2, qtd1, qtd2, direcao1, direcao2, 
                   zscore.iloc[-1], stop_desvios, gain_desvios
               )
               st.markdown("### 📊 Simulação de Stop/Gain por Desvios")
               col1, col2 = st.columns(2)
               with col1:
                   criar_card_metrica("Stop Loss Estimado", stop_valor)
               with col2:
                   criar_card_metrica("Take Profit Estimado", gain_valor)
               
               st.markdown("### 📈 Análise de Desvios")
               st.write(f"Z-Score Atual: {zscore.iloc[-1]:.2f}")
               st.write(f"Stop Loss definido em {stop_desvios} desvios")
               st.write(f"Take Profit definido em {gain_desvios} desvios")

if __name__ == "__main__":
   main()