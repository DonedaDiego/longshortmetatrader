import streamlit as st
from datetime import datetime
from db_longshort import *

def main():
   inicializar_db()
   
   st.set_page_config(page_title="Long&Short Tracker", layout="wide")
   
   st.markdown("""
       <h1 style='text-align: center; color: #ffffff;'>
           📈 Long&Short - Acompanhamento de Posições
       </h1>
       <hr style='margin: 1rem 0;'>
   """, unsafe_allow_html=True)

   tab_cadastro, tab_consulta, tab_gerenciar = st.tabs(["Cadastrar", "Consultar", "Gerenciar"])

   with tab_cadastro:
    if 'form_submitted' not in st.session_state:
        st.session_state.form_submitted = False
    
    with st.form(key="cadastro_form"):
        # Valores default vazios
        nome = st.text_input("Nome do Operador", value="" if st.session_state.form_submitted else st.session_state.get('nome', ''))
        col1, col2 = st.columns(2)
        
        with col1:
            acao1 = st.text_input("Ativo 1", value="" if st.session_state.form_submitted else st.session_state.get('acao1', ''))
            qtd1 = st.number_input("Quantidade 1", min_value=1, value=1 if st.session_state.form_submitted else st.session_state.get('qtd1', 1))
            preco1 = st.number_input("Preço 1", min_value=0.01, value=0.01 if st.session_state.form_submitted else st.session_state.get('preco1', 0.01))
            direcao1 = st.selectbox("Direção 1", ["LONG", "SHORT"])
        
        with col2:
            acao2 = st.text_input("Ativo 2", value="" if st.session_state.form_submitted else st.session_state.get('acao2', ''))
            qtd2 = st.number_input("Quantidade 2", min_value=1, value=1 if st.session_state.form_submitted else st.session_state.get('qtd2', 1))
            preco2 = st.number_input("Preço 2", min_value=0.01, value=0.01 if st.session_state.form_submitted else st.session_state.get('preco2', 0.01))
            direcao2 = st.selectbox("Direção 2", ["LONG", "SHORT"])
        
        meia_vida = st.number_input("Meia Vida", min_value=0.1, value=0.1 if st.session_state.form_submitted else st.session_state.get('meia_vida', 0.1))
        periodo = st.number_input("Período", min_value=1, value=1 if st.session_state.form_submitted else st.session_state.get('periodo', 1))
        zscore = st.number_input("Z-Score", value=0.0 if st.session_state.form_submitted else st.session_state.get('zscore', 0.0))
        data_limite = st.date_input("Data Limite")
                        
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            submit_button = st.form_submit_button(label="Cadastrar")
        with col_btn2:
            clear_button = st.form_submit_button(label="Limpar")

        if clear_button:
            for key in st.session_state.keys():
                if key.startswith(("nome", "acao", "qtd", "preco", "direcao", "meia_vida", "periodo", "zscore")):
                    del st.session_state[key]
            st.rerun()
        
        if submit_button:
            try:
                data_entrada = datetime.now()
                data_limite = datetime.combine(data_limite, datetime.min.time())
                
                op = Operacao(
                    usuario=nome.upper(),
                    data_entrada=data_entrada,
                    acao1=acao1.upper(),
                    acao2=acao2.upper(),
                    qtd1=qtd1,
                    qtd2=qtd2,
                    preco1=preco1,
                    preco2=preco2,
                    direcao1=direcao1,
                    direcao2=direcao2,
                    meia_vida=meia_vida,
                    periodo=periodo,
                    zscore=zscore,
                    data_limite=data_limite
                )
                cadastrar_operacao(op)
                st.success("Operação cadastrada com sucesso!")
                st.session_state.form_submitted = True
                st.rerun()
            except Exception as e:
                st.error(f"Erro ao cadastrar operação: {str(e)}")

   with tab_consulta:
       nome_consulta = st.text_input("Nome do Operador para consulta").upper()
       status = st.selectbox("Status", ["Todas", "Aberta", "Fechada"])
       
       if st.button("Consultar"):
           status_query = None if status == "Todas" else status
           df = listar_operacoes(nome_consulta, status_query)
           if not df.empty:
               st.dataframe(df)
           else:
               st.info("Nenhuma operação encontrada")

   with tab_gerenciar:
       operacoes = listar_operacoes(status="Aberta")
       if not operacoes.empty:
           st.write("### Operações em Aberto")
           for idx, op in operacoes.iterrows():
               with st.expander(f"{op['acao1']} vs {op['acao2']} - {op['data_entrada']}"):
                   col1, col2 = st.columns(2)
                   with col1:
                        if st.button("Fechar Operação", key=f"fechar_{idx}"):
                           deletar_operacao(op['id'])
                           st.success("Operação fechada!")
                           st.rerun()
                   with col2:
                       if st.button("Deletar Operação", key=f"deletar_{idx}"):
                           deletar_operacao(op['id'])
                           st.success("Operação deletada!")
                           st.rerun()
       else:
           st.info("Não há operações em aberto")

if __name__ == "__main__":
   main()