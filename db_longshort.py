#banco de dados
from dataclasses import dataclass
from datetime import datetime
import sqlite3
import pandas as pd
import time

def listar_operacoes(usuario: str = None, status: str = None):
    start_time = time.time()
    conn = sqlite3.connect('longshort.db')
    query = 'SELECT * FROM operacoes'
    if usuario or status:
        query += ' WHERE'
        if usuario:
            query += f' usuario="{usuario}"'
        if status and usuario:
            query += f' AND status="{status}"'
        elif status:
            query += f' status="{status}"'
    df = pd.read_sql_query(query, conn)
    conn.close()
    print(f"Tempo para listar operações: {time.time() - start_time:.2f} segundos")
    return df


@dataclass
class Operacao:
    usuario: str
    data_entrada: str
    acao1: str
    acao2: str
    qtd1: int
    qtd2: int
    preco1: float
    preco2: float
    direcao1: str
    direcao2: str
    meia_vida: float
    periodo: int
    zscore: float
    data_limite: str
    status: str = "Aberta"
    data_saida: str = None
    resultado: float = 0

def inicializar_db():
    """Inicializa o banco de dados apenas quando necessário."""
    try:
        conn = sqlite3.connect('longshort.db')
        cursor = conn.cursor()
        cursor.execute('''CREATE TABLE IF NOT EXISTS operacoes (
                         id INTEGER PRIMARY KEY,
                         usuario TEXT,
                         data_entrada TEXT,
                         acao1 TEXT,
                         acao2 TEXT,
                         qtd1 INTEGER,
                         qtd2 INTEGER,
                         preco1 REAL,
                         preco2 REAL,
                         direcao1 TEXT,
                         direcao2 TEXT, 
                         meia_vida REAL,
                         periodo INTEGER,
                         zscore REAL,
                         data_limite TEXT,
                         status TEXT DEFAULT 'Aberta',
                         data_saida TEXT,
                         resultado REAL DEFAULT 0)''')
        conn.commit()
    finally:
        conn.close()

def cadastrar_operacao(op: Operacao):
    """Cadastra uma operação no banco de dados."""
    conn = sqlite3.connect('longshort.db')
    try:
        c = conn.cursor()
        c.execute('''INSERT INTO operacoes 
                     (usuario, data_entrada, acao1, acao2, qtd1, qtd2, preco1, preco2, 
                      direcao1, direcao2, meia_vida, periodo, zscore, data_limite, 
                      status, data_saida, resultado)
                     VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                  (op.usuario, op.data_entrada, op.acao1, op.acao2, op.qtd1, op.qtd2, 
                   op.preco1, op.preco2, op.direcao1, op.direcao2, op.meia_vida, 
                   op.periodo, op.zscore, op.data_limite, op.status, op.data_saida, 
                   op.resultado))
        conn.commit()
    finally:
        conn.close()

def listar_operacoes(usuario: str = None, status: str = None, limite: int = 100):
    """Lista operações do banco de dados, com filtros opcionais e limite de registros."""
    conn = sqlite3.connect('longshort.db')
    try:
        query = 'SELECT * FROM operacoes'
        filtros = []
        parametros = []

        if usuario:
            filtros.append("usuario = ?")
            parametros.append(usuario)
        if status:
            filtros.append("status = ?")
            parametros.append(status)

        if filtros:
            query += " WHERE " + " AND ".join(filtros)
        query += f" LIMIT {limite}"

        df = pd.read_sql_query(query, conn, params=parametros)
        return df
    finally:
        conn.close()

def fechar_operacao(id: int, data_saida: str, resultado: float):
    """Atualiza uma operação para status 'Fechada' com a data de saída e resultado."""
    conn = sqlite3.connect('longshort.db')
    try:
        c = conn.cursor()
        c.execute('''UPDATE operacoes 
                     SET status = ?, data_saida = ?, resultado = ?
                     WHERE id = ?''',
                  ('Fechada', data_saida, resultado, id))
        conn.commit()
    finally:
        conn.close()

def deletar_operacao(id: int):
    """Deleta uma operação do banco de dados pelo ID."""
    conn = sqlite3.connect('longshort.db')
    try:
        c = conn.cursor()
        c.execute('DELETE FROM operacoes WHERE id = ?', (id,))
        conn.commit()
    finally:
        conn.close()

# Inicializa o banco de dados apenas se este script for executado diretamente.
if __name__ == "__main__":
    inicializar_db()
    print("Banco de dados inicializado com sucesso.")