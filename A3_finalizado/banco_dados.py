import mysql.connector
import pandas as pd

# Conexão com o banco
conexao = mysql.connector.connect(
    host='localhost',
    database='analise_transito',
    user='root',
    password='23245623'
)

try:
    # Query para buscar os dados
    consulta = """
        SELECT * 
        FROM acidente a
        LEFT JOIN endereco e ON e.id = a.endereco_id
        LEFT JOIN cidade c ON c.id = e.cidade_id
        WHERE c.nome = 'Belo Horizonte'
    """
    
    # Carregar dados diretamente
    df = pd.read_sql(consulta, conexao)
    
    # Informações sobre os dados carregados
    print("✅ Dados carregados com sucesso!")
    print(f"\nTotal de registros: {len(df)}")
    print(f"Total de colunas: {len(df.columns)}")
    print(f"\nPrimeiras linhas:\n")
    print(df.head())
    print(f"\nColunas disponíveis:")
    print(df.columns.tolist())
    
except Exception as e:
    print(f"❌ Erro ao carregar dados: {e}")
    
finally:
    # Fechar conexão
    conexao.close()
    print("\n✅ Conexão fechada.")