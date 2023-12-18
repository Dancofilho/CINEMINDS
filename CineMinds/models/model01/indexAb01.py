# Importando bibliotecas necessárias
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import time

# Carregando os dados dos arquivos csv fornecidos
movies = pd.read_csv('../../data/movies.csv')
ratings = pd.read_csv('../../data/ratings.csv')

# Pré-processamento dos dados
# Verificar a estrutura dos dados

# Medir o tempo de execução do pré-processamento
start_time = time.time()
print(movies.head())
print(movies.info())

print(ratings.head())
print(ratings.info())

# fazer as transformações necessárias
movies['movieId'] = movies['movieId'].astype(int)
# Dividir a coluna 'genres' em varias colunas
genres = movies['genres'].str.get_dummies(sep='|')
movies = pd.concat([movies, genres], axis=1)

# Remover colunas desnecessárias
ratings.drop('timestamp', axis=1, inplace=True)

# Tratar valores nulos
print(movies.isnull().sum())
print(ratings.isnull().sum())

# Se houver valores ausentes, você pode remover as linhas ou preencher os valores ausentes
# Por exemplo, para preencher com a média dos valores existentes:
ratings.fillna(ratings.mean(), inplace=True)


end_time = time.time()
execution_time = (end_time - start_time)
 
print(f"Tempo de execução: {execution_time:.2f} segundos.")

# Criar uma matriz de classificação de usuarios para filmes
user_movie_ratings = ratings.pivot(index='userId', columns='movieId', values='rating').fillna(0)

# Calcular a similaridade do cosseno entre os usuários
user_similarity = cosine_similarity(user_movie_ratings)

# Criar um dataframe com as similaridades e índices de usuário
user_similarity_df = pd.DataFrame(user_similarity, index=user_movie_ratings.index, columns=user_movie_ratings.index)

# Criar uma função para fazer recomendações para um usuário específico
# Medir o uso de memória 
import psutil

def recommend_movies(user_id, num_recommendations=10):
    # Obter os usuários mais similares ao usuário de referência
    similar_users = user_similarity_df[user_id].sort_values(ascending=False)[1:num_recommendations + 1].index

    # Obter os filmes classificados por usuários similares
    recommended_movies = user_movie_ratings.loc[similar_users].mean().sort_values(ascending=False).index

    return recommended_movies

if __name__ == '__main__' : 
    user_id = 1
    
    # Medir o uso de memória antes da execução da função recommend_movies
    process = psutil.Process()
    memory_before = process.memory_info().rss
    
    recommended_movies = recommend_movies(user_id)
    
    # Medir o uso de memória após a execução da função recommend_movies
    memory_after = process.memory_info().rss
    memory_usage = (memory_after - memory_before)

    print(f"Recomendações de filmes para o usuário {user_id}:")
    print(recommended_movies)
    print(f"Tempo de execução: {execution_time:.2f} segundos.")
    print(f"Uso de memória: {memory_usage} bytes.")

# Criar uma função para fazer recomendações para um usuário específico
def recommend_movies(user_id, num_recommendations=10):
    # Obter os usuários mais similares ao usuário de referência
    similar_users = user_similarity_df[user_id].sort_values(ascending=False)[1:num_recommendations + 1].index

    # Obter os filmes classificados por usuários similares
    recommended_movies = user_movie_ratings.loc[similar_users].mean().sort_values(ascending=False).index

    return recommended_movies

# Teste da função de recomendação para um usuário específico
user_id = 1
recommended_movies = recommend_movies(user_id)
print(f"Recomendações de filmes para o usuario {user_id}:")
print(recommended_movies)

