import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import time
import psutil
import cProfile

# Medir o uso de CPU e memória inicial do processo

# Obtém o uso de CPU e memória atual
cpu_usage = psutil.cpu_percent()
memory_usage = psutil.virtual_memory().percent

print(f"Uso de CPU: {cpu_usage}%")
print(f"Uso de memória: {memory_usage}%")

# Medir o tempo de execução do modelo de recomendação baseado em conteúdo
start_time_aplication = time.time()

# Carregar os dados dos filmes
# Tempo para carregar os dados
start_time = time.time()
movies = pd.read_csv('../../data/movies.csv')

# Criar um vetorizador TF-IDF para processar os gêneros dos filmes
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(movies['genres'].fillna(''))

# Calcular a similaridade de cosseno entre os filmes
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# Fim da contagem de tempo
end_time = time.time()

# Cálculo do tempo de execução
execution_time = end_time - start_time

print(f"Tempo de execução: {execution_time} segundos")

def recommend_content_based(movie_title, cosine_sim=cosine_sim, movies=movies):
    # Encontrar o índice do filme fornecido
    idx = movies.index[movies['title'] == movie_title].tolist()[0]

    # Calcular as pontuações de similaridade de cosseno para todos os filmes
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Ordenar os filmes com base nas pontuações de similaridade
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Obter os índices dos 10 filmes mais similares
    movie_indices = [i[0] for i in sim_scores[1:11]]

    # Retornar os títulos dos filmes recomendados
    return movies['title'].iloc[movie_indices]

# Executar o perfil de desempenho do modelo de recomendação baseado em conteúdo
cProfile.run('recommend_content_based("Toy Story (1995)")')

# Exemplo: fazer recomendações para o filme "Toy Story (1995)"
movie_title = "Toy Story (1995)"
recommended_movies = recommend_content_based(movie_title)
print(recommended_movies)

# Fim da contagem de tempo da aplicação do modelo de recomendação baseado em conteúdo
end_time_aplication = time.time()

# Cálculo do tempo de execução da aplicação do modelo de recomendação baseado em conteúdo
execution_time_aplication = end_time_aplication - start_time_aplication

print(f"Tempo de execução da aplicação: {execution_time_aplication} segundos")

# Medir o uso de CPU e memória final do processo

# Obtém o uso de CPU e memória atual
cpu_usage = psutil.cpu_percent()
memory_usage = psutil.virtual_memory().percent

print(f"Uso de CPU: {cpu_usage}%")
print(f"Uso de memória: {memory_usage}%")

