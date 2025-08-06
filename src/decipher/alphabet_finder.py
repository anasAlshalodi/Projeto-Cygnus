import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Reutilizando nossa função de geração de sinal
def generate_structured_signal(length=1000):
    word_a = [1, 0, 1, 0]; word_b = [0, 0, 1, 1]
    separator = [0, 0, 0, 0]; word_c = [1, 1, 1, 1]
    phrase_1 = word_a + separator + word_b
    phrase_2 = word_b + separator + word_a + separator + word_c
    base = phrase_1 + phrase_2
    repetitions = length // len(base)
    signal = np.tile(base, repetitions)
    noise_mask = np.random.rand(len(signal)) < 0.01
    signal[noise_mask] = 1 - signal[noise_mask]
    return signal[:length]

def find_optimal_clusters(signal, word_length=4, max_clusters=10):
    """
    Testa diferentes números de clusters e calcula métricas para encontrar o ideal.
    """
    # Fatiar o sinal em janelas
    trimmed_length = len(signal) - (len(signal) % word_length)
    windows = signal[:trimmed_length].reshape(-1, word_length)

    inertias = []
    silhouette_scores = []
    cluster_range = range(2, max_clusters + 1)

    print(f"Testando de 2 a {max_clusters} clusters...")

    for k in cluster_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
        kmeans.fit(windows)

        # Métrica para o Método do Cotovelo
        inertias.append(kmeans.inertia_)

        # Métrica da Pontuação da Silhueta
        score = silhouette_score(windows, kmeans.labels_)
        silhouette_scores.append(score)
        print(f"  k={k}, Silhueta={score:.4f}, Inércia={kmeans.inertia_:.2f}")

    # Plotar os resultados
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Gráfico do Cotovelo
    ax1.plot(cluster_range, inertias, 'bo-')
    ax1.set_xlabel('Número de Clusters (k)')
    ax1.set_ylabel('Inércia')
    ax1.set_title('Método do Cotovelo para k Ótimo')
    ax1.grid(True)

    # Gráfico da Silhueta
    ax2.plot(cluster_range, silhouette_scores, 'ro-')
    ax2.set_xlabel('Número de Clusters (k)')
    ax2.set_ylabel('Pontuação da Silhueta')
    ax2.set_title('Pontuação da Silhueta para k Ótimo')
    ax2.grid(True)

    plt.suptitle('Análise para Determinação do Número de Palavras no Alfabeto')
    plt.show()


if __name__ == "__main__":
    print("🤖 Iniciando o Localizador Automático de Alfabeto (Fase 2.5) 🤖")
    structured_signal = generate_structured_signal()
    find_optimal_clusters(structured_signal, word_length=4, max_clusters=10)