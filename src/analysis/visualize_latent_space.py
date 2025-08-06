import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import os ### CORREÇÃO: Importar a biblioteca os

# --- Configurações e Funções Auxiliares ---
BITS_POR_NUMERO = 16
TAMANHO_SEQUENCIA = 10
TAMANHO_ENTRADA_GERADOR = 100

def is_prime(n):
    if n <= 1: return False
    if n <= 3: return True
    if n % 2 == 0 or n % 3 == 0: return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0: return False
        i += 6
    return True

def decode_sequence(sequence):
    sequence_01 = (sequence + 1) / 2.0
    binary_matrix = np.round(sequence_01).astype(int)
    decoded_numbers = []
    for binary_list in binary_matrix:
        binary_string = "".join(map(str, binary_list))
        decoded_number = int(binary_string, 2)
        decoded_numbers.append(decoded_number)
    return decoded_numbers

# --- Carregando o Cérebro da IA ---
print("Carregando o modelo do Gerador treinado...")
try:
    ### CORREÇÃO: Construir o caminho a partir da raiz do projeto ###
    # __file__ é o caminho do script atual. Subimos dois níveis ('..', '..') para chegar na raiz.
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    model_path = os.path.join(project_root, 'generator_model.keras')
    
    generator = tf.keras.models.load_model(model_path)
    print(f"Modelo carregado de: {model_path}")
except Exception as e:
    print(f"Erro: {e}. Certifique-se de que o 'generator_model.keras' existe na pasta raiz do projeto.")
    exit()

# --- O resto do script continua igual ---
print("\nGerando um lote de 'sementes de sonhos' para mapeamento...")
num_samples = 500
latent_points = tf.random.normal([num_samples, TAMANHO_ENTRADA_GERADOR])
generated_tensors = generator.predict(latent_points, verbose=0)

print("Decodificando e avaliando cada sonho...")
qualities = []
for tensor in generated_tensors:
    numbers = decode_sequence(tensor)
    prime_count = sum(1 for num in numbers if is_prime(num))
    quality = prime_count / len(numbers) # Qualidade = % de primos no sonho
    qualities.append(quality)

qualities = np.array(qualities)

print("Projetando o espaço de 100 dimensões em um mapa 2D (isso pode levar um minuto)...")
tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000) 
points_2d = tsne.fit_transform(latent_points.numpy())

print("Renderizando o mapa da mente da IA...")
plt.figure(figsize=(12, 10))
scatter = plt.scatter(points_2d[:, 0], points_2d[:, 1], c=qualities, cmap='viridis', alpha=0.7)
plt.colorbar(scatter, label='Qualidade do Sonho (% de Primos)')
plt.title('Mapa do Espaço Latente (A Mente do Gerador)')
plt.xlabel('Dimensão Abstrata 1')
plt.ylabel('Dimensão Abstrata 2')
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()

print("\nMapa gerado. Observe as 'ilhas' ou 'clusters' de alta qualidade (cores mais brilhantes).")
print("Essas são as regiões na mente da IA que aprenderam a sonhar com mais precisão.")