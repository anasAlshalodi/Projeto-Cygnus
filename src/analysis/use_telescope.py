import tensorflow as tf
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# --- Configurações e Funções Auxiliares (como antes) ---
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
    
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    generator = tf.keras.models.load_model('generator_model.keras')
    print("Modelo carregado com sucesso.")
except Exception as e:
    print(f"Erro: {e}. Certifique-se de que o 'generator_model.keras' está na pasta.")
    exit()

# --- Replicando a Análise para Encontrar os Melhores Pontos ---
print("\nGerando um lote de amostras para encontrar as melhores 'sementes'...")
num_samples = 500
latent_points = tf.random.normal([num_samples, TAMANHO_ENTRADA_GERADOR])
generated_tensors = generator.predict(latent_points, verbose=0)

qualities = []
for tensor in generated_tensors:
    numbers = decode_sequence(tensor)
    prime_count = sum(1 for num in numbers if is_prime(num))
    qualities.append(prime_count / len(numbers))

qualities = np.array(qualities)

# --- O "Telescópio": Focando nos Melhores Sonhos ---
# Encontra o índice da amostra que gerou o sonho de melhor qualidade
best_sample_index = np.argmax(qualities)
# Pega a "semente" (ponto no espaço latente) que corresponde a esse melhor sonho
best_latent_point = latent_points[best_sample_index]
best_quality = qualities[best_sample_index]

print(f"\nMelhor região encontrada: Qualidade de {best_quality:.0%}")
print("Gerando 5 novos 'sonhos de elite' a partir desta região...")

# Gera 5 novas sementes que são pequenas variações da nossa melhor semente
# Adicionamos um pouco de ruído para explorar a vizinhança
new_noise = tf.random.normal(shape=(5, TAMANHO_ENTRADA_GERADOR), stddev=0.1)
elite_latent_points = best_latent_point + new_noise

# Gera e decodifica os sonhos de elite
elite_tensors = generator.predict(elite_latent_points, verbose=0)

print("\n--- Sonhos Gerados pela Elite da IA ---")
for i, tensor in enumerate(elite_tensors):
    dreamed_numbers = decode_sequence(tensor)
    prime_count = sum(1 for num in dreamed_numbers if is_prime(num))
    quality = prime_count / len(dreamed_numbers)
    print(f"Sonho de Elite #{i+1} (Qualidade: {quality:.0%}): {dreamed_numbers}")