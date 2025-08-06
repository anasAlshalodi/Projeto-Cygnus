import tensorflow as tf
from tensorflow.keras import optimizers
import numpy as np
import matplotlib.pyplot as plt

# Importa as arquiteturas do nosso novo módulo central
from src.models.wgan_model import WGAN, build_generator, build_discriminator

# --- Parâmetros de Treinamento ---
LATENT_DIM = 128
EPOCHS = 100 # Ajuste conforme necessário para uma boa convergência
BATCH_SIZE = 64
IMAGE_SIZE = 32
MODEL_NAME = "prime_generator.keras"
NUM_IMAGES = 5000 # Quantidade de imagens a serem geradas para o dataset

# --- Geração dos Dados ---
def is_prime(n):
    if n < 2: return False
    for i in range(2, int(np.sqrt(n)) + 1):
        if n % i == 0: return False
    return True

def load_prime_data():
    """Gera imagens visualizando números primos em uma grade."""
    print(f"Gerando {NUM_IMAGES} imagens de números primos...")
    images = []
    primes = [i for i in range(IMAGE_SIZE * IMAGE_SIZE) if is_prime(i)]
    
    for _ in range(NUM_IMAGES):
        img = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
        # Escolhe um número aleatório de primos para exibir
        num_primes_to_show = np.random.randint(10, len(primes) // 2)
        indices_to_show = np.random.choice(len(primes), num_primes_to_show, replace=False)
        
        for index in indices_to_show:
            p = primes[index]
            row, col = p // IMAGE_SIZE, p % IMAGE_SIZE
            img[row, col] = 1.0 # Marca a posição do primo
            
        images.append(img)
        
    images = np.array(images).reshape((-1, IMAGE_SIZE, IMAGE_SIZE, 1))
    # Normalizar imagens para o intervalo [-1, 1] (para a ativação tanh do gerador)
    images = (images * 2) - 1
    
    dataset = tf.data.Dataset.from_tensor_slices(images)
    dataset = dataset.shuffle(buffer_size=NUM_IMAGES).batch(BATCH_SIZE)
    return dataset

# --- Funções de Perda WGAN ---
def discriminator_loss(real_img, fake_img):
    return tf.reduce_mean(fake_img) - tf.reduce_mean(real_img)

def generator_loss(fake_img):
    return -tf.reduce_mean(fake_img)

# --- Lógica de Treinamento ---
if __name__ == '__main__':
    print(f"--- Iniciando Treinamento para: {MODEL_NAME} ---")

    # 1. Construir os modelos
    generator = build_generator(LATENT_DIM)
    discriminator = build_discriminator((IMAGE_SIZE, IMAGE_SIZE, 1))
    
    # 2. Instanciar e compilar o modelo WGAN
    wgan = WGAN(discriminator=discriminator, generator=generator, latent_dim=LATENT_DIM)
    
    d_optimizer = optimizers.Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.9)
    g_optimizer = optimizers.Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.9)

    wgan.compile(
        d_optimizer=d_optimizer,
        g_optimizer=g_optimizer,
        d_loss_fn=discriminator_loss,
        g_loss_fn=generator_loss,
    )

    # 3. Carregar dados e treinar
    train_dataset = load_prime_data()
    print(f"\nIniciando treinamento por {EPOCHS} épocas...")
    wgan.fit(train_dataset, epochs=EPOCHS)

    # 4. Salvar o modelo gerador treinado
    generator.save(MODEL_NAME)
    print(f"\n✅ Modelo gerador de primos salvo em: {MODEL_NAME}")