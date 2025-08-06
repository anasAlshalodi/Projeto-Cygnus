import tensorflow as tf
from tensorflow.keras import optimizers
import numpy as np
import matplotlib.pyplot as plt

from src.models.wgan_model import WGAN, build_generator, build_discriminator

# --- Parâmetros ---
LATENT_DIM = 128
EPOCHS = 100
BATCH_SIZE = 64
IMAGE_SIZE = 32
MODEL_NAME = "euler_generator.keras"
NUM_IMAGES = 5000

# --- Geração dos Dados ---
def load_euler_data():
    """Gera imagens do gráfico da função exponencial y = e^x."""
    print(f"Gerando {NUM_IMAGES} imagens da função de Euler...")
    images = []
    
    for _ in range(NUM_IMAGES):
        img = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
        
        # Define o domínio de x para o gráfico
        x_min = -2.0 + np.random.uniform(-0.5, 0.5)
        x_max = 2.0 + np.random.uniform(-0.5, 0.5)
        x = np.linspace(x_min, x_max, IMAGE_SIZE)
        y = np.exp(x)
        
        # Normaliza y para caber na imagem
        y_scaled = (y - y.min()) / (y.max() - y.min() + 1e-8) * (IMAGE_SIZE - 1)
        
        for i in range(IMAGE_SIZE):
            px, py = i, int(round(y_scaled[i]))
            if 0 <= py < IMAGE_SIZE:
                # Desenha o ponto e um pouco de espessura
                for offset in range(-1, 2):
                    if 0 <= py + offset < IMAGE_SIZE:
                        img[IMAGE_SIZE - 1 - (py + offset), px] = 1.0 # Inverte o eixo y
                        
        images.append(img)
        
    images = np.array(images).reshape((-1, IMAGE_SIZE, IMAGE_SIZE, 1))
    images = (images * 2) - 1
    
    dataset = tf.data.Dataset.from_tensor_slices(images)
    dataset = dataset.shuffle(buffer_size=NUM_IMAGES).batch(BATCH_SIZE)
    return dataset

# --- Funções de Perda e Lógica Principal (idênticas) ---
def discriminator_loss(real_img, fake_img):
    return tf.reduce_mean(fake_img) - tf.reduce_mean(real_img)

def generator_loss(fake_img):
    return -tf.reduce_mean(fake_img)

if __name__ == '__main__':
    print(f"--- Iniciando Treinamento para: {MODEL_NAME} ---")
    generator = build_generator(LATENT_DIM)
    discriminator = build_discriminator((IMAGE_SIZE, IMAGE_SIZE, 1))
    wgan = WGAN(discriminator=discriminator, generator=generator, latent_dim=LATENT_DIM)
    d_optimizer = optimizers.Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.9)
    g_optimizer = optimizers.Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.9)
    wgan.compile(d_optimizer=d_optimizer, g_optimizer=g_optimizer, d_loss_fn=discriminator_loss, g_loss_fn=generator_loss)
    train_dataset = load_euler_data()
    print(f"\nIniciando treinamento por {EPOCHS} épocas...")
    wgan.fit(train_dataset, epochs=EPOCHS)
    generator.save(MODEL_NAME)
    print(f"\n✅ Modelo gerador da constante de Euler salvo em: {MODEL_NAME}")