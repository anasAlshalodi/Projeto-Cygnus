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
MODEL_NAME = "fibonacci_generator.keras"
NUM_IMAGES = 5000

# --- Geração dos Dados ---
def load_fibonacci_data():
    """Gera imagens da espiral de Fibonacci."""
    print(f"Gerando {NUM_IMAGES} imagens da espiral de Fibonacci...")
    images = []
    fib_nums = [0, 1, 1, 2, 3, 5, 8, 13, 21] 

    for _ in range(NUM_IMAGES):
        img = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
        x, y = IMAGE_SIZE // 2, IMAGE_SIZE // 2
        
        # Introduz uma pequena variação no ponto inicial
        x += np.random.randint(-2, 3)
        y += np.random.randint(-2, 3)
        
        # 0: direita, 1: cima, 2: esquerda, 3: baixo
        direction = 0
        
        # Desenha os quadrados da espiral
        for i in range(1, np.random.randint(5, len(fib_nums))):
            f = fib_nums[i]
            if direction == 0:
                nx, ny = x + f, y - f
            elif direction == 1:
                nx, ny = x, y - f
            elif direction == 2:
                nx, ny = x - f, y
            else: # direction == 3
                nx, ny = x, y + f
            
            # Garante que os quadrados fiquem dentro da imagem
            x1, y1 = max(0, min(x, nx)), max(0, min(y, ny))
            x2, y2 = min(IMAGE_SIZE, max(x, nx)), min(IMAGE_SIZE, max(y, ny))
            
            if x1 < x2 and y1 < y2:
                img[y1:y2, x1:x2] = 1.0

            x, y = nx, ny
            direction = (direction + 1) % 4
            
        images.append(img)
        
    images = np.array(images).reshape((-1, IMAGE_SIZE, IMAGE_SIZE, 1))
    images = (images * 2) - 1
    
    dataset = tf.data.Dataset.from_tensor_slices(images)
    dataset = dataset.shuffle(buffer_size=NUM_IMAGES).batch(BATCH_SIZE)
    return dataset

# --- Funções de Perda e Lógica Principal (idênticas ao script anterior) ---
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
    train_dataset = load_fibonacci_data()
    print(f"\nIniciando treinamento por {EPOCHS} épocas...")
    wgan.fit(train_dataset, epochs=EPOCHS)
    generator.save(MODEL_NAME)
    print(f"\n✅ Modelo gerador de Fibonacci salvo em: {MODEL_NAME}")