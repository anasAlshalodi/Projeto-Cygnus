import numpy as np
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# --- Parâmetros ---
GENERATOR_PATHS = {
    "primes": "prime_generator.keras",
    "fibonacci": "fibonacci_generator.keras",
    "constants": "euler_generator.keras" # <-- CORRIGIDO
    }
LATENT_DIM = 128
NUM_IMAGES_PER_CONCEPT = 5
IMAGE_SIZE = 32
FINAL_MESSAGE_FILE = "cygnus_message_final.txt"
VISUALIZATION_FILE = "cygnus_message_visualization.png"


# --- Funções do Pipeline ---

def generate_dreams(generator_path, n_images):
    """Carrega um gerador e gera 'sonhos' (imagens)."""
    try:
        generator = load_model(generator_path)
        noise = np.random.randn(n_images, LATENT_DIM)
        dreams = generator.predict(noise)
        # Desnormalizar de [-1, 1] para [0, 255]
        dreams = (dreams * 127.5 + 127.5).astype(np.uint8)
        return dreams
    except IOError:
        print(f"Erro: Modelo gerador não encontrado em '{generator_path}'. Execute os scripts de treinamento primeiro.")
        return None

def image_to_binary(image, threshold=128):
    """Converte uma única imagem em uma string binária."""
    binary_image = (image > threshold).astype(int)
    return "".join(map(str, binary_image.flatten()))

def build_rosetta_stone(image_size):
    """Cria um dicionário visual (Pedra de Roseta)."""
    header = "ROSETTA_STONE_BEGIN\n"
    # Padrão 1: Um quadrado para definir as dimensões
    square = np.zeros((image_size, image_size))
    square[4:-4, 4:-4] = 1 # Quadrado no centro
    square_binary = "".join(map(str, square.flatten().astype(int)))
    
    # Padrão 2: Um padrão de "zero" e "um" visual
    zero_pattern = np.zeros((image_size, image_size)) # Um frame vazio
    one_pattern = np.ones((image_size, image_size)) # Um frame cheio
    zero_binary = "".join(map(str, zero_pattern.flatten().astype(int)))
    one_binary = "".join(map(str, one_pattern.flatten().astype(int)))

    rosetta = (f"{header}"
               f"IMG_DIM:{square_binary}\n"
               f"SYMBOL_0:{zero_binary}\n"
               f"SYMBOL_1:{one_binary}\n"
               "ROSETTA_STONE_END\n")
    return rosetta

def apply_error_correction(binary_string):
    """Placeholder para um código de correção de erros (ECC)."""
    # Exemplo simples: repetir cada bit 3 vezes (código de repetição)
    # Em um projeto real, usaríamos algo mais robusto como Reed-Solomon.
    return "".join([bit*3 for bit in binary_string])

# --- Orquestração ---

if __name__ == "__main__":
    print("🚀 Iniciando a construção da Mensagem Cygnus Unificada...")
    
    final_message_content = []
    all_images_for_visualization = []

    # 1. Construir a Pedra de Roseta
    print("   - Construindo a Pedra de Roseta...")
    rosetta = build_rosetta_stone(IMAGE_SIZE)
    final_message_content.append(rosetta)

    # 2. Gerar e codificar imagens para cada conceito
    for concept, path in GENERATOR_PATHS.items():
        print(f"   - Gerando sonhos para o conceito: '{concept}'...")
        images = generate_dreams(path, NUM_IMAGES_PER_CONCEPT)
        if images is not None:
            all_images_for_visualization.extend(images)
            concept_header = f"CONCEPT_{concept.upper()}_BEGIN\n"
            final_message_content.append(concept_header)
            
            for i, img in enumerate(images):
                binary_str = image_to_binary(img.reshape(IMAGE_SIZE, IMAGE_SIZE))
                # Aqui você poderia aplicar ECC se quisesse
                # binary_str_ecc = apply_error_correction(binary_str)
                final_message_content.append(f"IMG_{i}:{binary_str}\n")
            
            concept_footer = f"CONCEPT_{concept.upper()}_END\n"
            final_message_content.append(concept_footer)

    # 3. Salvar a mensagem final em arquivo
    if len(final_message_content) > 1:
        with open(FINAL_MESSAGE_FILE, "w") as f:
            f.write("".join(final_message_content))
        print(f"\n✅ Mensagem final salva em: {FINAL_MESSAGE_FILE}")

        # 4. Criar e salvar uma visualização da mensagem
        num_total_images = len(all_images_for_visualization)
        if num_total_images > 0:
            cols = NUM_IMAGES_PER_CONCEPT
            rows = int(np.ceil(num_total_images / cols))
            fig, axes = plt.subplots(rows, cols, figsize=(cols*2, rows*2))
            axes = axes.flatten()
            for i, img in enumerate(all_images_for_visualization):
                axes[i].imshow(img.reshape(IMAGE_SIZE, IMAGE_SIZE), cmap='gray')
                axes[i].axis('off')
            for j in range(i + 1, len(axes)):
                axes[j].axis('off')
            plt.suptitle("Visualização da Mensagem Cygnus")
            plt.tight_layout()
            plt.savefig(VISUALIZATION_FILE)
            print(f"✅ Visualização da mensagem salva em: {VISUALIZATION_FILE}")
    else:
        print("\n❌ Nenhuma mensagem foi gerada. Verifique se os modelos geradores existem.")