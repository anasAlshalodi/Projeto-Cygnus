import numpy as np
from scipy.stats import entropy
from lempel_ziv_complexity import lempel_ziv_complexity

# --- Funções de Geração de Sinais de Teste ---
#    Não precisamos de dados reais ainda. Vamos criar nossos próprios
#    sinais para garantir que nossas funções de análise funcionam.

def generate_random_signal(length=1000):
    """Gera um sinal de ruído branco, puramente aleatório."""
    return np.random.randint(0, 2, length)

def generate_simple_repeating_signal(length=1000, period=10):
    """Gera um sinal altamente previsível, como um pulsar simples."""
    pattern = np.random.randint(0, 2, period)
    repetitions = length // period
    signal = np.tile(pattern, repetitions)
    # Garante que o comprimento final seja exatamente 'length'
    return signal[:length]

def generate_structured_signal(length=1000):
    """
    Gera um sinal com estrutura, como uma "linguagem" simples.
    Ele tem padrões, mas não é perfeitamente repetitivo.
    """
    # Define "palavras" em nosso mini-idioma
    word_a = [1, 0, 1, 0]
    word_b = [0, 0, 1, 1]
    separator = [0, 0, 0]
    
    # Cria "frases"
    phrase_1 = word_a + separator + word_b
    phrase_2 = word_b + separator + word_a + separator + word_a
    
    # Constrói o sinal repetindo as frases com alguma variação
    base = phrase_1 + phrase_2
    repetitions = length // len(base)
    signal = np.tile(base, repetitions)
    
    # Adiciona um pouco de ruído para não ser perfeito (ex: 1% de chance de bit-flip)
    noise_mask = np.random.rand(len(signal)) < 0.01
    signal[noise_mask] = 1 - signal[noise_mask]
    
    return signal[:length]

# --- Funções de Análise do Sinal ---

def calculate_shannon_entropy(signal):
    """
    Calcula a Entropia de Shannon do sinal.
    - Próximo de 1.0 para sinais aleatórios (em bits).
    - Próximo de 0.0 para sinais constantes ou muito simples.
    - Valores intermediários sugerem estrutura.
    """
    # Precisamos contar a frequência dos valores (0s e 1s)
    _, counts = np.unique(signal, return_counts=True)
    return entropy(counts, base=2)

def calculate_lz_complexity(signal):
    """
    Calcula a Complexidade Lempel-Ziv.
    Mede o número de padrões únicos em uma sequência.
    Valores mais altos indicam maior complexidade e mais padrões.
    """
    # A biblioteca espera uma string de bytes ou uma string normal
    signal_str = "".join(map(str, signal))
    return lempel_ziv_complexity(signal_str)

# --- Bloco Principal de Execução ---

if __name__ == "__main__":
    print("🚀 Iniciando o Analisador de Complexidade do Cygnus-Decipher 🚀")
    print("-" * 60)

    # 1. Gerar os sinais
    random_sig = generate_random_signal()
    simple_sig = generate_simple_repeating_signal()
    structured_sig = generate_structured_signal()

    # 2. Analisar cada sinal
    signals_to_analyze = {
        "Sinal Aleatório (Ruído)": random_sig,
        "Sinal Simples e Repetitivo (Pulsar Simulado)": simple_sig,
        "Sinal Estruturado (Linguagem Simulada)": structured_sig
    }

    for name, signal in signals_to_analyze.items():
        print(f"\nAnalisando: {name}")
        
        # Calcular métricas
        ent = calculate_shannon_entropy(signal)
        lz = calculate_lz_complexity(signal)
        
        # Normalizar a complexidade LZ para uma melhor comparação (dividindo pelo comprimento)
        lz_normalized = lz / len(signal)

        print(f"  -> Entropia de Shannon: {ent:.4f}")
        print(f"  -> Complexidade LZ Normalizada: {lz_normalized:.4f}")

    print("\n" + "-" * 60)
    print("📜 Interpretação dos Resultados:")
    print(" - O Sinal Aleatório deve ter a MAIOR entropia (próximo de 1.0).")
    print(" - O Sinal Simples deve ter a MENOR entropia e a MENOR complexidade LZ.")
    print(" - O Sinal Estruturado deve ter valores INTERMEDIÁRIOS, mostrando que não é aleatório, mas também não é trivial.")