import numpy as np

# --- Parâmetros baseados nos dados reais de GLEAM-X ---
PULSE_DURATION_SECONDS = 60
PERIOD_SECONDS = 18.18 * 60  # 18.18 minutos em segundos
SAMPLES_PER_SECOND = 100 # Resolução do nosso sinal simulado
TOTAL_DURATION_SECONDS = 4 * PERIOD_SECONDS # Simular 4 ciclos completos
OUTPUT_FILE = "sinal_gleamx.txt"

def generate_gleamx_signal():
    """
    Gera um sinal binário simulado com as propriedades de tempo do
    GLEAM-X J162759.5-523504.3.
    """
    total_samples = int(TOTAL_DURATION_SECONDS * SAMPLES_PER_SECOND)
    pulse_samples = int(PULSE_DURATION_SECONDS * SAMPLES_PER_SECOND)
    period_samples = int(PERIOD_SECONDS * SAMPLES_PER_SECOND)
    
    print(f"Gerando sinal simulado de GLEAM-X com {total_samples} bits...")
    signal = np.zeros(total_samples, dtype=int)
    
    # Cria os pulsos nos intervalos corretos
    for i in range(4): # Criar 4 pulsos
        start_index = i * period_samples
        end_index = start_index + pulse_samples
        if end_index < total_samples:
            signal[start_index:end_index] = 1
            
    # Adiciona um pouco de ruído aleatório (ex: 0.1% de chance de um bit inverter)
    noise_mask = np.random.rand(total_samples) < 0.001
    signal[noise_mask] = 1 - signal[noise_mask]
    
    return signal

if __name__ == "__main__":
    print("🚀 Gerando sinal simulado do 'Farol Cósmico' GLEAM-X...")
    gleamx_signal = generate_gleamx_signal()
    
    try:
        with open(OUTPUT_FILE, "w") as f:
            f.write("".join(map(str, gleamx_signal)))
        print(f"✅ Sinal simulado de GLEAM-X com {len(gleamx_signal)} bits salvo com sucesso em: {OUTPUT_FILE}")
    except IOError as e:
        print(f"❌ Erro ao salvar o arquivo: {e}")