import numpy as np

# --- Parâmetros ---
SIGNAL_LENGTH = 256
OUTPUT_FILE = "sinal_pi.txt"

def generate_pi_signal(length=256):
    """
    Gera um sinal baseado nos dígitos de Pi.
    A estrutura é '1' seguido por N '0's, onde N é o dígito de Pi.
    Ex: 3, 1, 4 -> 1000 10 10000
    """
    pi_digits = [int(d) for d in "314159265358979323846264338327950288419716939937510"]
    signal = []
    while len(signal) < length:
        for digit in pi_digits:
            if len(signal) >= length:
                break
            signal.append(1)
            # Adiciona um número de zeros igual ao dígito. Se o dígito for 0, não adiciona zeros.
            if digit > 0:
                zeros_to_add = min(digit, length - len(signal))
                signal.extend([0] * zeros_to_add)
    
    return np.array(signal[:length], dtype=int)

if __name__ == "__main__":
    print(f"🚀 Gerando sinal universal de teste baseado em Pi (π)...")
    
    pi_signal = generate_pi_signal(SIGNAL_LENGTH)
    
    # Salva o sinal em um arquivo de texto
    try:
        with open(OUTPUT_FILE, "w") as f:
            f.write("".join(map(str, pi_signal)))
        print(f"✅ Sinal de Pi com {len(pi_signal)} bits salvo com sucesso em: {OUTPUT_FILE}")
        print("\nConteúdo do sinal:")
        print("".join(map(str, pi_signal)))
    except IOError as e:
        print(f"❌ Erro ao salvar o arquivo: {e}")