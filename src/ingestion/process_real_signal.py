import blimpy as bl
import numpy as np
import argparse

def process_filterbank_file(filepath, output_path):
    """
    Lê um arquivo filterbank, extrai a série temporal e a binariza,
    salvando o resultado em pedaços para economizar memória.
    """
    print(f"📡 Carregando arquivo de dados reais: {filepath}")
    try:
        wf = bl.Waterfall(filepath)
    except Exception as e:
        print(f"❌ Erro ao ler o arquivo: {e}")
        return

    print("   - Extraindo série temporal (simplificado)...")
    time_series = wf.data.mean(axis=2).mean(axis=1)

    print("   - Binarizando a série temporal...")
    threshold = time_series.mean() + time_series.std()
    binary_signal = (time_series > threshold).astype(np.uint8) # Usar uint8 economiza memória

    # --- INÍCIO DA MUDANÇA ---
    # Em vez de criar uma string gigante, escrevemos em pedaços (chunks)
    
    chunk_size = 65536 # Escrever 64k bits de cada vez (um bom tamanho)
    
    print(f"   - Salvando sinal binário em pedaços de {chunk_size} bits...")
    try:
        with open(output_path, "w") as f:
            for i in range(0, len(binary_signal), chunk_size):
                chunk = binary_signal[i:i + chunk_size]
                f.write("".join(map(str, chunk)))
        
        print(f"✅ Sinal binário com {len(binary_signal)} bits salvo em: {output_path}")
    
    except IOError as e:
        print(f"❌ Erro ao salvar o arquivo de saída: {e}")
    # --- FIM DA MUDANÇA ---


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Processa um arquivo de dados de radiotelescópio (.fil).")
    parser.add_argument("input_file", type=str, help="Caminho para o arquivo .fil de entrada.")
    parser.add_argument("output_file", type=str, help="Caminho para o arquivo .txt de saída com o sinal binário.")
    
    args = parser.parse_args()
    
    process_filterbank_file(args.input_file, args.output_file)