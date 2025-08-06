import numpy as np
import argparse
from sklearn.cluster import KMeans
from gplearn.genetic import SymbolicRegressor
from gplearn.functions import make_function

# --- Funções Utilitárias e de IA Especializadas ---

def _protected_exp(x):
    with np.errstate(over='ignore'):
        return np.exp(x)
exp_func = make_function(function=_protected_exp, name='exp', arity=1)

def decode_signal_to_sequence(signal_path, sample_size, word_length=4, k_clusters=16):
    """Carrega um sinal, executa K-Means e retorna a sequência de números decodificada."""
    print(f"--- FASE 1: Decodificando o Sinal de '{signal_path}' ---")
    try:
        with open(signal_path, 'r') as f:
            signal_str = f.read(sample_size) if sample_size > 0 else f.read()
        signal = np.array([int(bit) for bit in signal_str.strip()])
    except Exception as e:
        print(f"❌ Erro ao carregar o arquivo: {e}")
        return None

    print(f"   - Amostra de {len(signal)} bits carregada.")
    
    trimmed_length = len(signal) - (len(signal) % word_length)
    if trimmed_length == 0:
        print("❌ Erro: Amostra muito pequena para formar uma 'palavra'.")
        return None
    windows = signal[:trimmed_length].reshape(-1, word_length)
    
    print(f"   - Executando K-Means com k={k_clusters} para encontrar o alfabeto...")
    kmeans = KMeans(n_clusters=k_clusters, random_state=42, n_init='auto').fit(windows)
    
    translated_signal = kmeans.predict(windows)
    def to_int(word): return int("".join(map(str, word)), 2)
    translation_map = {i: to_int(word) for i, word in enumerate(np.round(kmeans.cluster_centers_).astype(int))}
    semantic_signal = [translation_map.get(word_id, -1) for word_id in translated_signal]
    
    print("✅ Sequência de números decodificada com sucesso.")
    return np.array(semantic_signal)

def find_trend(n, sequence):
    """IA Especialista em Tendências (TrendHunter)."""
    print("\n--- FASE 2.1: Ativando o Caçador de Tendências (TrendHunter) ---")
    trend_brain = ('add', 'sub', 'mul', 'div', 'log', exp_func)
    est_trend = SymbolicRegressor(population_size=4000, generations=20,
                                  stopping_criteria=1e-5, verbose=1,
                                  feature_names=['n'], function_set=trend_brain,
                                  const_range=(-1., 1.), random_state=42)
    est_trend.fit(n, sequence)
    print(f"📈 Hipótese de Tendência encontrada: {est_trend._program}")
    return est_trend

def find_oscillation(n, detrended_sequence):
    """IA Especialista em Oscilações (RhythmFinder)."""
    print("\n--- FASE 2.2: Ativando o Localizador de Ritmos (RhythmFinder) ---")
    rhythm_brain = ('add', 'sub', 'mul', 'sin', 'cos')
    est_rhythm = SymbolicRegressor(population_size=4000, generations=20,
                                   stopping_criteria=1e-5, verbose=1,
                                   feature_names=['n'], function_set=rhythm_brain,
                                   const_range=(-1., 1.), random_state=42)
    est_rhythm.fit(n, detrended_sequence)
    print(f"🌊 Hipótese de Oscilação encontrada: {est_rhythm._program}")
    return est_rhythm

def run_architect_analysis(sequence):
    """O 'Arquiteto' que gerencia as IAs especialistas."""
    print("\n" + "="*50)
    print("🏛️ INICIANDO ARQUITETURA 'ARCHITECT' DO CYGNUS 🏛️")
    print("="*50)
    
    n = np.arange(len(sequence)).reshape(-1, 1)

    # 1. Convoca o TrendHunter
    trend_model = find_trend(n, sequence)
    trend_prediction = trend_model.predict(n)

    # 2. Processa e limpa os dados para o RhythmFinder
    print("\n   - Processando e limpando a previsão de tendência...")
    trend_prediction = np.nan_to_num(trend_prediction, nan=1.0, posinf=1.0, neginf=1.0)
    trend_prediction[np.abs(trend_prediction) < 1e-6] = 1e-6
    detrended_sequence = sequence / trend_prediction
    detrended_sequence = np.nan_to_num(detrended_sequence, nan=0.0, posinf=0.0, neginf=0.0)
    
    # 3. Convoca o RhythmFinder
    oscillation_model = find_oscillation(n, detrended_sequence)

    # 4. Sintetiza os resultados
    print("\n" + "="*50)
    print("🏆 SÍNTESE FINAL DO ARQUITETO 🏆")
    print("="*50)
    print(f"📈 Tendência Descoberta (T): {trend_model._program}")
    print(f"🌊 Oscilação Descoberta (O): {oscillation_model._program}")
    
    final_formula_str = f"mul({trend_model._program}, {oscillation_model._program})"
    print(f"\n🧩 Hipótese da Fórmula Combinada (T * O): {final_formula_str}")
    
    # 5. Calcula o erro final
    final_prediction = trend_prediction * oscillation_model.predict(n)
    final_error = np.mean(np.abs(sequence - final_prediction))
    print(f"\n📊 Erro (Fitness) da Fórmula Combinada Final: {final_error:.6f}")
    
    if final_error < 0.1: # Usamos um critério de sucesso mais flexível
        print("\n✅ Veredito: SUCESSO! A equipe de IAs especialistas desvendou a regra oculta.")
    else:
        print("\n❌ Veredito: FALHA INFORMATIVA. A lógica do sinal não é uma simples combinação de Tendência x Oscilação.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Executa a análise final do Cygnus em um sinal.")
    parser.add_argument("signal_file", type=str, help="Caminho para o arquivo de sinal.")
    parser.add_argument("--sample_size", type=int, default=100000, help="Analisar os N primeiros bits.")
    
    args = parser.parse_args()
    
    # Decodifica o sinal real para obter a sequência de números
    decoded_sequence = decode_signal_to_sequence(args.signal_file, args.sample_size)
    
    # Se a decodificação foi bem-sucedida, executa a análise do Arquiteto
    if decoded_sequence is not None:
        run_architect_analysis(decoded_sequence)