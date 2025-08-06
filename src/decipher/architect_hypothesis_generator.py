import numpy as np
from gplearn.genetic import SymbolicRegressor
from gplearn.functions import make_function

# --- Funções de IA Especializadas ---

def _protected_exp(x):
    with np.errstate(over='ignore'):
        return np.exp(x)
exp_func = make_function(function=_protected_exp, name='exp', arity=1)

def find_trend(n, sequence, generations=20):
    """
    IA Especialista em Tendências (TrendHunter).
    Usa um cérebro matemático focado em crescimento/decaimento.
    """
    print("\n--- 🧠 Ativando o Caçador de Tendências (TrendHunter) ---")
    trend_brain = ('add', 'sub', 'mul', 'div', 'log', exp_func)
    
    est_trend = SymbolicRegressor(population_size=4000, generations=generations,
                                  stopping_criteria=1e-5, verbose=1,
                                  feature_names=['n'], function_set=trend_brain,
                                  const_range=(-1., 1.), random_state=42)
    
    est_trend.fit(n, sequence)
    print(f"📈 Hipótese de Tendência encontrada: {est_trend._program}")
    return est_trend

def find_oscillation(n, detrended_sequence, generations=20):
    """
    IA Especialista em Oscilações (RhythmFinder).
    Usa um cérebro matemático focado em periodicidade.
    """
    print("\n--- 🎶 Ativando o Localizador de Ritmos (RhythmFinder) ---")
    rhythm_brain = ('add', 'sub', 'mul', 'sin', 'cos')

    est_rhythm = SymbolicRegressor(population_size=4000, generations=generations,
                                   stopping_criteria=1e-5, verbose=1,
                                   feature_names=['n'], function_set=rhythm_brain,
                                   const_range=(-1., 1.), random_state=42)

    est_rhythm.fit(n, detrended_sequence)
    print(f"🌊 Hipótese de Oscilação encontrada: {est_rhythm._program}")
    return est_rhythm


# --- O Arquiteto (Gerenciador Principal) ---

def run_architect_analysis(sequence):
    """
    O 'Arquiteto' que gerencia as IAs especialistas para encontrar a fórmula final.
    """
    print("\n" + "="*50)
    print("🏛️ INICIANDO ARQUITETURA 'ARCHITECT' DO CYGNUS 🏛️")
    print("="*50)
    
    n = np.arange(len(sequence)).reshape(-1, 1)

    # 1. O Arquiteto convoca o TrendHunter
    trend_model = find_trend(n, sequence)
    trend_prediction = trend_model.predict(n)

    # 2. O Arquiteto processa o trabalho do TrendHunter para isolar a oscilação
    print("   - Processando e limpando a previsão de tendência...")

    # --- INÍCIO DA MUDANÇA (FILTRO DE SANIDADE) ---
    # Primeiro, substituímos quaisquer valores inválidos (NaN, inf) na previsão por 1.0
    # para que não afetem a divisão.
    trend_prediction = np.nan_to_num(trend_prediction, nan=1.0, posinf=1.0, neginf=1.0)

    # Em seguida, evitamos a divisão por zero, como antes.
    trend_prediction[np.abs(trend_prediction) < 1e-6] = 1e-6

    detrended_sequence = sequence / trend_prediction

    # Filtro de Sanidade Final: garantimos que o resultado final também não tenha inválidos.
    detrended_sequence = np.nan_to_num(detrended_sequence, nan=0.0, posinf=0.0, neginf=0.0)
        
    # 3. O Arquiteto convoca o RhythmFinder com a tarefa já simplificada
    oscillation_model = find_oscillation(n, detrended_sequence)

    # 4. O Arquiteto sintetiza os resultados
    print("\n" + "="*50)
    print("🏆 SÍNTESE FINAL DO ARQUITETO 🏆")
    print("="*50)
    print(f"📈 Tendência Descoberta (T): {trend_model._program}")
    print(f"🌊 Oscilação Descoberta (O): {oscillation_model._program}")
    
    # Tentamos construir a fórmula final
    final_formula_str = f"mul({trend_model._program}, {oscillation_model._program})"
    print(f"\n🧩 Hipótese da Fórmula Combinada (T * O): {final_formula_str}")
    
    # 5. Verificação Final: O Arquiteto calcula o erro da solução combinada
    final_prediction = trend_prediction * oscillation_model.predict(n)
    final_error = np.mean(np.abs(sequence - final_prediction))
    print(f"\n📊 Erro (Fitness) da Fórmula Combinada Final: {final_error:.6f}")
    if final_error < 0.001:
        print("\n✅ Veredito: SUCESSO! A equipe de IAs especialistas desvendou a regra oculta.")
    else:
        print("\n❌ Veredito: FALHA. A equipe não conseguiu convergir para uma solução precisa.")


if __name__ == "__main__":
    # O mesmo teste de estresse que falhou antes.
    print("--- Gerando sinal de teste com a regra S(n) = exp(-0.1n) * sin(0.5n) ---")
    test_rule = lambda n: np.exp(-0.1 * n) * np.sin(0.5 * n)
    test_sequence = np.array([test_rule(i) for i in range(100)])
    
    run_architect_analysis(test_sequence)