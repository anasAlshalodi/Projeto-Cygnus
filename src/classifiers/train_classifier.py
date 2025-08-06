import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import numpy as np
import os 
N_SAMPLES_PER_CLASS = 2000
SIGNAL_LENGTH = 256
def add_noise(signal: np.ndarray, noise_type: str = 'gaussian', level: float = 0.1) -> np.ndarray:
    signal = signal.flatten()
    if noise_type == 'gaussian':
        noise = np.random.normal(0, level, signal.shape)
        return signal + noise
    elif noise_type == 'bitflip':
        num_flips = int(level * len(signal))
        flip_indices = np.random.choice(len(signal), num_flips, replace=False)
        noisy_signal = signal.copy()
        noisy_signal[flip_indices] *= -1
        return noisy_signal
    elif noise_type == 'fade':
        fade_effect = np.linspace(1.0, 1.0 - level, len(signal))
        return signal * fade_effect
    else:
        return signal

def generate_pulsar_signals(num_samples):
    signals = []
    for _ in range(num_samples):
        noise = np.random.randn(SIGNAL_LENGTH) * 0.5; pulse_position = SIGNAL_LENGTH // 4 + np.random.randint(-10, 10); pulse = np.exp(-((np.arange(SIGNAL_LENGTH) - pulse_position)**2) / 50); signals.append(noise + pulse * 2)
    return np.array(signals)
def generate_frb_signals(num_samples):
    signals = [];
    for _ in range(num_samples):
        noise = np.random.randn(SIGNAL_LENGTH) * 0.5; pulse_position = np.random.randint(20, SIGNAL_LENGTH - 20); pulse_width = np.random.uniform(1, 3); pulse_amplitude = np.random.uniform(3, 5); pulse = pulse_amplitude * np.exp(-((np.arange(SIGNAL_LENGTH) - pulse_position)**2) / (2 * pulse_width**2)); signals.append(noise + pulse)
    return np.array(signals)
def generate_solar_flare_signals(num_samples):
    signals = [];
    for _ in range(num_samples):
        noise = np.random.randn(SIGNAL_LENGTH) * 0.5; start_point = np.random.randint(0, SIGNAL_LENGTH // 2); decay_rate = np.random.uniform(0.05, 0.1); flare_amplitude = np.random.uniform(2, 4); time_points = np.arange(SIGNAL_LENGTH - start_point); flare = flare_amplitude * np.exp(-decay_rate * time_points); noise[start_point:] += flare; signals.append(noise)
    return np.array(signals)

# --- FUNÇÕES DE GERAÇÃO DE SINAIS ---
def generate_ai_signals(generator_filename, num_samples, target_length=256):
    try:
        ### CORREÇÃO: Caminho do modelo relativo à raiz do projeto ###
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        model_path = os.path.join(project_root, generator_filename)

        generator = load_model(model_path)
        noise = tf.random.normal([num_samples, 100])
        signals = generator.predict(noise, verbose=0)
        flat_signals = signals.reshape(num_samples, -1)
        processed_signals = np.zeros((num_samples, target_length))
        for i, signal in enumerate(flat_signals):
            current_len = len(signal)
            copy_len = min(current_len, target_length)
            processed_signals[i, :copy_len] = signal[:copy_len]
        return processed_signals
    except Exception as e:
        print(f"Erro ao carregar ou usar o gerador em {model_path}: {e}")
        return None

# --- CONSTRUÇÃO DO CLASSIFICADOR (sem alterações) ---
def build_universal_classifier(num_classes):
    model = Sequential(name="Classificador_Universal_v3_Robusto"); model.add(Input(shape=(SIGNAL_LENGTH, 1))); model.add(Conv1D(filters=32, kernel_size=5, activation='relu')); model.add(MaxPooling1D(pool_size=2)); model.add(Dropout(0.3)); model.add(Conv1D(filters=64, kernel_size=5, activation='relu')); model.add(MaxPooling1D(pool_size=2)); model.add(Dropout(0.3)); model.add(Flatten()); model.add(Dense(128, activation='relu')); model.add(Dense(num_classes, activation='softmax')); model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']); return model

# --- EXECUÇÃO PRINCIPAL (COM DATA AUGMENTATION) ---
print("Gerando datasets da biblioteca expandida...")
class_generators = {
    'Pulsar': lambda n: generate_pulsar_signals(n),
    'FRB': lambda n: generate_frb_signals(n),
    'Solar Flare': lambda n: generate_solar_flare_signals(n),
    'Primes': lambda n: generate_ai_signals('generator_model.keras', n),
    'Fibonacci': lambda n: generate_ai_signals('generator_fibonacci_model.keras', n),
    'Golden Ratio': lambda n: generate_ai_signals('generator_golden_ratio_model.keras', n),
    'Euler': lambda n: generate_ai_signals('generator_euler_model.keras', n)
}
# ... (resto da lógica de geração de dados e treinamento sem alterações)
all_signals = []
all_labels = []

print("Gerando sinais e aplicando aumento de dados com ruído...")
for i, (name, gen_func) in enumerate(class_generators.items()):
    print(f"- Gerando classe: {name}")
    clean_signals = gen_func(N_SAMPLES_PER_CLASS)
    if clean_signals is None:
        print(f"ERRO: Falha ao gerar sinais para {name}. Verifique os modelos .keras.")
        exit()
    noisy_signals = []
    for signal in clean_signals:
        noise_type = np.random.choice(['gaussian', 'bitflip', 'fade'])
        level = np.random.uniform(0.05, 0.2)
        noisy_signal = add_noise(signal, noise_type, level)
        noisy_signals.append(noisy_signal)
    all_signals.extend(clean_signals)
    all_signals.extend(noisy_signals)
    all_labels.extend([i] * (N_SAMPLES_PER_CLASS * 2))

X = np.array(all_signals)
y = to_categorical(all_labels)
X = X.reshape(X.shape[0], X.shape[1], 1)
print(f"\nDataset final criado com {X.shape[0]} amostras totais ({N_SAMPLES_PER_CLASS} limpas + {N_SAMPLES_PER_CLASS} com ruído por classe).")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print("Datasets prontos para o treinamento.")

num_classes = len(class_generators)
classifier = build_universal_classifier(num_classes)
print("\n--- Treinando o Classificador Universal v3 (Robusto) ---")
classifier.summary()
history = classifier.fit(X_train, y_train, epochs=25, batch_size=64, validation_split=0.1)

loss, accuracy = classifier.evaluate(X_test, y_test)
print(f"\nAcurácia do Classificador Robusto: {accuracy:.2%}")

### CORREÇÃO: Salvar o modelo na pasta raiz do projeto ###
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
model_save_path = os.path.join(project_root, 'universal_classifier_model_v3_robust.keras')
classifier.save(model_save_path)
print(f"Modelo Robusto v3 salvo em: {model_save_path}")

### CORREÇÃO: Adicionar a função generate_pi_signal que estava faltando ###
def generate_pi_signal(signal_length=256):
    pi_digits_str = "1415926535897932384626433832795028841971693993751058209749445923078164062862089986280348253421170679821480865132823066470938446095505822317253594081284811174502841027019385211055596446229489549303819644288109756659334461284756482337867831652712019091"
    pi_digits = np.array([int(d) for d in pi_digits_str[:signal_length]])
    if len(pi_digits) < signal_length:
        padding = np.zeros(signal_length - len(pi_digits))
        pi_digits = np.concatenate([pi_digits, padding])
    signal = (pi_digits / 4.5) - 1.0
    return signal.reshape(1, signal_length, 1)

# --- Teste Final com Pi ---
print("\n--- TESTE UNIVERSAL FINAL COM PI (π) ---")
pi_signal = generate_pi_signal()
prediction_probs = classifier.predict(pi_signal)[0]
class_names = ['Natural (Pulsar)', 'Natural (FRB)', 'Natural (Solar Flare)', 'Artificial (Primos)', 'Artificial (Fibonacci)', 'Artificial (Prop. Áurea)', 'Artificial (Euler)']
veredito_index = np.argmax(prediction_probs)
veredito_nome = class_names[veredito_index]
confianca = np.max(prediction_probs) * 100

print("O classificador analisou o sinal de Pi e determinou as seguintes probabilidades:")
for name, prob in zip(class_names, prediction_probs):
    print(f"- {name}: {prob:.2%}")

print(f"\nVeredito Final: O sinal de Pi se parece mais com um sinal do tipo '{veredito_nome}'.")
print(f"Confiança: {confianca:.2f}%")