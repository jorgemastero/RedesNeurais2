from tensorflow import keras
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
import pandas as pd
import keras_tuner as kt
import numpy as np
import tensorflow as tf
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import Callback
import os
import pickle
import matplotlib.pyplot as plt
import datetime

# === Configurações Globais ===
NORMALIZATION_FACTOR = 100 * 100
os.makedirs("my_logs", exist_ok=True)
os.makedirs("tuner_dir", exist_ok=True)

# === Classes Auxiliares ===
class TrainingHistory:
    def __init__(self):
        self.history = []
        self.columns = [
            'Tentativa', 'Trial', 'Execução', 
            'Val_MAE', 'Test_MAE', 
            'Val_Loss', 'Test_Loss',
            'Units1', 'Units2', 'Units3',
            'Dropout', 'Activation', 'Regularizer',
            'Optimizer', 'Learning_Rate'
        ]
    
    def add_result(self, tentativa, trial_num, exec_num, val_mae, test_mae, val_loss, test_loss, hps):
        self.history.append({
            'Tentativa': tentativa,
            'Trial': trial_num,
            'Execução': exec_num,
            'Val_MAE': val_mae,
            'Test_MAE': test_mae,
            'Val_Loss': val_loss,
            'Test_Loss': test_loss,
            'Units1': hps.get('units1', None),
            'Units2': hps.get('units2', None),
            'Units3': hps.get('units3', None),
            'Dropout': hps.get('dropout', None),
            'Activation': hps.get('activation', None),
            'Regularizer': hps.get('regularizer', None),
            'Optimizer': hps.get('optimizer', None),
            'Learning_Rate': hps.get('learning_rate', None)
        })
    
    def get_dataframe(self):
        return pd.DataFrame(self.history, columns=self.columns)

class MonitorMAECallback(Callback):
    def __deepcopy__(self, memo):
        return MonitorMAECallback()

    def on_epoch_end(self, epoch, logs=None):
        mae = logs.get('mae')
        val_mae = logs.get('val_mae')
        print(f" Epoch {epoch + 1}: MAE = {mae:.6f}, Val MAE = {val_mae:.6f}")

# === Funções de Pré-processamento ===
def normalizar_saida(y):
    return y / NORMALIZATION_FACTOR

def desnormalizar_saida(y_normalizado):
    return y_normalizado * NORMALIZATION_FACTOR

def train_datas(numbers):
    base_per_op = numbers // 4
    remainder = numbers % 4
    op_counts = [base_per_op + (1 if i < remainder else 0) for i in range(4)]
    X, y = [], []

    for op, count in enumerate(op_counts):
        for _ in range(count):
            number1 = np.random.uniform(-100, 100)
            number2 = np.random.uniform(-100, 100)
            
            if op == 3:
                while number2 == 0:
                    number2 = np.random.uniform(-100, 100)
                result = number1 / number2
            elif op == 0:
                result = number1 + number2
            elif op == 1:
                result = number1 - number2
            elif op == 2:
                result = number1 * number2

            number1 = number1 / 100
            number2 = number2 / 100
            result = normalizar_saida(result)

            operation_one_hot = [0, 0, 0, 0]
            operation_one_hot[op] = 1 

            X.append([number1, number2] + operation_one_hot)
            y.append(result)

    return np.array(X), np.array(y)

# === Funções de Modelagem ===
def build_model(hp):
    model = keras.Sequential()
    model.add(keras.Input(shape=(6,)))

    activation_choice = hp.Choice('activation', ['relu', 'tanh', 'silu'])
    hp_units1 = hp.Int('units1', 128, 1024, step=128)
    hp_units2 = hp.Int('units2', 64, 512, step=64)
    hp_units3 = hp.Int('units3', 32, 256, step=32)
    hp_units4 = hp.Int('units4', 16, 128, step=16)
    hp_dropout = hp.Float('dropout', 0.2, 0.4, step=0.1)
    hp_regularizer = hp.Choice('regularizer', ['none', 'l1', 'l2', 'l1_l2'])
    hp_reg_factor = hp.Float('reg_factor', 1e-4, 1e-2, sampling='log')

    if hp_regularizer == 'l1':
        kernel_regularizer = regularizers.l1(hp_reg_factor)
    elif hp_regularizer == 'l2':
        kernel_regularizer = regularizers.l2(hp_reg_factor)
    elif hp_regularizer == 'l1_l2':
        kernel_regularizer = regularizers.l1_l2(l1=hp_reg_factor, l2=hp_reg_factor)
    else:
        kernel_regularizer = None

    model.add(keras.layers.Dense(hp_units1, activation=activation_choice, kernel_regularizer=kernel_regularizer))
    model.add(keras.layers.Dropout(hp_dropout))
    model.add(keras.layers.Dense(hp_units2, activation=activation_choice, kernel_regularizer=kernel_regularizer))
    model.add(keras.layers.Dropout(hp_dropout))
    model.add(keras.layers.Dense(hp_units3, activation=activation_choice, kernel_regularizer=kernel_regularizer))
    model.add(keras.layers.Dropout(hp_dropout))
    model.add(keras.layers.Dense(hp_units4, activation=activation_choice, kernel_regularizer=kernel_regularizer))
    model.add(keras.layers.Dropout(hp_dropout))
    model.add(keras.layers.Dense(1))

    optimizer_choice = hp.Choice('optimizer', ['adam', 'rmsprop', 'sgd', 'nadam'])
    learning_rate = hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4])

    if optimizer_choice == 'adam':
        optimizer = keras.optimizers.Adam(learning_rate)
    elif optimizer_choice == 'rmsprop':
        optimizer = keras.optimizers.RMSprop(learning_rate)
    elif optimizer_choice == 'sgd':
        momentum = hp.Float('momentum', 0.0, 0.9, step=0.1)
        optimizer = keras.optimizers.SGD(learning_rate, momentum=momentum)
    elif optimizer_choice == 'nadam':
        optimizer = keras.optimizers.Nadam(learning_rate)

    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    return model

def get_callbacks():
    log_dir = os.path.join("my_logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    os.makedirs(log_dir, exist_ok=True)
    
    return [
        keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
        keras.callbacks.ModelCheckpoint("keras_model.keras", save_best_only=True),
        MonitorMAECallback(),
        keras.callbacks.TensorBoard(
            log_dir=log_dir,
            histogram_freq=1,
            write_graph=True,
            write_images=True,
            update_freq='epoch'
        )
    ]

def save_model_with_scaler(model, scaler, filename):
    bundle = {
        'model': model,
        'scaler': scaler
    }
    with open(filename, 'wb') as f:
        pickle.dump(bundle, f)

def load_model_with_scaler(filename):
    with open(filename, 'rb') as f:
        bundle = pickle.load(f)
    return bundle['model'], bundle['scaler']

def run_cross_validation(X, y, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    cv_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        print(f"\n=== Fold {fold + 1} ===")
        
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        
        tuner = kt.RandomSearch(
            hypermodel=build_model,
            objective='val_loss',
            max_trials=5,
            executions_per_trial=1,
            overwrite=True,
            directory="tuner_dir",
            project_name=f"math_operations_fold{fold}",
        )
        
        tuner.search(
            X_train, y_train,
            epochs=30,
            validation_data=(X_val, y_val),
            callbacks=get_callbacks(),
            verbose=1
        )
        
        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
        model = tuner.hypermodel.build(best_hps)
        
        history = model.fit(
            X_train, y_train,
            epochs=50,
            validation_data=(X_val, y_val),
            callbacks=get_callbacks(),
            verbose=0
        )
        
        val_loss, val_mae = model.evaluate(X_val, y_val, verbose=1)
        cv_scores.append(val_mae)
        print(f"Fold {fold + 1} - Val MAE: {val_mae:.6f}")
    
    print("\n=== Resultados Cross-Validation ===")
    print(f"MAE médio: {np.mean(cv_scores):.6f} (+/- {np.std(cv_scores):.6f})")
    return np.mean(cv_scores)

def testar_modelo_manual(model, scaler):
    """Interface simplificada para terminal"""
    print("\n" + "="*50)
    print("🔢 TESTE MANUAL DO MODELO".center(50))
    print("="*50)
    
    while True:
        print("\nOpções:")
        print("1. Testar com novos valores")
        print("2. Sair")
        
        escolha = input("\nEscolha uma opção: ")
        
        if escolha == '2':
            break
            
        try:
            print("\nInsira os valores para teste:")
            num1 = float(input("Número 1 (-100 a 100): "))
            num2 = float(input("Número 2 (-100 a 100): "))
            print("\nOperações disponíveis:")
            print("0: Adição (+)")
            print("1: Subtração (-)")
            print("2: Multiplicação (*)")
            print("3: Divisão (/)")
            op = int(input("Digite o número da operação: "))
            
            if op not in [0,1,2,3]:
                print("Erro: Operação inválida!")
                continue
                
            # Preprocessamento
            X_input = np.array([[num1/100, num2/100] + [1 if i == op else 0 for i in range(4)]])
            X_input = scaler.transform(X_input)
            
            # Predição
            y_pred = desnormalizar_saida(model.predict(X_input, verbose=0)[0][0])
            
            # Cálculo real
            ops = [
                lambda a,b: a+b,
                lambda a,b: a-b,
                lambda a,b: a*b,
                lambda a,b: a/b if b != 0 else float('inf')
            ]
            real = ops[op](num1, num2)
            
            # Exibir resultados
            print("\n" + " RESULTADOS ".center(50, '-'))
            print(f"Operação: {['Adição','Subtração','Multiplicação','Divisão'][op]}")
            print(f"Entrada: {num1} e {num2}")
            print(f"Resultado Real: {real:.4f}")
            print(f"Resultado Previsto: {y_pred:.4f}")
            print(f"Erro Absoluto: {abs(real - y_pred):.4f}")
            print("-"*50)
            
            # Plot simples (opcional)
            plt.figure(figsize=(8,3))
            plt.bar(['Real', 'Previsto'], [real, y_pred], color=['blue', 'orange'])
            plt.title('Comparação Resultado Real vs Previsto')
            plt.ylabel('Valor')
            plt.show()
            
        except Exception as e:
            print(f"\n⚠️ Erro: {str(e)}")
            print("Por favor, insira valores válidos.")

# === Fluxo Principal ===
if __name__ == "__main__":
    # 1. Geração de dados
    print("Gerando dados de treinamento...")
    X, y = train_datas(10000)
    operation_types = np.argmax(X[:, 2:6], axis=1)
    
    # 2. Divisão dos dados
    print("\nDividindo dados em conjuntos de treino/validação/teste...")
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=operation_types
    )
    
    op_types_temp = np.argmax(X_temp[:, 2:6], axis=1)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=42, stratify=op_types_temp
    )
    
    # 3. Normalização
    print("\nNormalizando dados...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    
    # 4. Validação Cruzada
    print("\n=== EXECUTANDO VALIDAÇÃO CRUZADA ===")
    cv_score = run_cross_validation(np.concatenate((X_train, X_val)), 
                                  np.concatenate((y_train, y_val)), 
                                  n_splits=3)
    
    # 5. Treinamento Final
    print("\n=== TREINAMENTO FINAL ===")
    erro_minimo = 0.005
    melhor_mae_atingido = float('inf')
    tentativas = 0
    limite_tentativas = 2
    training_history = TrainingHistory()
    
    while melhor_mae_atingido > erro_minimo and tentativas < limite_tentativas:
        print(f"\n===== Tentativa {tentativas + 1} =====")
        
        tuner = kt.RandomSearch(
            hypermodel=build_model,
            objective='val_loss',
            max_trials=5,
            executions_per_trial=2,
            overwrite=True,
            directory="tuner_dir",
            project_name="math_operations_final",
        )
        
        tuner.search(
            X_train, y_train,
            epochs=50,
            validation_data=(X_val, y_val),
            callbacks=get_callbacks(),
            verbose=1
        )
        
        trials = tuner.oracle.get_best_trials(num_trials=10)
        for trial_num, trial in enumerate(trials):
            hps = trial.hyperparameters.values
            
            for exec_num in range(2):
                model = tuner.hypermodel.build(trial.hyperparameters)
                model.fit(X_train, y_train, epochs=50, validation_data=(X_val, y_val), verbose=0)
                
                val_loss, val_mae = model.evaluate(X_val, y_val, verbose=0)
                test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
                
                training_history.add_result(
                    tentativas + 1, trial_num + 1, exec_num + 1,
                    val_mae, test_mae, val_loss, test_loss,
                    hps
                )
        
        best_hps = tuner.get_best_hyperparameters(num_trials=2)[0]
        model = tuner.hypermodel.build(best_hps)
        
        history = model.fit(
            np.concatenate((X_train, X_val)),
            np.concatenate((y_train, y_val)),
            epochs=100,
            batch_size=64,
            validation_split=0.2,
            callbacks=get_callbacks(),
            verbose=0
        )
        
        best_model = keras.models.load_model("keras_model.keras")
        val_loss, val_mae = best_model.evaluate(X_val, y_val, verbose=0)
        test_loss, test_mae = best_model.evaluate(X_test, y_test, verbose=0)
        
        print(f"\n📊 Tentativa {tentativas + 1} - Avaliação:")
        print(f"Validação - MAE: {val_mae:.6f} | Loss: {val_loss:.6f}")
        print(f"Teste     - MAE: {test_mae:.6f} | Loss: {test_loss:.6f}")
        
        if test_mae < melhor_mae_atingido:
            melhor_mae_atingido = test_mae
            save_model_with_scaler(best_model, scaler, 'melhor_modelo_tunado.pkl')
            print("✅ Novo melhor modelo salvo!")
        
        tentativas += 1
    
    # 6. Resultados Finais
    print("\n" + " RESULTADOS FINAIS ".center(50, '='))
    print(f"\nMelhor MAE no teste: {melhor_mae_atingido:.6f}")
    
    best_model, scaler = load_model_with_scaler('melhor_modelo_tunado.pkl')
    
    # 7. Avaliação Detalhada
    print("\n=== DESEMPENHO POR OPERAÇÃO ===")
    y_pred = best_model.predict(X_test)
    operation_types_test = np.argmax(X_test[:, 2:6], axis=1)
    
    for op, op_name in enumerate(['Adição', 'Subtração', 'Multiplicação', 'Divisão']):
        mask = operation_types_test == op
        y_true_op = y_test[mask]
        y_pred_op = y_pred[mask].flatten()
        mae = np.mean(np.abs(y_true_op - y_pred_op))
        mse = np.mean((y_true_op - y_pred_op) ** 2)
        print(f"{op_name}: MAE = {mae:.6f} | MSE = {mse:.6f}")
    
    # 8. Visualização da Evolução do Modelo
print("\n=== GRÁFICO DE EVOLUÇÃO DO MODELO ===")
plt.figure(figsize=(12, 6))

# Plotar histórico de treinamento do melhor modelo
if 'history' in locals():
    plt.subplot(1, 2, 1)
    plt.plot(history.history['mae'], label='Train MAE')
    plt.plot(history.history['val_mae'], label='Validation MAE')
    plt.title('Evolução do MAE durante o Treinamento')
    plt.ylabel('MAE')
    plt.xlabel('Época')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Evolução da Loss durante o Treinamento')
    plt.ylabel('Loss')
    plt.xlabel('Época')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

# 9. Dataset com Melhores Tentativas
print("\n=== RANKING DAS MELHORES TENTATIVAS ===")
history_df = training_history.get_dataframe()

# Adicionar coluna de desempenho combinado
history_df['Desempenho'] = history_df['Test_MAE'] * 0.7 + history_df['Val_MAE'] * 0.3

# Ordenar por desempenho
best_trials_df = history_df.sort_values(by='Desempenho').head(10)

# Exibir as melhores tentativas
print("\nTop 10 Melhores Tentativas:")
print(best_trials_df.to_string(index=False))

# Salvar histórico completo em CSV
history_df.to_csv('historico_treinamento_completo.csv', index=False)
print("\nHistórico completo salvo em 'historico_treinamento_completo.csv'")

# 10. Detalhamento por Época das Melhores Tentativas
print("\n=== DETALHAMENTO POR ÉPOCA DAS MELHORES TENTATIVAS ===")
best_trial_info = best_trials_df.iloc[0]

# Recriar o modelo da melhor tentativa para obter o histórico por época
tuner = kt.RandomSearch(
    hypermodel=build_model,
    objective='val_loss',
    max_trials=10,
    executions_per_trial=2,
    overwrite=False,
    directory="tuner_dir",
    project_name="math_operations_final",
)

# Obter os hiperparâmetros da melhor tentativa
best_hps = kt.HyperParameters()
for key, value in best_trial_info.items():
    if key in ['Units1', 'Units2', 'Units3', 'Dropout', 'Activation', 'Regularizer', 'Optimizer', 'Learning_Rate']:
        best_hps.values[key.lower()] = value

# Construir e treinar o modelo novamente para obter o histórico por época
model = tuner.hypermodel.build(best_hps)
history = model.fit(
    X_train, y_train,
    epochs=50,
    validation_data=(X_val, y_val),
    verbose=0
)

# Plotar evolução por época da melhor tentativa
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['mae'], label='Train MAE')
plt.plot(history.history['val_mae'], label='Validation MAE')
plt.title(f'Melhor Tentativa {best_trial_info["Tentativa"]}-{best_trial_info["Trial"]}-{best_trial_info["Execução"]}\nEvolução do MAE')
plt.ylabel('MAE')
plt.xlabel('Época')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title(f'Melhor Tentativa {best_trial_info["Tentativa"]}-{best_trial_info["Trial"]}-{best_trial_info["Execução"]}\nEvolução da Loss')
plt.ylabel('Loss')
plt.xlabel('Época')
plt.legend()

plt.tight_layout()
plt.show()

# Exibir estatísticas por época da melhor tentativa
best_epoch = np.argmin(history.history['val_mae'])
print(f"\nMelhor época: {best_epoch + 1}")
print(f"MAE de Treino na melhor época: {history.history['mae'][best_epoch]:.6f}")
print(f"MAE de Validação na melhor época: {history.history['val_mae'][best_epoch]:.6f}")
print(f"Loss de Treino na melhor época: {history.history['loss'][best_epoch]:.6f}")
print(f"Loss de Validação na melhor época: {history.history['val_loss'][best_epoch]:.6f}")

# 11. Teste Manual (mantido do código original)
testar_modelo_manual(best_model, scaler)