# RedesNeurais2
Preditor de Operações Matemáticas com Machine Learning
📌 Visão Geral do Projeto
Este projeto implementa uma rede neural para prever resultados de operações matemáticas básicas (adição, subtração, multiplicação e divisão) a partir de pares de números e indicadores de operação. O sistema inclui:

Geração de dados sintéticos balanceados

Otimização avançada de hiperparâmetros com Keras Tuner

Monitoramento completo do treinamento

Modelo pronto para implantação

🛠 Componentes Principais
🔧 Funcionalidades Centrais
Geração de Dados: Cria conjuntos balanceados de operações matemáticas

Pré-processamento: Normalização e divisão estratificada dos dados

Arquitetura do Modelo: Rede neural de 3 camadas configurável

Otimização de Hiperparâmetros: Busca aleatória com Keras Tuner

Avaliação: Validação cruzada e métricas específicas por operação

📊 Monitoramento de Desempenho
Classe TrainingHistory: Registra todos os resultados de treinamento

Integração com TensorBoard: Visualiza métricas de treinamento

Callbacks Personalizados: Monitoramento em tempo real do MAE

🚀 Começando
⚙️ Instalação
bash
pip install tensorflow scikit-learn keras-tuner pandas numpy matplotlib ipywidgets
🏃 Executando o Projeto
python
python preditor_operacoes_matematicas.py
📈 Arquitetura do Modelo
python
Modelo: "sequential"
_________________________________________________________________
 Camada (Tipo)               Formato de Saída         Parâmetros   
=================================================================
 dense (Dense)               (None, 64-512)           448-3,072 
                                                                 
 dropout (Dropout)           (None, 64-512)           0         
                                                                 
 dense_1 (Dense)             (None, 32-256)           2,080-131,328
                                                                 
 dropout_1 (Dropout)         (None, 32-256)           0         
                                                                 
 dense_2 (Dense)             (None, 16-128)           528-32,896
                                                                 
 dense_3 (Dense)             (None, 1)                17-129    
                                                                 
=================================================================
📋 Principais Recursos
Recurso	Descrição	Implementação
Geração de Dados	Operações matemáticas balanceadas	Função train_datas()
Otimização	Ajuste de arquitetura e parâmetros	Keras Tuner RandomSearch
Validação	Validação cruzada estratificada	sklearn KFold
Pronto para Produção	Empacotamento modelo+scaler	save_model_with_scaler()
📊 Desempenho Esperado
Métrica	Valor Alvo	Limite Aceitável
MAE Teste	< 0.005	< 0.01
MSE Teste	< 0.0001	< 0.0005
Diferença Val/Test	< 15%	< 25%
💡 Exemplos de Uso
Teste Interativo
python
# Carregar modelo salvo
model, scaler = load_model_with_scaler('melhor_modelo_tunado.pkl')

# Criar interface
create_test_interface(model, scaler)
Análise de Treinamento
python
# Obter histórico de treinamento
results_df = training_history.get_dataframe()

# Mostrar top 5 modelos

print(results_df.sort_values('Test_MAE').head(5))

🌟 Boas Práticas Implementadas
✅ Divisão estratificada dos dados
✅ Otimização abrangente de hiperparâmetros
✅ Prevenção de overfitting (dropout, regularização)
✅ Monitoramento completo do treinamento
✅ Serialização pronta para produção

📅 Histórico de Versões

v1.0: Versão inicial com funcionalidades básicas

v1.1: Adicionada validação cruzada e melhorias no logging

v1.2: Aprimoramento no empacotamento e interface                
