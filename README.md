# Modelo de Rede Neural para Operações Matemáticas


Este projeto implementa uma rede neural para prever resultados de operações matemáticas básicas (adição, subtração, multiplicação e divisão) utilizando TensorFlow e Keras Tuner para otimização de hiperparâmetros.

# 📋 Sumário
Funcionalidades
    
    Uso
    
    Estrutura do Código
    
    Fluxo de Trabalho
    
    Resultados
    
    Contribuição
    
# ✨ Funcionalidades

    ✅ Geração automática de dados de treinamento
    
    ✅ Normalização e pré-processamento inteligente
    
    ✅ Modelagem com arquitetura flexível e configurável
    
    ✅ Otimização de hiperparâmetros com Keras Tuner
    
    ✅ Validação cruzada robusta
    
    ✅ Múltiplas tentativas de treinamento com persistência do melhor modelo
    
    ✅ Avaliação detalhada por tipo de operação
    
    ✅ Interface interativa para teste manual
    
    ✅ Visualização completa da evolução do treinamento

# O script irá:

    Gerar dados de treinamento
    
    Treinar e otimizar o modelo
    
    Salvar o melhor modelo encontrado
    
    Exibir resultados e gráficos de desempenho
    
    Para testar o modelo manualmente após o treinamento:


# Carregar modelo e scaler
    model, scaler = load_model_with_scaler('melhor_modelo_tunado.pkl')

# Testar manualmente
    testar_modelo_manual(model, scaler)
    📂 Estrutura do Código
    math-operations-model/
    │
    ├── math_operations_model.py  # Script principal
    ├── requirements.txt          # Dependências
    ├── my_logs/                  # Logs do TensorBoard
    ├── tuner_dir/                # Resultados do Keras Tuner
    └── README.md                 # Este arquivo
    
# 🔄 Fluxo de Trabalho
    Geração de Dados
    
    Cria 10.000 exemplos balanceados de operações matemáticas
    
    Garante distribuição uniforme entre as operações
    
    Pré-processamento
    
    Normalização de entradas e saídas
    
    Codificação one-hot das operações
    
    Divisão em conjuntos de treino/validação/teste (60%/20%/20%)
    
    Modelagem
    
    Arquitetura: 4 camadas densas com dropout

# Hiperparâmetros otimizáveis:

    Unidades por camada
    
    Taxa de dropout
    
    Função de ativação
    
    Regularização
    
    Otimizador e taxa de aprendizado
    
    Treinamento
    
    Validação cruzada com 3 folds
    
    Busca aleatória de hiperparâmetros
    
    Múltiplas tentativas até atingir MAE < 0.005
    
    Avaliação
    
    Desempenho por tipo de operação
    
    Gráficos de evolução do treinamento
    
    Ranking das melhores tentativas

# 📊 Resultados
    O modelo gera automaticamente:
    
        Gráficos de Evolução
        ![Evolução do MAE e Lss](https://github.com/user-attachments/assets/0d5aa7a0-e59a-4e62-8269-497896926f0a)
        ![evoluçãoMelhorTentativa](https://github.com/user-attachments/assets/c3f134b9-6274-4e5a-9ab0-be02f051d985)


    
        MAE e Loss durante o treinamento
    
        Comparação entre treino e validação
    
        Dataset de Resultados
    
        historico_treinamento_completo.csv com todas as tentativas
    
        Ranking das 10 melhores configurações
    
        Métricas por Operação
    
        MAE e MSE específicos para cada operação matemática

# Modelo Salvo

    melhor_modelo_tunado.pkl contendo o modelo e scaler
