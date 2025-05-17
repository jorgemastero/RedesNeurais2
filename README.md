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
![Evolução do MAE e Lss](https://github.com/user-attachments/assets/f4c07d83-d068-4fc4-a333-193e1e7b3b73)

        MAE e Loss durante o treinamento
        
![duranteTreino](https://github.com/user-attachments/assets/34ce78ad-9634-4ebc-850b-152fa7799ba1)

        Comparação entre treino e validação
        
![evoluçãoMelhorTentativa](https://github.com/user-attachments/assets/4a6b6bc4-01c6-41ee-96cb-233fbd098608)
    
        Ranking das 10 melhores configurações
        
![RankingTentativas](https://github.com/user-attachments/assets/d7c17fb8-e8d8-48b8-af9e-cb0f52a65078)

        Métricas por Operação
        
![avaliaçãoFinal](https://github.com/user-attachments/assets/b056f8bd-10ca-4a56-9310-804402d7b67d)
    
        Exemplo de resultados:

![testeManual](https://github.com/user-attachments/assets/e47454bf-f431-4737-b5f8-ef58345141ea)
![testeManual2](https://github.com/user-attachments/assets/6c057c53-794f-40a9-8f53-ecd0527af572)
![testeManualDivisão](https://github.com/user-attachments/assets/7a9f156a-9907-4c41-a12a-436565214316)
      
# Modelo Salvo

    melhor_modelo_tunado.pkl contendo o modelo e scaler
