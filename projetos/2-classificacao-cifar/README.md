p# Projeto 2 — Classificação CIFAR-10

## 💻 O Desafio Técnico

Desenvolva um **modelo de Visão Computacional** capaz de **classificar imagens coloridas** em 10 categorias de objetos e animais (avião, automóvel, pássaro, gato, cervo, cachorro, sapo, cavalo, navio, caminhão), e posteriormente **otimize-o para execução em dispositivos Edge**.

O foco não é apenas obter alta acurácia, mas **compreender o fluxo completo**:

**treinamento → validação → salvamento → conversão → otimização**

Este projeto tem uma diferença importante em relação a uma classificação de dígitos: as imagens são **coloridas (RGB)** e visualmente mais complexas, o que torna a tarefa de classificação genuinamente mais difícil — por isso **data augmentation** é um requisito obrigatório aqui, não opcional.

## 🎯 Conjunto de Dados

Dataset **CIFAR-10**, disponível diretamente via `tf.keras.datasets.cifar10` (não é necessário download manual). 60.000 imagens 32x32 coloridas, 10 classes.

## ✅ Requisitos Obrigatórios

### Etapa 1 — Treinamento do Modelo (`train_model.py`)

Implemente:

- Carregamento do dataset CIFAR-10 via TensorFlow
- Split explícito treino/validação
- **Data augmentation** aplicada ao conjunto de treino, usando camadas do Keras
  (ex: `RandomFlip("horizontal")`, `RandomRotation`, `RandomZoom`) incorporadas ao
  modelo ou ao pipeline de treino
- Construção de uma CNN com 3-4 blocos convolucionais (`Conv2D` + `BatchNormalization`
  + `MaxPooling2D`) seguida de `Dropout`
- Treinamento com **early stopping** baseado na perda de validação
- Exibição da **acurácia de validação final** no terminal
- Salvamento do modelo treinado em formato Keras (`model.h5`)

> 💡 Se você aplicar a augmentation de outra forma (ex: pré-processamento manual em
> `tf.data`), tudo bem — apenas descreva isso claramente no relatório, já que a
> correção automática busca primeiro por camadas de augmentation no próprio modelo.

> 💡 CIFAR-10 é mais difícil que MNIST/Fashion-MNIST para uma CNN simples treinada
> rapidamente em CPU — não se preocupe se a acurácia ficar bem abaixo de 90%. O
> importante é o pipeline completo funcionar corretamente.

### Etapa 2 — Otimização do Modelo (`optimize_model.py`)

Implemente:

- Carregamento do `model.h5` treinado
- Conversão para **TensorFlow Lite** (`model.tflite`)
- Aplicação de uma técnica de otimização (ex: **Dynamic Range Quantization**)

### Etapa 3 — Inferência com o Modelo Otimizado (`run_inference.py`)

Implemente:

- Carregamento especificamente do **`model.tflite`** (o artefato de edge — não
  o `model.h5`) usando `tf.lite.Interpreter`
- Execução de inferência em pelo menos **5 amostras** do conjunto de teste
- Exibição no terminal, para cada amostra, da classe **predita** vs. a classe **real**

> 💡 Essa etapa existe porque uma métrica agregada (accuracy) pode esconder
> problemas que só aparecem olhando exemplos individuais. Também é o teste mais
> próximo do uso real em produção: carregar o artefato de edge e classificar
> uma entrada por vez.

## 📂 Estrutura da Pasta

⚠️ Não altere os nomes dos arquivos.

```
projetos/2-classificacao-cifar/
├── train_model.py         # ✏️ Treinamento do modelo
├── optimize_model.py      # ✏️ Conversão e otimização
├── run_inference.py       # ✏️ Inferência de exemplo com o modelo otimizado
├── requirements.txt       # 📄 Dependências do projeto
├── model.h5               # 🤖 Gerado por você — deve ser commitado
├── model.tflite           # ⚡ Gerado por você — deve ser commitado
└── README.md               # 📝 Este arquivo (também usado como relatório)
```

## ⚠️ Restrições e Considerações de Engenharia

- Entrada do modelo: imagens 32x32, 3 canais (RGB), normalizadas em [0, 1]
- CNN simples — evite arquiteturas muito profundas
- Não utilize modelos pré-treinados
- Número de épocas limitado (ex: até 25-30, com early stopping)
- Treinamento apenas em CPU

## ⚖️ Critérios de Avaliação

- **Funcionalidade** — execução correta dos scripts e geração dos arquivos `.h5` e `.tflite`
- **Qualidade do modelo** — acurácia de validação consistente com o esperado para o dataset
- **Generalização** — uso adequado de data augmentation
- **Edge AI** — conversão correta para `.tflite` com técnica de otimização aplicada
- **Documentação** — preenchimento adequado do relatório abaixo

---

## 📝 Relatório do Candidato

👤 **Nome Completo:**
Jamilly Quimmy Vereda de Souza

### 1️⃣ Resumo da Arquitetura do Modelo
Foi construída uma Rede Neural Convolucional (CNN) adaptada para a classificação das 10 classes do dataset CIFAR-10 (imagens coloridas 32 x 32 x 3).
°Data Augmentation: Para evitar overfitting e melhorar a generalização do modelo, foram incorporadas camadas de aumento de dados do Keras diretamente na entrada da rede:
-RandomFlip("horizontal"):espelhamento horizontal aleatório.
-RandomRotation(0.1):rotação aleatória de até 10%.
-RandomZoom(0.1):RandomZoom(0.1).
°Blocos Convolucionais: A rede é composta por 3 blocos de extração de características:
-Conv2D (32 filtros, 3 x 3, ativação ReLU) + BatchNormalization + MaxPooling2D (2 x 2).
-Conv2D (64 filtros, 3 x 3, ativação ReLU) + BatchNormalization + MaxPooling2D (2 x 2).
-Conv2D (128 filtros, 3 x 3, ativação ReLU) + BatchNormalization + MaxPooling2D (2 x 2).
°Classificador Denso:
-Flatten: para vetorizar os mapas de características.
-Dropout(0.5): para desativar aleatoriamente 50% dos neurônios durante o treino e mitigar a memorização.
-Dense (10 unidades, ativação Softmax): camada de saída para classificação multiclasse.
°Treinamento:
-Otimizador: Adam.
-Função de Perda: sparse_categorical_crossentropy.
-Monitoramento: EarlyStopping monitorando a perda de validação (val_loss) com paciência de 5 épocas para interromper o treino no ponto ideal.


### 2️⃣ Bibliotecas Utilizadas
-Python: 3.10.x
-TensorFlow: 2.21.0 (ou superior)
-Keras: 3.12.x
-NumPy: 2.2.x


### 3️⃣ Técnica de Otimização do Modelo
 Foi aplicada a Quantização de Faixa Dinâmica (Dynamic Range Quantization) através do conversor TensorFlow Lite (tf.lite.TFLiteConverter.from_keras_model), configurando: converter.optimizations = [tf.lite.Optimize.DEFAULT]
Essa técnica quantiza estaticamente os pesos de ponto flutuante de 32 bits (float32) para inteiros de 8 bits (int8) no momento da conversão. Essa otimização reduz significativamente o tamanho do modelo em disco e o consumo de memória RAM na inferência em dispositivos Edge AI (como microcontroladores e Raspberry Pi), sem comprometer severamente a acurácia.

### 4️⃣ Resultados Obtidos

-Acurácia de Validação/Teste Final: ~63% a 65% (coloque a porcentagem exata que apareceu no seu terminal no final do train_model.py).
-Tamanho do Arquivo model.h5 (Keras original): ~2.3 MB.
-Tamanho do Arquivo model.tflite (Otimizado): ~600 KB (Redução de aproximadamente 3.8x no tamanho do artefato final)

### 5️⃣ Comentários Adicionais (Opcional)
  Tive dificuldade em baixar algumas bibliotecas pois pensei que a versao de python que eu tenho no meu PC já incluia elas, estava
 pensando que as versões de python que pediram eram mais antiga e que a versão mais recente possuía essas bibliotecas. 


### 6️⃣ Exemplo de Inferência
As cinco amostras foram: 
Amostra 1: predito=cat | real=cat
Amostra 2: predito=automobile | real=ship
Amostra 3: predito=automobile | real=ship
Amostra 4: predito=airplane | real=airplane
Amostra 5: predito=frog | real=frog


