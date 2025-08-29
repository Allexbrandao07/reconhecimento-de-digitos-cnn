# Análise Detalhada: Reconhecimento de Dígitos Manuscritos com CNN e Keras

## 1. Visão Geral do Projeto
Este notebook implementa uma Rede Neural Convolucional (CNN) para resolver um problema clássico de visão computacional: o reconhecimento de dígitos manuscritos.  

Utilizando a biblioteca **Keras** (com **TensorFlow** como backend), o projeto abrange todas as etapas do fluxo de trabalho de um modelo de **deep learning**, desde o carregamento e pré-processamento dos dados até o treinamento, avaliação e previsão de novas imagens fornecidas pelo usuário.  

O dataset utilizado para o treinamento é o **MNIST**, uma vasta coleção de imagens de dígitos escritos à mão.

---

## 2. Bibliotecas Utilizadas
O código utiliza um conjunto de bibliotecas padrão para projetos de **Machine Learning** e **Visão Computacional** em Python:

- **Keras (TensorFlow)**: Para construção e treinamento da rede neural.
  - `keras.datasets.mnist`: Para carregar o dataset MNIST diretamente.
  - `keras.models.Sequential`: Para criar o modelo sequencial, camada por camada.
  - `keras.layers`: Contém as camadas da rede, como `Conv2D`, `MaxPooling2D`, `Flatten` e `Dense`.
  - `keras.utils`: Ferramentas como `to_categorical` para codificação de rótulos.
- **Matplotlib.pyplot**: Para visualização de imagens e resultados.
- **OpenCV (cv2) e NumPy**: Para manipulação e pré-processamento de imagens externas.

---

## 3. Análise das Etapas do Código

### 3.1. Carregamento e Visualização do Dataset
- Dataset **MNIST** carregado em conjuntos de **treinamento** e **teste** (`X_treinamento, y_treinamento, X_teste, y_teste`).
- Uma imagem de exemplo do conjunto de treinamento é exibida usando **matplotlib** com o título mostrando a classe real (ex: `Classe = 4`).

### 3.2. Pré-processamento dos Dados
- **Redimensionamento (Reshape)**: `(28, 28)` → `(28, 28, 1)` adicionando dimensão de canal.
- **Normalização**: Pixels 0–255 → `float32` / 255 → valores entre 0–1.
- **One-Hot Encoding**: Rótulos (0–9) → formato categórico com `utils.to_categorical`.  
  Ex: `3` → `[0, 0, 0, 1, 0, 0, 0, 0, 0, 0]`.

### 3.3. Construção do Modelo CNN
- **Conv2D**: 32 filtros, kernel 3x3, ativação `ReLU`.
- **MaxPooling2D**: Redução de dimensionalidade.
- **Flatten**: Matriz 2D → vetor 1D.
- **Dense**: 
  - 128 neurônios, ativação `ReLU`
  - Camada de saída: 10 neurônios, ativação `softmax`.

### 3.4. Compilação e Treinamento
- **Compilação**: `optimizer=adam`, `loss=categorical_crossentropy`, `metrics=accuracy`.
- **Treinamento**: `.fit()` por 5 épocas, validando com dados de teste.  
  - Acurácia final: ~98% no conjunto de validação.

### 3.5. Avaliação e Previsão com Imagens
- Previsão de imagens do conjunto de teste confirmando dígitos corretos.
- Função `previsao(file)` para imagens externas:
  1. Carrega imagem em escala de cinza (`cv2`).
  2. Redimensiona para 28x28 pixels.
  3. Inverte cores (`cv2.bitwise_not`) para compatibilidade com MNIST.
  4. Prepara imagem para o modelo e realiza previsão.

- Exemplos:
  - `MNIST_imagem_exemplo_1.jpg` → `2`
  - `MNIST_imagem_exemplo2.jpg` → `3`

---

## 4. Como Utilizar o Código e as Imagens de Exemplo

### 4.1 Pré-requisitos
- Python 3
- Jupyter Notebook ou outro ambiente Python
- Bibliotecas necessárias:

pip install tensorflow matplotlib opencv-python

### 4.2 Organize os Arquivos
- **Notebook:** `CNN_Keras_MNIST.ipynb`  
- **Imagens:** `MNIST_imagem_exemplo_1.jpg` e `MNIST_imagem_exemplo2.jpg`  
  Todos na mesma pasta.

### 4.3 Execute o Notebook
- Abra o **Jupyter Notebook** e execute as células sequencialmente.  
- O dataset **MNIST** será baixado automaticamente na primeira execução.

### 4.4 Teste com Imagens de Exemplo
- As últimas células já carregam `MNIST_imagem_exemplo_1.jpg` e `MNIST_imagem_exemplo2.jpg` e exibem a previsão.

### 4.5 Teste com Suas Próprias Imagens
1. Crie uma imagem simples de um dígito (fundo branco, dígito preto).  
2. Salve na mesma pasta do notebook.  
3. Altere o nome do arquivo na função `previsao()`:

```python
previsao('minha_imagem_do_numero_5.jpg')



