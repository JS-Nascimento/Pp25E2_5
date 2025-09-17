# Projeto de Machine Learning - Análise de Algoritmos e Técnicas

Este repositório contém implementações e análises de diferentes algoritmos de machine learning aplicados a diversos datasets. Cada branch TPxx representa um trabalho prático específico com técnicas e abordagens distintas.

## 📊 Resumo dos Trabalhos Práticos

### TP01 - Classificação KNN com Heart Disease Dataset

**Arquivo**: `projeto_knn_heart_unico.ipynb`

**Técnicas Implementadas**:
- **Algoritmo**: K-Nearest Neighbors (KNN)
- **Pré-processamento**: 
  - OneHotEncoder para variáveis categóricas
  - StandardScaler para variáveis numéricas
- **Validação**: Divisão treino/teste (80%/20%) com estratificação

**Features Implementadas**:
- Carregamento e identificação automática de features/target
- Pipeline de transformação de dados com ColumnTransformer
- Análise do impacto do parâmetro K (1 a 20)
- Matriz de confusão para visualização de resultados
- Gráfico de acurácia vs valor de K

**Resultados Obtidos**:
- Melhor acurácia obtida com análise de diferentes valores de K
- Visualização clara do trade-off entre complexidade e performance
- Matriz de confusão demonstrando classificação binária (doença cardíaca)

**Informações Técnicas**:
- Dataset: Heart Disease (classificação binária)
- Métrica principal: Acurácia
- Estratégia de validação: Hold-out simples
- Foco: Análise de hiperparâmetros e impacto do K no KNN

---

### TP02 - Detecção de Fake News com TF-IDF e KNN

**Arquivo**: `tp2.ipynb`

**Técnicas Implementadas**:
- **Algoritmo**: K-Nearest Neighbors aplicado a texto
- **Vetorização**: TF-IDF (Term Frequency-Inverse Document Frequency)
- **Pré-processamento**: 
  - Combinação de título + texto
  - Remoção de stop words
  - N-gramas (unigramas e bigramas)

**Features Implementadas**:
- Vetorização TF-IDF com 5000 features máximas
- Análise comparativa de diferentes valores de K (3, 5, 7, 9, 11, 15, 21, 31, 51)
- Validação cruzada estratificada (5 folds)
- Múltiplas métricas de avaliação

**Resultados Obtidos**:
- **Métricas de Performance**:
  - Acurácia, Precisão, Recall, F1-Score
  - Especificidade e Sensibilidade
  - AUC-ROC com curva ROC
  - Curva Precision-Recall

**Visualizações**:
- Matriz de confusão detalhada
- Curva ROC com AUC
- Gráfico de impacto do K nas acurácias
- Distribuição de probabilidades preditas

**Informações Técnicas**:
- Dataset: Fake News (Fake.csv + True.csv)
- Classificação binária: Fake (0) vs True (1)
- Vetorização: TF-IDF com parâmetros otimizados
- Validação: Cross-validation + hold-out test
- Interpretação automática dos resultados baseada em métricas

---

### TP03 - Classificação Sonar com PCA e Decision Tree

**Arquivo**: `sonar_pca_tree.ipynb`

**Técnicas Implementadas**:
- **Algoritmo**: Decision Tree Classifier
- **Redução de Dimensionalidade**: PCA (Principal Component Analysis)
- **Pipeline**: StandardScaler → PCA → DecisionTree
- **Otimização**: GridSearchCV com validação cruzada

**Features Implementadas**:
- Pipeline automatizado com StandardScaler, PCA e Decision Tree
- Busca de hiperparâmetros para:
  - Número de componentes PCA (5, 10, 15, 20, 30, None)
  - Critério de divisão (gini, entropy)
  - Parâmetros de profundidade e folhas
  - Pruning via ccp_alpha

**Resultados Obtidos**:
- **Métricas Completas**:
  - Acurácia, Precisão, Recall, Especificidade
  - F1-Score e ROC AUC
  - Matriz de confusão
  - Relatório de classificação detalhado

**Visualizações**:
- Curva ROC com AUC
- Gráfico de variância explicada acumulada pelo PCA
- Interpretação automática dos resultados

**Informações Técnicas**:
- Dataset: Sonar (detecção de minas vs rochas)
- Classificação binária: R (rock) = 0, M (mine) = 1
- Estratégia: StratifiedKFold 5-fold cross-validation
- Scoring: ROC AUC para otimização
- Pruning: Otimização automática do ccp_alpha

---

### TP04 - Feature Engineering com K-Means e SVM

**Arquivo**: `experimento_kmeans_svm.ipynb`

**Técnicas Implementadas**:
- **Clustering**: K-Means para feature engineering
- **Classificadores**: SVM (Linear, Polynomial, RBF) e Random Forest
- **Feature Engineering**: Distância ao centróide mais próximo
- **Otimização**: GridSearchCV para múltiplos kernels

**Features Implementadas**:
- Seleção automática de K ótimo via:
  - Método do cotovelo (inércia)
  - Índice de silhueta
- Criação de nova feature: distância mínima aos centróides
- Comparação baseline vs feature expandida
- GridSearch para SVM e Random Forest

**Resultados Obtidos**:
- **Análise Comparativa**:
  - Performance baseline vs com feature de clustering
  - Impacto de diferentes valores de K
  - Múltiplas métricas (Accuracy, Precision, Recall, F1, AUC)

**Visualizações**:
- Gráficos de método do cotovelo e silhueta
- Curvas ROC comparativas (baseline vs expandido)
- Análise de performance por valor de K
- Comparativo de ganhos por modelo

**Informações Técnicas**:
- Dataset: Spotify tracks (114.000 amostras, 21 features)
- Pré-processamento: StandardScaler para SVM e K-Means
- Estratégia: Hold-out (80%/20%) com estratificação
- Feature Engineering: Distância euclidiana aos centróides
- Modelos: SVM (3 kernels) + Random Forest com hiperparâmetros otimizados

---

### TP05 - Pipeline Completo de Text Mining

**Arquivo**: `text_ml_pipeline.ipynb`

**Técnicas Implementadas**:
- **Modelagem de Tópicos**: Latent Dirichlet Allocation (LDA)
- **Classificação**: Regressão Logística com TF-IDF
- **Visualização**: t-SNE para agrupamentos
- **Explicabilidade**: LIME e SHAP

**Features Implementadas**:
- **Pipeline Completo de Texto**:
  - Limpeza e pré-processamento automático
  - Seleção automática de número de tópicos (coerência/perplexidade)
  - TF-IDF com n-gramas e stop words
  - GridSearch para otimização de hiperparâmetros

**Modelagem de Tópicos**:
- LDA com seleção automática de número de tópicos
- Análise de coerência (c_v) ou perplexidade como proxy
- Extração e salvamento de top palavras por tópico

**Visualizações e Explicabilidade**:
- **t-SNE**: Visualização 2D dos documentos
- **LIME**: Explicação local de instâncias individuais
- **SHAP**: Análise de importância de features globais
- **Curva ROC**: Avaliação de performance

**Resultados Obtidos**:
- **Métricas de Classificação**:
  - Acurácia: 90.45%
  - Precisão: 89.93%
  - Recall: 91.10%
  - F1-Score: 90.51%
  - AUC-ROC: 96.60%

**Análise Automática**:
- Identificação de palavras mais discriminativas
- Análise de tópicos extraídos pelo LDA
- Geração automática de conclusões baseadas em métricas

**Informações Técnicas**:
- Dataset: IMDB Movie Reviews (50.000 reviews)
- Classificação de sentimento binária (positivo/negativo)
- Pipeline: TF-IDF → Regressão Logística
- Explicabilidade: LIME + SHAP com visualizações
- Outputs automatizados: gráficos, relatórios HTML e CSV

---

## 🛠️ Tecnologias e Bibliotecas Utilizadas

### Core ML/Data Science
- **scikit-learn**: Algoritmos de ML, pré-processamento, validação
- **pandas**: Manipulação e análise de dados
- **numpy**: Computação numérica
- **matplotlib/seaborn**: Visualização de dados

### Técnicas Específicas
- **TF-IDF**: Vetorização de texto (TP02, TP05)
- **PCA**: Redução de dimensionalidade (TP03)
- **LDA**: Modelagem de tópicos (TP05)
- **t-SNE**: Visualização de alta dimensionalidade (TP05)

### Explicabilidade e Interpretação
- **LIME**: Local Interpretable Model-agnostic Explanations (TP05)
- **SHAP**: SHapley Additive exPlanations (TP05)

### Algoritmos Implementados
- **KNN**: K-Nearest Neighbors (TP01, TP02)
- **Decision Tree**: Árvores de decisão com pruning (TP03)
- **SVM**: Support Vector Machines (kernels linear, poly, RBF) (TP04)
- **Random Forest**: Ensemble de árvores (TP04)
- **Logistic Regression**: Regressão logística (TP05)
- **K-Means**: Clustering para feature engineering (TP04)

## 📈 Principais Contribuições Acadêmicas

1. **Análise Comparativa de Algoritmos**: Comparação sistemática de diferentes abordagens em diversos tipos de dados

2. **Feature Engineering Avançado**: Uso de clustering para criação de novas features (TP04)

3. **Pipeline Completo de Text Mining**: Implementação end-to-end desde pré-processamento até explicabilidade (TP05)

4. **Otimização de Hiperparâmetros**: GridSearchCV sistemático em todos os projetos

5. **Validação Robusta**: Uso de validação cruzada estratificada e múltiplas métricas

6. **Interpretabilidade**: Implementação de técnicas de explicabilidade (LIME, SHAP)

7. **Visualização Avançada**: t-SNE, curvas ROC, matrizes de confusão e análises gráficas

## 🎯 Datasets Utilizados

- **Heart Disease**: Classificação médica binária
- **Fake News**: Detecção de notícias falsas (texto)
- **Sonar**: Detecção de objetos subaquáticos
- **Spotify Tracks**: Classificação de características musicais
- **IMDB Reviews**: Análise de sentimento em reviews de filmes

## 📊 Métricas e Avaliação

Todos os projetos implementam avaliação abrangente com:
- Acurácia, Precisão, Recall, F1-Score
- AUC-ROC e curvas ROC
- Matrizes de confusão
- Validação cruzada estratificada
- Análise de hiperparâmetros
- Interpretação automática de resultados

---

**Autor**: Jorge Nascimento  
**Objetivo**: Demonstração prática de técnicas de Machine Learning e Text Mining com foco em reprodutibilidade e interpretabilidade.