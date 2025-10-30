# Feature Engineering: Bibliotecas, Técnicas e Recursos

**Data de Atualização:** 30 de Outubro de 2025  
**Fonte:** Busca intensiva

Este documento contém uma coleção abrangente de recursos sobre Feature Engineering, incluindo bibliotecas, técnicas de extração, seleção e transformação de features, além de métodos de redução de dimensionalidade.

---

## 📋 Índice

- [Bibliotecas de Feature Engineering](#bibliotecas-de-feature-engineering)
- [Técnicas de Seleção de Features](#técnicas-de-seleção-de-features)
- [Redução de Dimensionalidade](#redução-de-dimensionalidade)
- [Feature Importance](#feature-importance)
- [Conceitos Fundamentais](#conceitos-fundamentais)

---

## Bibliotecas de Feature Engineering

### scikit-learn Feature Extraction
- **URL:** [https://scikit-learn.org/stable/modules/feature_extraction.html](https://scikit-learn.org/stable/modules/feature_extraction.html)
- **Tipo:** Módulo de Biblioteca
- **Descrição:** Módulo para extrair features de datasets em formatos suportados por algoritmos de ML. O módulo sklearn.feature_extraction pode ser usado para extrair features em um formato suportado por algoritmos de machine learning a partir de datasets consistindo de formatos como texto e imagem.

**Principais Funcionalidades:**
- **DictVectorizer:** Converte dicionários de features em arrays NumPy
- **FeatureHasher:** Implementa feature hashing para vetorização de alta velocidade
- **Text Feature Extraction:** Extração de features de texto
  - CountVectorizer: Bag of Words
  - TfidfVectorizer: TF-IDF
  - HashingVectorizer: Hashing trick
- **Image Feature Extraction:** Extração de patches de imagens

**Aplicações:**
- Pré-processamento de texto
- Vetorização de dados categóricos
- Extração de features de imagens

---

### Feature-engine
- **URL:** [https://github.com/feature-engine/feature_engine](https://github.com/feature-engine/feature_engine)
- **Tipo:** Biblioteca Python
- **Descrição:** Biblioteca Python com múltiplos transformadores para engenharia e seleção de features para uso em modelos de machine learning. Feature-engine é uma biblioteca Python com múltiplos transformadores para engenharia e seleção de features.

**Principais Transformadores:**
- **Missing Data Imputation:** Imputação de dados faltantes
- **Categorical Encoding:** Codificação de variáveis categóricas
- **Discretisation:** Discretização de variáveis contínuas
- **Outlier Handling:** Tratamento de outliers
- **Variable Transformation:** Transformação de variáveis
- **Feature Selection:** Seleção de features

**Vantagens:**
- Compatível com scikit-learn pipelines
- Fácil de usar
- Bem documentado
- Código aberto

---

### Featuretools
- **URL:** [https://featuretools.alteryx.com/](https://featuretools.alteryx.com/)
- **Tipo:** Framework
- **Descrição:** Framework para realizar engenharia de features automatizada, se destacando em datasets temporais e relacionais. Featuretools é um framework para realizar automated feature engineering, excelling em transformar datasets temporais e relacionais em matrizes de features para machine learning.

**Conceitos Principais:**
- **Deep Feature Synthesis (DFS):** Algoritmo para criar features automaticamente
- **Entities:** Tabelas de dados
- **Relationships:** Relações entre tabelas
- **Primitives:** Operações básicas de transformação e agregação

**Aplicações:**
- Automated feature engineering
- Dados temporais
- Dados relacionais
- Redução de tempo de desenvolvimento

**Exemplo de Uso:**
```python
import featuretools as ft

# Criar entityset
es = ft.EntitySet(id="customers")

# Adicionar entidades
es = es.add_dataframe(dataframe_name="customers", dataframe=customers_df, index="customer_id")
es = es.add_dataframe(dataframe_name="transactions", dataframe=transactions_df, index="transaction_id", time_index="transaction_time")

# Adicionar relacionamento
es = es.add_relationship("customers", "customer_id", "transactions", "customer_id")

# Executar DFS
feature_matrix, feature_defs = ft.dfs(entityset=es, target_dataframe_name="customers")
```

---

## Técnicas de Seleção de Features

### SHAP (SHapley Additive exPlanations)
- **URL:** [https://github.com/slundberg/shap](https://github.com/slundberg/shap)
- **Tipo:** Biblioteca Python
- **Descrição:** Abordagem baseada em teoria dos jogos para explicar a saída de qualquer modelo de machine learning. SHAP (SHapley Additive exPlanations) é uma abordagem baseada em teoria dos jogos para explicar a saída de qualquer modelo de machine learning.

**Conceitos Fundamentais:**
- **Shapley Values:** Valores de Shapley da teoria dos jogos cooperativos
- **Additive Feature Attribution:** Atribuição aditiva de features
- **Model-Agnostic:** Funciona com qualquer modelo

**Tipos de Explainers:**
- **TreeExplainer:** Para modelos baseados em árvores (XGBoost, LightGBM, CatBoost)
- **DeepExplainer:** Para redes neurais profundas
- **KernelExplainer:** Model-agnostic, baseado em LIME
- **LinearExplainer:** Para modelos lineares

**Aplicações:**
- Explicabilidade de modelos
- Feature selection baseada em importância
- Debugging de modelos
- Interpretação de predições

**Visualizações:**
- Summary plots
- Dependence plots
- Force plots
- Waterfall plots

---

## Redução de Dimensionalidade

### PCA, t-SNE, UMAP
- **URL:** [https://medium.com/@aastha.code/dimensionality-reduction-pca-t-sne-and-umap-41d499da2df2](https://medium.com/@aastha.code/dimensionality-reduction-pca-t-sne-and-umap-41d499da2df2)
- **Tipo:** Técnicas
- **Descrição:** Técnicas de redução de dimensionalidade para visualização e pré-processamento de dados.

### Principal Component Analysis (PCA)
**Descrição:** Técnica linear de redução de dimensionalidade que encontra as direções de máxima variância nos dados.

**Características:**
- Método linear
- Preserva variância global
- Rápido e eficiente
- Determinístico

**Quando Usar:**
- Dados com relações lineares
- Pré-processamento para ML
- Redução de ruído
- Compressão de dados

**Implementação (scikit-learn):**
```python
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X)
```

### t-SNE (t-Distributed Stochastic Neighbor Embedding)
**Descrição:** Técnica não-linear de redução de dimensionalidade especialmente adequada para visualização de dados de alta dimensão.

**Características:**
- Método não-linear
- Preserva estrutura local
- Excelente para visualização
- Estocástico (resultados variam)

**Quando Usar:**
- Visualização de dados
- Exploração de clusters
- Dados com estrutura não-linear

**Limitações:**
- Lento para grandes datasets
- Não preserva distâncias globais
- Não determinístico

**Implementação (scikit-learn):**
```python
from sklearn.manifold import TSNE

tsne = TSNE(n_components=2, perplexity=30)
X_embedded = tsne.fit_transform(X)
```

### UMAP (Uniform Manifold Approximation and Projection)
**Descrição:** Técnica moderna de redução de dimensionalidade que preserva tanto estrutura local quanto global.

**Características:**
- Método não-linear
- Preserva estrutura local e global
- Mais rápido que t-SNE
- Melhor escalabilidade

**Vantagens sobre t-SNE:**
- Mais rápido
- Preserva estrutura global
- Melhor para grandes datasets
- Pode ser usado para transformação de novos dados

**Quando Usar:**
- Visualização de grandes datasets
- Preservação de estrutura global
- Clustering
- Pré-processamento para ML

**Implementação (umap-learn):**
```python
import umap

reducer = umap.UMAP(n_components=2)
X_embedded = reducer.fit_transform(X)
```

### Comparação: PCA vs t-SNE vs UMAP

| Característica | PCA | t-SNE | UMAP |
|---|---|---|---|
| Linearidade | Linear | Não-linear | Não-linear |
| Velocidade | Rápido | Lento | Médio |
| Estrutura Local | Não | Sim | Sim |
| Estrutura Global | Sim | Não | Sim |
| Determinístico | Sim | Não | Não |
| Escalabilidade | Excelente | Ruim | Boa |
| Novos Dados | Sim | Não | Sim |

---

## Feature Importance

### XGBoost Feature Importance
- **URL:** [https://www.machinelearningmastery.com/feature-importance-and-feature-selection-with-xgboost-in-python/](https://www.machinelearningmastery.com/feature-importance-and-feature-selection-with-xgboost-in-python/)
- **Tipo:** Técnica
- **Descrição:** Como estimar a importância de features para um problema de modelagem preditiva usando XGBoost.

**Tipos de Feature Importance no XGBoost:**

1. **Weight (Frequency):**
   - Número de vezes que uma feature aparece em uma árvore
   - Métrica padrão

2. **Gain:**
   - Melhoria média de ganho quando a feature é usada em splits
   - Considera a qualidade dos splits

3. **Cover:**
   - Número médio de observações afetadas pelos splits usando a feature
   - Considera a quantidade de dados

**Implementação:**
```python
from xgboost import XGBClassifier
import matplotlib.pyplot as plt

# Treinar modelo
model = XGBClassifier()
model.fit(X_train, y_train)

# Plot feature importance
plt.figure(figsize=(10, 8))
xgb.plot_importance(model, importance_type='gain')
plt.show()

# Obter importâncias
importances = model.feature_importances_
```

**Aplicações:**
- Feature selection
- Compreensão do modelo
- Debugging
- Redução de dimensionalidade

---

## Conceitos Fundamentais

### O que é Feature Engineering?

Feature engineering é o processo de usar conhecimento do domínio para criar features (variáveis) que tornam os algoritmos de machine learning mais eficazes. É frequentemente considerado a parte mais importante e demorada de um projeto de ML.

### Tipos de Feature Engineering:

1. **Feature Extraction:**
   - Criar novas features a partir de dados brutos
   - Exemplo: Extrair dia da semana de uma data

2. **Feature Transformation:**
   - Transformar features existentes
   - Exemplo: Logaritmo, normalização, padronização

3. **Feature Selection:**
   - Selecionar as features mais relevantes
   - Exemplo: Remover features com baixa variância

4. **Feature Construction:**
   - Criar features através de combinações
   - Exemplo: Interações entre features

### Pipeline Típico de Feature Engineering:

1. **Análise Exploratória de Dados (EDA)**
   - Entender os dados
   - Identificar padrões
   - Detectar anomalias

2. **Tratamento de Dados Faltantes**
   - Imputação
   - Remoção
   - Criação de flags

3. **Codificação de Variáveis Categóricas**
   - One-Hot Encoding
   - Label Encoding
   - Target Encoding

4. **Transformação de Features**
   - Normalização
   - Padronização
   - Transformações matemáticas

5. **Criação de Novas Features**
   - Features de domínio
   - Agregações
   - Interações

6. **Seleção de Features**
   - Baseada em importância
   - Baseada em correlação
   - Métodos wrapper

7. **Redução de Dimensionalidade**
   - PCA
   - t-SNE
   - UMAP

---

## Melhores Práticas

### 1. Compreenda o Domínio
- Conhecimento do domínio é crucial
- Consulte especialistas
- Pesquise sobre o problema

### 2. Comece Simples
- Baseline com features básicas
- Adicione complexidade gradualmente
- Meça o impacto de cada feature

### 3. Use Pipelines
- scikit-learn Pipelines
- Evite data leakage
- Facilita reprodutibilidade

### 4. Valide Corretamente
- Use cross-validation
- Separe train/validation/test
- Cuidado com temporal leakage

### 5. Documente Tudo
- Registre transformações
- Explique features criadas
- Mantenha código organizado

### 6. Automatize Quando Possível
- Use Featuretools
- Scripts reutilizáveis
- Versionamento de features

---

## Ferramentas Complementares

### scikit-learn Feature Selection
- **URL:** [https://scikit-learn.org/stable/modules/feature_selection.html](https://scikit-learn.org/stable/modules/feature_selection.html)
- **Métodos:**
  - VarianceThreshold
  - SelectKBest
  - SelectPercentile
  - RFE (Recursive Feature Elimination)
  - SelectFromModel

### Category Encoders
- **URL:** [https://contrib.scikit-learn.org/category_encoders/](https://contrib.scikit-learn.org/category_encoders/)
- **Encoders:**
  - One-Hot Encoding
  - Target Encoding
  - Binary Encoding
  - Hash Encoding
  - Leave-One-Out Encoding

---

## Estatísticas

- **Total de Recursos:** 6
- **Bibliotecas:** 3
- **Técnicas:** 3
- **Métodos de Redução:** 3

---

## Referências

1. scikit-learn Documentation
2. Feature-engine GitHub
3. Featuretools Documentation
4. SHAP GitHub Repository
5. Machine Learning Mastery
6. Medium Articles on Dimensionality Reduction

---

**Última Atualização:** 30 de Outubro de 2025  
**Mantido por:** Manus AI
