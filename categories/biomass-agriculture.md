# 🌾 Biomassa e Agricultura

Modelos e datasets especializados em agricultura, biomassa, imagens aéreas e análise de culturas.

## 📊 Estatísticas

- **Total de Repositórios:** 10
- **Datasets:** 3 grandes datasets (300GB+ total)
- **Papers:** 5 papers de pesquisa
- **Plataformas:** 2 plataformas de dados

## 🔥 Destaques Principais

### 1. Agriculture-Vision Dataset ⭐⭐⭐⭐⭐

**Repositório:** https://github.com/SHI-Labs/Agriculture-Vision

**Descrição:** Dataset oficial das competições CVPR 2020-2023 com 21,061 imagens RGB+NIR aéreas para segmentação semântica em agricultura.

**Características:**
- 21,061 imagens RGB + Near-Infrared
- Imagens aéreas de alta resolução
- Semantic segmentation annotations
- Multi-modal (RGB + NIR)
- Dataset oficial CVPR

**Aplicações:**
- Detecção de doenças em culturas
- Monitoramento de crescimento
- Análise de biomassa
- Segmentação de campos

**Como Usar:**
```python
# Clonar o repositório
git clone https://github.com/SHI-Labs/Agriculture-Vision.git

# Baixar o dataset
# Seguir instruções no README do repositório
```

---

### 2. AGBD HuggingFace Dataset ⭐⭐⭐⭐⭐

**Dataset:** https://huggingface.co/datasets/prs-eth/AGBD

**Descrição:** Dataset massivo de 300GB combinando imagens Sentinel-2 com dados de biomassa GEDI L4A, com cobertura global.

**Características:**
- 300GB de dados
- Sentinel-2 satellite imagery
- GEDI L4A biomass labels
- Cobertura global
- Streamable via HuggingFace

**Aplicações:**
- Estimativa de biomassa global
- Monitoramento florestal
- Análise de carbono
- Pré-treino de modelos

**Como Usar:**
```python
from datasets import load_dataset

# Carregar dataset streamable (não precisa baixar tudo)
dataset = load_dataset("prs-eth/AGBD", streaming=True)

# Iterar sobre batches
for batch in dataset['train'].iter(batch_size=32):
    images = batch['image']
    biomass = batch['agbd']
    # Treinar modelo
```

---

### 3. Satellite Image Deep Learning Datasets ⭐⭐⭐⭐⭐

**Repositório:** https://github.com/satellite-image-deep-learning/datasets

**Descrição:** Coleção curada de 100+ datasets de imagens de satélite para deep learning, organizados por domínio.

**Características:**
- 100+ datasets curados
- Multi-domain (Agriculture, Forestry, Urban, etc.)
- Descrições detalhadas
- Links diretos para download
- Constantemente atualizado

**Categorias:**
- Agriculture & Vegetation
- Forestry & Biomass
- Urban Planning
- Disaster Response
- Climate Change

**Como Usar:**
```bash
# Clonar o repositório
git clone https://github.com/satellite-image-deep-learning/datasets.git

# Explorar a lista de datasets
cd datasets
# Seguir links para datasets específicos
```

---

## 📚 Papers de Pesquisa

### 4. AgriNet - Agriculture-Specific Pretrained Models

**Paper:** https://arxiv.org/abs/2207.03881

**Resumo:** Modelos pré-treinados especificamente para agricultura, superando ImageNet em tarefas agrícolas.

**Contribuições:**
- 160K agricultural images de 19 localizações
- 423 classes de plantas e doenças
- 5 modelos pré-treinados (VGG16, VGG19, Inception-v3, InceptionResNet-v2, Xception)
- AgriNet-VGG19: 94% accuracy, 92% F1-score

**Impacto:**
- Domain-specific pretraining para agricultura
- Melhor performance que ImageNet genérico
- Transfer learning eficiente

**Citação:**
```bibtex
@article{agrinet2022,
  title={AgriNet: Deep Learning Approach for Agriculture-Specific Pretrained Models},
  journal={arXiv preprint arXiv:2207.03881},
  year={2022}
}
```

---

### 5. Crops and Weeds Dataset

**Paper:** https://arxiv.org/abs/2108.05789

**Resumo:** O maior dataset de culturas e ervas daninhas, com 1.2M imagens indoor e 540K imagens de campo.

**Contribuições:**
- 1.2M indoor images
- 540K field images
- Multi-species coverage
- Largest crop/weed dataset

**Aplicações:**
- Detecção de ervas daninhas
- Classificação de culturas
- Agricultura de precisão
- Pré-treino de modelos

**Citação:**
```bibtex
@article{cropsweeds2021,
  title={Crops and Weeds: A Large-Scale Dataset},
  journal={arXiv preprint arXiv:2108.05789},
  year={2021}
}
```

---

### 6. Novel Features UAV - GFKuts & Graph Fusion

**Paper:** https://pmc.ncbi.nlm.nih.gov/articles/PMC8271736/

**Resumo:** Métodos avançados de extração de features usando grafos e fusão multi-fonte para imagens UAV.

**Contribuições:**
- GFKuts (Graph-based Feature Extraction)
- Graph Fusion para combinar múltiplas fontes
- NDVI e features espectrais
- Análise de imagens UAV

**Técnicas:**
- Segmentação em superpixels
- Construção de grafos de adjacência
- Graph convolution
- Multi-source fusion

**Implementação Exemplo:**
```python
def extract_gfkuts(image, ndvi):
    """Graph-based feature extraction"""
    # 1. Segmentar em superpixels
    segments = slic(image, n_segments=100)
    
    # 2. Construir grafo
    graph = build_adjacency_graph(segments)
    
    # 3. Extrair features
    node_features = extract_node_features(segments, ndvi)
    
    # 4. Graph convolution
    gfkuts_features = graph_conv(graph, node_features)
    
    return gfkuts_features
```

---

### 7. Temporal Features Satellite - LAI, GPP, NPP

**Paper:** https://www.tandfonline.com/doi/full/10.1080/10106049.2022.2153930

**Resumo:** Uso de features temporais de satélite (LAI, GPP, NPP) para estimativa de biomassa.

**Features:**
- **LAI** (Leaf Area Index) - Índice de área foliar
- **GPP** (Gross Primary Productivity) - Produtividade primária bruta
- **NPP** (Net Primary Productivity) - Produtividade primária líquida
- Time-series analysis

**Aplicações:**
- Estimativa de biomassa temporal
- Monitoramento de crescimento
- Análise de produtividade
- Previsão de safras

**Estimação de LAI:**
```python
def estimate_lai_from_ndvi(ndvi):
    """Estimar LAI usando NDVI"""
    # Equação empírica
    lai = 3.618 * ndvi - 0.118
    return lai
```

---

### 8. Feature Selection RFFS

**Paper:** https://www.sciencedirect.com/science/article/abs/pii/S2352938522001768

**Resumo:** Método de seleção de features RFFS (Recursive Feature Forward Selection) para análise de biomassa.

**Contribuições:**
- RFFS algorithm
- Melhor que Boruta em alguns casos
- Feature importance ranking
- Aplicação em biomassa

**Implementação:**
```python
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestRegressor

# RFFS
rf = RandomForestRegressor(n_estimators=100)
rffs = RFE(estimator=rf, n_features_to_select=10, step=1)

# Fit
rffs.fit(X_train, y_train)

# Select features
X_selected = X_train[:, rffs.support_]
```

---

## 🌐 Plataformas de Dados

### 9. SmartFarmingLab Field Dataset Survey

**Website:** https://smartfarminglab.github.io/field_dataset_survey/

**Descrição:** Survey de 45 datasets curados para agricultura de campo (não satélite).

**Características:**
- 45 curated datasets
- Field images (ground-level)
- Survey paper com comparações
- Metadata detalhado

**Categorias:**
- Crop classification
- Disease detection
- Weed detection
- Yield prediction

---

### 10. Roboflow Universe

**Plataforma:** https://universe.roboflow.com/

**Descrição:** Plataforma com 1000+ datasets públicos, incluindo centenas na categoria agricultura.

**Características:**
- 1000+ public datasets
- Easy-to-use Roboflow API
- Pre-processed datasets
- Annotation tools
- Model training integration

**Categorias de Agricultura:**
- Crop detection
- Pest identification
- Plant disease
- Fruit counting
- Weed segmentation

**Como Usar:**
```python
from roboflow import Roboflow

rf = Roboflow(api_key="YOUR_API_KEY")
project = rf.workspace().project("crop-detection")
dataset = project.version(1).download("yolov8")
```

---

## 📈 Comparação de Datasets

| Dataset | Tamanho | Tipo | Modalidade | Aplicação Principal |
|---------|---------|------|------------|---------------------|
| Agriculture-Vision | 21K images | Aerial | RGB+NIR | Segmentation |
| AGBD | 300GB | Satellite | Multi-spectral | Biomass estimation |
| Crops and Weeds | 1.7M images | Ground+Field | RGB | Classification |
| Roboflow Universe | 1000+ datasets | Varied | Varied | Multi-purpose |

## 🎯 Casos de Uso

### 1. Estimativa de Biomassa
- AGBD Dataset para pré-treino
- Temporal Features (LAI, GPP, NPP)
- GFKuts para feature extraction

### 2. Detecção de Doenças
- AgriNet pretrained models
- Agriculture-Vision para transfer learning
- Crops and Weeds para fine-tuning

### 3. Agricultura de Precisão
- Roboflow datasets específicos
- SmartFarmingLab field datasets
- RFFS para feature selection

### 4. Monitoramento de Culturas
- Satellite datasets
- Temporal analysis
- Multi-spectral imaging

## 🚀 Começando

### Passo 1: Escolher Dataset
```bash
# Para biomassa: AGBD
# Para doenças: Agriculture-Vision
# Para classificação: Crops and Weeds
```

### Passo 2: Pré-processar
```python
# Normalização
# Augmentation
# Feature extraction
```

### Passo 3: Treinar Modelo
```python
# Transfer learning com AgriNet
# Fine-tuning
# Ensemble
```

### Passo 4: Avaliar
```python
# Métricas: R², RMSE, MAE
# Validação cruzada
# Test set evaluation
```

## 📚 Recursos Adicionais

- [Awesome Agriculture](https://github.com/brycejohnston/awesome-agriculture)
- [Awesome Remote Sensing](https://github.com/robmarkcole/satellite-image-deep-learning)
- [Papers with Code - Agriculture](https://paperswithcode.com/task/agriculture)

## 🤝 Contribuindo

Conhece um dataset ou paper incrível de agricultura? Contribua!

---

[⬅️ Voltar para o índice principal](../README.md)
