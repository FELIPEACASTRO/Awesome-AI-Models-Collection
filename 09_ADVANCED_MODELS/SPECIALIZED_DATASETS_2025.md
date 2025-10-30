# Datasets Especializados por Domínio - 2025

**Data de Atualização:** 30 de Outubro de 2025  
**Fonte:** Busca intensiva com processamento paralelo

Este documento contém uma coleção abrangente de datasets especializados organizados por domínio de aplicação, incluindo datasets de alta qualidade para pesquisa e desenvolvimento em IA.

---

## 📋 Índice

- [Neuroimagem e Neurociência](#neuroimagem-e-neurociência)
- [Finanças e Mercado de Ações](#finanças-e-mercado-de-ações)
- [Direção Autônoma](#direção-autônoma)
- [Processamento de Linguagem Natural](#processamento-de-linguagem-natural)
- [Visão Computacional](#visão-computacional)
- [Reconhecimento de Fala](#reconhecimento-de-fala)
- [Sistemas de Recomendação](#sistemas-de-recomendação)
- [Análise de Sentimentos](#análise-de-sentimentos)
- [Detecção de Objetos](#detecção-de-objetos)
- [Compreensão de Vídeo](#compreensão-de-vídeo)
- [Nuvens de Pontos 3D](#nuvens-de-pontos-3d)
- [NLP Biomédico](#nlp-biomédico)
- [Clima e Meteorologia](#clima-e-meteorologia)
- [Robótica](#robótica)
- [Redes Neurais de Grafos](#redes-neurais-de-grafos)

---

## Neuroimagem e Neurociência

### OpenNeuro
- **URL:** [https://openneuro.org/](https://openneuro.org/)
- **Domínio:** Neuroimagem/Neurociência
- **Descrição:** Plataforma para dados de neuroimagem, hospedando mais de 1.240 datasets públicos com dados de mais de 51.000 participantes. Suporta múltiplas modalidades de imagem, incluindo MRI, PET, MEG, EEG e iEEG.
- **Estatísticas:**
  - 1.240+ datasets
  - 51.000+ participantes
  - Múltiplas modalidades (MRI, PET, MEG, EEG, iEEG)
- **Aplicações:**
  - Pesquisa em neurociência
  - Diagnóstico médico
  - Análise de imagens cerebrais

---

## Finanças e Mercado de Ações

### Stock Market Dataset (Kaggle)
- **URL:** [https://www.kaggle.com/datasets/jacksoncrow/stock-market-dataset](https://www.kaggle.com/datasets/jacksoncrow/stock-market-dataset)
- **Domínio:** Séries Temporais Financeiras, Mercado de Ações
- **Tamanho:** 2.75 GB
- **Descrição:** Contém preços históricos diários de ações e ETFs negociados na NASDAQ, recuperados do Yahoo Finance via yfinance. Inclui dados de Abertura, Máxima, Mínima, Fechamento, Fechamento Ajustado e Volume. O dataset tem aproximadamente 2.75 GB e contém dados até 01 de abril de 2020. É atualizado trimestralmente.
- **Características:**
  - Preços históricos diários
  - Ações e ETFs da NASDAQ
  - Dados OHLCV (Open, High, Low, Close, Volume)
  - Atualização trimestral
- **Aplicações:**
  - Previsão de preços
  - Análise técnica
  - Trading algorítmico
  - Análise de risco

---

## Direção Autônoma

### The KITTI Vision Benchmark Suite
- **URL:** [https://www.cvlibs.net/datasets/kitti/](https://www.cvlibs.net/datasets/kitti/)
- **Domínio:** Direção Autônoma / Visão Computacional
- **Descrição:** Conjunto de dados de visão de referência para pesquisa em direção autônoma, capturados em Karlsruhe, áreas rurais e rodovias. Inclui tarefas como estéreo, fluxo óptico, odometria visual, detecção e rastreamento de objetos 3D e 2D. É um dos datasets pioneiros e mais citados na área.
- **Tarefas Suportadas:**
  - Estéreo
  - Fluxo óptico
  - Odometria visual
  - Detecção de objetos 3D/2D
  - Rastreamento de objetos
- **Aplicações:**
  - Veículos autônomos
  - SLAM
  - Percepção 3D

### nuScenes
- **URL:** [https://www.nuscenes.org/](https://www.nuscenes.org/)
- **Domínio:** Direção Autônoma
- **Descrição:** Dataset multimodal de larga escala para veículos autônomos, contendo 1000 cenas de 20 segundos cada, com anotações 3D completas.
- **Características:**
  - 1000 cenas
  - Anotações 3D completas
  - Múltiplos sensores (câmeras, LIDAR, radar)
- **Aplicações:**
  - Detecção 3D
  - Tracking
  - Previsão de trajetória

---

## Processamento de Linguagem Natural

### WikiNER
- **URL:** [https://huggingface.co/datasets/mnaguib/WikiNER](https://huggingface.co/datasets/mnaguib/WikiNER)
- **Domínio:** Processamento de Linguagem Natural (NER) Multilíngue
- **Descrição:** Contém 7.200 artigos da Wikipedia rotulados manualmente em nove idiomas (Inglês, Alemão, Francês, Polonês, Italiano, Espanhol, Holandês, Português e Russo) para Reconhecimento de Entidades Nomeadas (NER).
- **Idiomas:** 9 (EN, DE, FR, PL, IT, ES, NL, PT, RU)
- **Artigos:** 7.200
- **Aplicações:**
  - Named Entity Recognition
  - NLP multilíngue
  - Extração de informação

---

## Visão Computacional

### ImageNet
- **URL:** [https://www.image-net.org/](https://www.image-net.org/)
- **Domínio:** Visão Computacional - Classificação de Imagens
- **Descrição:** Um grande banco de dados de imagens organizado de acordo com a hierarquia WordNet, com mais de 14 milhões de imagens e 21.841 categorias (synsets). É fundamental para o avanço da visão computacional e pesquisa em aprendizado profundo, sendo usado principalmente para classificação de imagens.
- **Estatísticas:**
  - 14+ milhões de imagens
  - 21.841 categorias
  - Hierarquia WordNet
- **Aplicações:**
  - Classificação de imagens
  - Transfer learning
  - Benchmarking de modelos

### COCO (Common Objects in Context)
- **URL:** [https://cocodataset.org/](https://cocodataset.org/)
- **Domínio:** Visão Computacional
- **Descrição:** Dataset de larga escala para detecção de objetos, segmentação e captioning.
- **Características:**
  - 330k imagens
  - 80 categorias de objetos
  - Anotações de segmentação
- **Aplicações:**
  - Detecção de objetos
  - Segmentação de instâncias
  - Image captioning

---

## Reconhecimento de Fala

### LibriSpeech ASR corpus
- **URL:** [https://www.openslr.org/12/](https://www.openslr.org/12/)
- **Domínio:** Reconhecimento Automático de Fala (ASR)
- **Descrição:** Corpus de aproximadamente 1000 horas de fala em inglês lida, com taxa de amostragem de 16kHz. Projetado para treinamento e teste de sistemas de Reconhecimento Automático de Fala (ASR).
- **Estatísticas:**
  - 1000 horas de áudio
  - Taxa de amostragem: 16kHz
  - Inglês lido
- **Aplicações:**
  - Treinamento de modelos ASR
  - Benchmarking de sistemas de fala
  - Transfer learning para fala

---

## Sistemas de Recomendação

### MovieLens
- **URL:** [https://grouplens.org/datasets/movielens/](https://grouplens.org/datasets/movielens/)
- **Domínio:** Sistemas de Recomendação
- **Descrição:** Coleção de datasets de avaliações de filmes, amplamente utilizada para pesquisa em sistemas de recomendação.
- **Versões:**
  - MovieLens 100K
  - MovieLens 1M
  - MovieLens 10M
  - MovieLens 25M
- **Aplicações:**
  - Filtragem colaborativa
  - Sistemas de recomendação
  - Análise de preferências

---

## Análise de Sentimentos

### Twitter Sentiment Analysis Dataset
- **URL:** [https://www.kaggle.com/datasets/kazanova/sentiment140](https://www.kaggle.com/datasets/kazanova/sentiment140)
- **Domínio:** Análise de Sentimentos
- **Descrição:** Dataset contendo 1.6 milhões de tweets rotulados para análise de sentimentos.
- **Características:**
  - 1.6M tweets
  - Rotulação binária (positivo/negativo)
  - Dados reais de redes sociais
- **Aplicações:**
  - Análise de sentimentos
  - Monitoramento de marca
  - Análise de opinião pública

---

## Detecção de Objetos

### YOLO Datasets
- **URL:** [https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data](https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data)
- **Domínio:** Detecção de Objetos
- **Descrição:** Coleção de datasets formatados para treinamento de modelos YOLO, incluindo exemplos e tutoriais.
- **Aplicações:**
  - Detecção de objetos em tempo real
  - Treinamento de modelos YOLO
  - Visão computacional aplicada

---

## Compreensão de Vídeo

### Kinetics
- **URL:** [https://www.deepmind.com/open-source/kinetics](https://www.deepmind.com/open-source/kinetics)
- **Domínio:** Compreensão de Vídeo
- **Desenvolvedor:** DeepMind
- **Descrição:** Dataset de larga escala para reconhecimento de ações em vídeos, contendo milhões de clipes de vídeo.
- **Versões:**
  - Kinetics-400
  - Kinetics-600
  - Kinetics-700
- **Aplicações:**
  - Reconhecimento de ações
  - Compreensão de vídeo
  - Análise temporal

---

## Nuvens de Pontos 3D

### ShapeNet
- **URL:** [https://shapenet.org/](https://shapenet.org/)
- **Domínio:** Visão Computacional 3D
- **Descrição:** Repositório de modelos 3D anotados, contendo milhões de formas 3D organizadas por categoria.
- **Características:**
  - Milhões de modelos 3D
  - Anotações semânticas
  - Múltiplas categorias
- **Aplicações:**
  - Reconstrução 3D
  - Síntese de formas
  - Compreensão de geometria

---

## NLP Biomédico

### PubMed Datasets
- **URL:** [https://pubmed.ncbi.nlm.nih.gov/](https://pubmed.ncbi.nlm.nih.gov/)
- **Domínio:** NLP Biomédico
- **Descrição:** Coleção de abstracts e artigos científicos da área biomédica, amplamente utilizada para NLP médico.
- **Características:**
  - Milhões de abstracts
  - Terminologia médica
  - Atualizações constantes
- **Aplicações:**
  - NLP médico
  - Extração de informação biomédica
  - Pesquisa científica

---

## Clima e Meteorologia

### Climate Data Store (CDS)
- **URL:** [https://cds.climate.copernicus.eu/](https://cds.climate.copernicus.eu/)
- **Domínio:** Clima e Meteorologia
- **Desenvolvedor:** Copernicus
- **Descrição:** Repositório de dados climáticos e meteorológicos de alta qualidade, incluindo dados históricos e previsões.
- **Aplicações:**
  - Previsão do tempo
  - Análise climática
  - Modelagem ambiental

---

## Robótica

### RoboNet
- **URL:** [https://www.robonet.wiki/](https://www.robonet.wiki/)
- **Domínio:** Robótica
- **Descrição:** Dataset de larga escala para aprendizado de robótica, contendo vídeos de manipulação robótica de múltiplos robôs.
- **Características:**
  - Múltiplos robôs
  - Tarefas de manipulação
  - Dados visuais
- **Aplicações:**
  - Aprendizado de robótica
  - Manipulação de objetos
  - Transfer learning em robótica

---

## Redes Neurais de Grafos

### Open Graph Benchmark (OGB)
- **URL:** [https://ogb.stanford.edu/](https://ogb.stanford.edu/)
- **Domínio:** Graph Neural Networks
- **Desenvolvedor:** Stanford
- **Descrição:** Coleção de benchmarks realistas e diversos para Graph Neural Networks, incluindo datasets de grafos de larga escala.
- **Características:**
  - Múltiplos domínios
  - Grafos de larga escala
  - Benchmarks padronizados
- **Aplicações:**
  - Graph Neural Networks
  - Análise de redes sociais
  - Descoberta de medicamentos

---

## Estatísticas Gerais

- **Total de Datasets:** 15+
- **Domínios Cobertos:** 15
- **Datasets de Larga Escala:** 10+
- **Datasets Multimodais:** 3
- **Datasets Multilíngues:** 2

---

## Tabela Resumo por Domínio

| Domínio | Dataset Principal | Tamanho | Aplicação |
|---|---|---|---|
| Neuroimagem | OpenNeuro | 1.240+ datasets | Pesquisa cerebral |
| Finanças | Stock Market (Kaggle) | 2.75 GB | Trading algorítmico |
| Direção Autônoma | KITTI | - | Veículos autônomos |
| NLP | WikiNER | 7.200 artigos | NER multilíngue |
| Visão | ImageNet | 14M+ imagens | Classificação |
| Fala | LibriSpeech | 1000 horas | ASR |
| Recomendação | MovieLens | 25M avaliações | Sistemas de recomendação |
| Sentimentos | Twitter Sentiment | 1.6M tweets | Análise de sentimentos |
| Vídeo | Kinetics | Milhões de clipes | Reconhecimento de ações |
| 3D | ShapeNet | Milhões de modelos | Reconstrução 3D |

---

## Referências

1. OpenNeuro Platform
2. Kaggle Datasets
3. KITTI Vision Benchmark
4. Hugging Face Datasets
5. ImageNet Project
6. OpenSLR
7. GroupLens Research
8. DeepMind Research
9. Stanford OGB

---

**Última Atualização:** 30 de Outubro de 2025  
**Mantido por:** Manus AI
