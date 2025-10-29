# 🚀 Awesome AI Models Collection

Uma coleção curada de **110+ repositórios GitHub** com modelos de IA avançados, organizados por categoria para facilitar a descoberta e o aprendizado.

## 📚 Índice

- [Sobre](#sobre)
- [Categorias](#categorias)
- [Estatísticas](#estatísticas)
- [Como Usar](#como-usar)
- [Contribuindo](#contribuindo)
- [Licença](#licença)

## 🎯 Sobre

Esta coleção foi criada para reunir os melhores e mais avançados repositórios de modelos de IA disponíveis no GitHub e Hugging Face. Cada repositório foi cuidadosamente selecionado com base em critérios como número de downloads, qualidade do código, documentação e relevância para a comunidade de IA.

## 📊 Estatísticas

- **Total de Repositórios:** 110+
- **Categorias:** 11
- **Modelos do Hugging Face:** 30+
- **Repositórios GitHub:** 80+
- **Downloads Totais (HF):** 500M+

## 🗂️ Categorias

### 1. [Biomassa e Agricultura](./categories/biomass-agriculture.md) (10 repos)
Modelos e datasets especializados em agricultura, biomassa, imagens aéreas e análise de culturas.

**Destaques:**
- Agriculture-Vision (21K RGB+NIR images)
- AgriNet (160K agricultural images)
- AGBD Dataset (300GB Sentinel-2)

### 2. [NLP e Large Language Models](./categories/nlp-llms.md) (23 repos)
Modelos de processamento de linguagem natural, transformers, BERT, GPT e LLMs.

**Destaques:**
- Hugging Face Transformers (framework principal)
- BERT, RoBERTa, GPT-2
- Qwen 2.5, LLMs from Scratch

### 3. [Visão Computacional](./categories/computer-vision.md) (20 repos)
Modelos de classificação de imagens, detecção de objetos, segmentação e YOLO.

**Destaques:**
- CLIP (OpenAI)
- Segment Anything Model 2 (SAM 2)
- YOLOv10, Ultralytics

### 4. [Áudio e Fala](./categories/audio-speech.md) (11 repos)
Reconhecimento de fala, síntese de voz, análise de áudio e modelos de linguagem de áudio.

**Destaques:**
- OpenAI Whisper
- Pyannote Audio (speaker diarization)
- FunASR

### 5. [GANs e Generative AI](./categories/gans-generative.md) (10 repos)
Redes adversariais generativas, modelos de difusão e geração de imagens.

**Destaques:**
- Stable Diffusion
- Hugging Face Diffusers
- PyTorch-GAN

### 6. [Reinforcement Learning](./categories/reinforcement-learning.md) (7 repos)
Algoritmos de aprendizado por reforço, deep RL e implementações práticas.

**Destaques:**
- Deep RL Algorithms (PyTorch)
- Offline RL
- Udacity Deep RL Nanodegree

### 7. [AI Agents](./categories/ai-agents.md) (3 repos)
Agentes autônomos de IA, sistemas multi-agente e frameworks.

**Destaques:**
- SuperAGI
- Awesome AI Agents
- 500 AI Agents Projects

### 8. [PyTorch](./categories/pytorch.md) (6 repos)
Framework PyTorch, exemplos, tutoriais e implementações.

**Destaques:**
- PyTorch (framework oficial)
- PyTorch Examples
- Awesome PyTorch List

### 9. [TensorFlow](./categories/tensorflow.md) (4 repos)
Framework TensorFlow, Keras, cursos e implementações.

**Destaques:**
- TensorFlow (framework oficial)
- TensorFlow Deep Learning Course
- DeepLearning.AI Certificate

### 10. [ML Frameworks](./categories/ml-frameworks.md) (3 repos)
Outros frameworks de machine learning e deep learning.

**Destaques:**
- Awesome Machine Learning
- NeoML
- Neural Fortran

### 11. [Multimodal AI](./categories/multimodal.md) (2 repos)
Modelos multimodais que trabalham com múltiplas modalidades (texto, imagem, áudio).

**Destaques:**
- Awesome Unified Multimodal Models
- Multimodal LLMs

### 12. [Modelos Especializados](./categories/specialized.md) (6 repos)
Modelos especializados para domínios específicos (proteínas, vídeo, etc).

**Destaques:**
- ESM2 (Protein Language Model)
- Tarsier2 (Video LLM)
- ELECTRA

### 13. [Coleções e Recursos](./categories/collections.md) (5 repos)
Listas curadas, coleções de projetos e recursos educacionais.

**Destaques:**
- 500 AI Projects with Code
- Top Deep Learning Repos
- Machine Learning Repos

## 🚀 Como Usar

### Navegação Rápida

1. **Por Categoria:** Acesse os arquivos em `./categories/` para ver repositórios organizados por tema
2. **Por Popularidade:** Veja os modelos mais baixados do Hugging Face
3. **Por Framework:** Filtre por PyTorch, TensorFlow, etc.

### Clonando Repositórios

Para clonar um repositório específico:

```bash
# Exemplo: Clonar o Hugging Face Transformers
git clone https://github.com/huggingface/transformers.git
```

### Usando Modelos do Hugging Face

```python
from transformers import pipeline

# Exemplo: Usar BERT para análise de sentimento
classifier = pipeline("sentiment-analysis")
result = classifier("I love this collection!")
print(result)
```

## 📈 Repositórios Mais Populares

### Top 10 por Downloads (Hugging Face)

1. **sentence-transformers/all-MiniLM-L6-v2** - 134.8M downloads
2. **Falconsai/nsfw_image_detection** - 97.8M downloads
3. **google/electra-base-discriminator** - 75.7M downloads
4. **google-bert/bert-base-uncased** - 54.3M downloads
5. **dima806/fairface_age_image_detection** - 51.9M downloads
6. **FacebookAI/roberta-large** - 21.6M downloads
7. **openai/clip-vit-base-patch32** - 20.6M downloads
8. **timm/mobilenetv3_small_100** - 18.5M downloads
9. **laion/clap-htsat-fused** - 18.5M downloads
10. **sentence-transformers/all-mpnet-base-v2** - 17.6M downloads

### Top 10 por Stars (GitHub)

1. **tensorflow/tensorflow** - Framework de ML
2. **pytorch/pytorch** - Framework de DL
3. **huggingface/transformers** - Modelos transformer
4. **openai/whisper** - Speech recognition
5. **facebookresearch/segment-anything** - SAM 2
6. **ultralytics/ultralytics** - YOLO
7. **AUTOMATIC1111/stable-diffusion-webui** - SD Web UI
8. **CompVis/stable-diffusion** - Stable Diffusion
9. **google-research/bert** - BERT original
10. **Stability-AI/stablediffusion** - SD oficial

## 🎓 Recursos de Aprendizado

### Cursos e Tutoriais

- **TensorFlow Deep Learning** - Curso completo de DL com TensorFlow
- **Hands-On LLMs** - Livro prático sobre Large Language Models
- **Deep RL Nanodegree** - Programa completo de Reinforcement Learning
- **DeepLearning.AI TensorFlow Certificate** - Certificação profissional

### Papers e Pesquisas

- **AgriNet** - Agriculture-Specific Pretrained Models
- **Crops and Weeds** - Largest crop/weed dataset
- **Novel Features UAV** - Graph-based feature extraction
- **Temporal Features Satellite** - LAI, GPP, NPP Time-Series

## 🤝 Contribuindo

Contribuições são bem-vindas! Se você conhece um repositório incrível de IA que deveria estar nesta lista:

1. Fork este repositório
2. Adicione o repositório na categoria apropriada
3. Atualize as estatísticas
4. Envie um Pull Request

### Critérios de Inclusão

- ✅ Código open source
- ✅ Documentação clara
- ✅ Ativamente mantido
- ✅ Relevante para a comunidade de IA
- ✅ Qualidade comprovada (stars, downloads, papers)

## 📝 Licença

Esta coleção é disponibilizada sob a licença MIT. Os repositórios individuais têm suas próprias licenças - consulte cada repositório para detalhes.

## 🌟 Agradecimentos

Agradecimentos especiais a todos os desenvolvedores e pesquisadores que criaram e mantêm esses projetos incríveis!

## 📞 Contato

- **GitHub:** [@FELIPEACASTRO](https://github.com/FELIPEACASTRO)
- **Criado em:** Outubro 2025
- **Última Atualização:** Outubro 2025

---

⭐ Se esta coleção foi útil para você, considere dar uma estrela!

🔗 Compartilhe com outros entusiastas de IA!

💡 Sugestões? Abra uma issue!
