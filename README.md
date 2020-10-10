# PictureText
PictureText converts a list of short documents to an interactive tree map with minimal code. It defaults to SBERT for text representation, leverages Hierarchical Agglomerative Clustering (HAC) for grouping and tree maps to visualize text interactively.

<p align="center">
  <img src="assets/cover.gif" width=1000>
</p>

Given a corpus of short documents (think news headlines) it can group them into hierarchical groups, that semantically belong together. It also allows the reader to explore each group in more detail by going deeper into a hierarchy and dynamically pulling out of it when needed.

The approach is intended for grouping large sets of non-domain specific short texts. For instance: news headlines, natural language questions and social media posts would be good candidates.

## Demo
Checkout the Colab notebook

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1mTrwk9hYl7bXYUr7e5hbCzv7Bim9ML8Y?usp=sharing)

## Getting started

### Installation
```py
conda create --name pt python=3.6
conda install -n pt nb_conda_kernels
conda activate pt
pip install -r requirements.txt
```

### A simple example
Consider the default values and their result
```py
txt=['The cat sits outside',
     'A man is playing guitar',
     'I love pasta',
     'The new movie is awesome',
     'The cat plays in the garden',
     'A woman watches TV',
     'The new movie is so great',
     'Do you like pizza?',
     'Burgers are my favorite',
     'I like chips',
     'I will have french fries with my burger'
       ]
from picture_text_summary import PictureText

# initializing just sets the text corpus
pt = PictureText(txt) 

# Calling the method does the heavy lifting: 
# 1. HAC 
# 2. text embedding 
pt() 

# This step puts it all together:
# 1. converts HAC into a treemap format
# 2. determines a summary for each cluster and 
# 3. draws & return a treemap
pt.make_picture() 
```
<p align="left">
  <img src="assets/default_settings.png" width=500>
</p>

### Outline of approach
<p align="left">
  <img src="assets/solution_steps.png" width=500>
</p>

- Perform any required preprocessing to get to a list of document strings
- Embed / Encode all documents with the method of choice, by default I use [SBERT](https://www.sbert.net)
- Use HAC to get a “linkage” table of hierarchical assignments of each point to the rest of the data. Here I use [fastcluster](http://danifold.net/fastcluster.html), ward linkage by default.
- Convert to layers for treemap. Iteratively create “layers” by selecting a set number of splits to each layer
- Summarize. Generate a summary for each layer. In the default setting, I use the point closest to the average of the cluster. Using the average of the cluster to represent its centroid is used in a number of few-shot, unsupervised settings
- Draw treemap. Use [plotly](https://plotly.com/)'s [treemap](https://plotly.com/python/treemaps/) for interactive visualization

## Customization

Consider the default values and their result
```python
from picture_text_summary import PictureText
pt = PictureText(txt)
pt(hac_method='ward', hac_metric='euclidean')
pt.make_picture(layer_depth = 6,
                layer_min_size = 0.1,
                layer_max_extension = 1,
                treemap_average_score = None, 
                treemap_maxdepth=3,)
```
<p align="left">
  <img src="assets/default_settings.png" width=500>
</p>

### Selecting Layer Settings
Changing `layer_depth` parameter sets the number of layers produced by the split.
```python
pt.make_picture(layer_depth = 1)
```
<p align="left">
  <img src="assets/layer_depth1.png" width=500>
</p>

Changing `layer_min_size` parameters determines what is the minimal acceptable size of a new cluster for each layer. By default `layer_min_size` is 0.1 (or 10%) meaning if a layer has a cluster smaller than 10% we will try to find another cluster to add to the layer hoping that the next one will be bigger. We will do so up to increasing the relative number of additional clusters up to 1 (or 100%, `layer_max_extension` = 1). Increasing both of these significantly basically means that we get a lot more clusters a lot earlier.

```python
pt.make_picture(layer_depth = 1,
                layer_min_size = 0.9,
                layer_max_extension = 3,
                )
```
<p align="left">
  <img src="assets/min_size.png" width=500>
</p>

### Selecting Clustering Settings

The defaults are the following

```python
pt = PictureText(txt)
pt(hac_method='ward', hac_metric='euclidean')
```

However, those get fed directly into fastcluster, hence all choices from the fastcluster documentation are available here too.

## BYO-NLP
The key features to this sort of approach are the embeddings as well as the method of multi-doc summarization. You can use your NLP tools of choice there.

### Text embeddings
The default set of text embeddings is via [SBERT](https://www.sbert.net)'s `distilbert-base-nli-stsb-mean-tokens`.

```python
from picture_text_summary import sbert_encoder
pt = PictureText(txt)
pt(encoder=sbert_encoder)
```
However, any mapping of a list of text to encoding can be used instead.
```py
def silly_encoder(text_list):
    return [[1]]*len(text_list)

pt(encoder=silly_encoder)
pt.make_picture()
```
<p align="left">
  <img src="assets/silly_encoder.png" width=500>
</p>

### Summarizer
The default sumary method is to take the cluster member closest to the cluster averag. However, any mapping of a list of texts and embeddings into a text summary can be used instead.
```python
def silly_summarizer(txt,txt_embeddings):
   return "All the same to me", 0
pt.make_picture(summarizer = silly_summarizer,)
```
<p align="left">
  <img src="assets/silly_summarizer.png" width=500>
</p>
