# picture_text
Hierarchical Clustering (HAC) with tree maps on text

<p align="center">
  <img src="assets/tree_map1.gif" width=1000>
</p>

## Customization

### Customizing treemaps

```python
pt = PictureText(txt)
pt(hac_method='ward', hac_metric='euclidean')
pt.make_picture(layer_depth = 6,
                layer_min_size = 0.1,
                layer_max_extension = 1,
                treemap_average_score = None, 
                treemap_maxdepth=3,)
```
### Customizing clustering methods

```python
pt = PictureText(txt)
pt(hac_method='ward', hac_metric='euclidean')
```

## BYO-NLP
The key features to this sort of approach are the embeddings as well as the method of multi-doc summarization. You can use your NLP tools of choice there.
#### Text embeddings
The default set of text embeddings is via [SBERT](https://www.sbert.net)'s `distilbert-base-nli-stsb-mean-tokens`
```python
pt = PictureText(txt)
pt(hac_method='ward', hac_metric='euclidean')
```

#### Summarizer
```python
def silly_summarizer(txt,txt_embeddings):
   return "All the same to me", 0
pt.make_picture(summarizer = silly_summarizer,)
```
<p align="center">
  <img src="assets/tree_map1.gif" width=1000>
</p>
