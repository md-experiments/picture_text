from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import fastcluster
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster import hierarchy
from hac_tools import HAC

def cluster_summary_simple(clust_txt, model, clust_emb = [], top_n=1, text_if_empty='blank'):
    """
    Returns a summary for a list of documents assuming they belong to the same cluster.
    Takes the document embedding closest to the average cluster embedding as summary. 
    Additionally, averages the similarity of all documents to the cluster average as a measure of quality of the cluster

    Args:
        clust_txt (list): list of sentences to summarize
        model (SBERT model): model used to embed text if clust_emb not provided
        clust_emb (list, optional): list of embeddings of the sentences clust_txt, assume ordering matches, defaults to []
        top_n (int, optional): number of sentences to provide as summary, defaults to 1
        text_if_empty (string): Text to use as summary if empty list of documents provided, defaults to 'blank'

    Returns:
        summary_txt (str or list): summary sentence or sentences 
        centroid_similarity (float): average similarity to the centroid

    >>> cluster_summary_simple(['txt1','txt2'], model = None, clust_emb = [[1,2],[4,5]], top_n=1, text_if_empty='blank')
    ('Txt2', 0.9965696683150038)
    >>> cluster_summary_simple([], model = None, clust_emb = [], top_n=1, text_if_empty='blank')
    ('blank', 1)
    """
    if clust_txt == []:
        summary_txt = text_if_empty
        centroid_similarity = 1
    else:
        df=pd.DataFrame()
        df['titles']=clust_txt
        if len(clust_emb)==0:
            clust_emb=model.encode(clust_txt, batch_size=16, 
                                        show_progress_bar=False, convert_to_numpy=True)
        clust_avg=np.mean(clust_emb, axis=0, keepdims=True)
        df['cluster_rank']=0.5*(1 + cosine_similarity(clust_emb,clust_avg))
        df1=df.sort_values('cluster_rank',ascending=False).copy()
        centroid_similarity = df['cluster_rank'].mean()
        if top_n==1:
            summary_txt = str(df1.iloc[0].titles).title()
        else:
            summary_txt = list(df1.head(top_n).titles.values)
    return summary_txt, centroid_similarity

def unroll_tree_map(X, method='single', depth=3, nr_splits=3,min_size=0.1,max_extension=1):
    """
   Starting from a list of vectors, performs HAC using fastcluster, then splits results into a 
   series of layers with each layer consisting of a roughly equivalent number of slices

    Args:
        X (list): List of all vectors 
        method (string, optional): Method used in HAC, feeds directly into fastcluster, defaults to 'single'
        depth (int, optional): Number of layers to return. This will be the number of drilldowns available in treemap, defaults to 3
        nr_splits (int, optional): Number of clusters to seek to split each layer into, defaults to 3
        min_size (float, optional): Minimal size for a cluster, as a % of total number of observations in X,
            defaults to 0.1 (meaning the smallest cluster should be at least 10% of overall size)
        max_extension (float, optional): Percent extension to nr_splits if min_size not met by all clusters, defaults to 1.0
    Example: 
        - if nr_splits = 3, min_size = 0.1, max_extension=1 
        - max_extension = 1 means up to 100% increase in nr_splits, i.e. up to 6 splits in this case
        - only 1 out of 3 clusters initially are > 10% 
        - Initially this will add 2 more splits (3 - 1) to a total of 5 which is less then the max_extension allowance of 6
        - If again 2 of the 5 are under 10%, this would mean increasing number of splits to 7, however, the max is 6 so we end up with 6

    >>> X=[[x] for x in [1001,1000,1,10,99,100,101]]
    >>> df = unroll_tree_map(X)
    >>> df.drop('cluster_table',axis=1)
        id parent cluster_members value
    9    9     -1       [4, 5, 6]     3
    10  10     -1          [2, 3]     2
    7    7     -1          [0, 1]     2
    5    5      9             [5]     1
    6    6      9             [6]     1
    4    4      9             [4]     1
    2    2     10             [2]     1
    3    3     10             [3]     1
    0    0      7             [0]     1
    1    1      7             [1]     1
    >>> list(df['cluster_table'].values)
    [{4: [4, '', '', 0, 1], 5: [5, '', '', 0, 1], 6: [6, '', '', 0, 1], 8: [8, 5, 6, 1.0, 2], 9: [9, 4, 8, 1.0, 3]}, {2: [2, '', '', 0, 1], 3: [3, '', '', 0, 1], 10: [10, 2, 3, 9.0, 2]}, {0: [0, '', '', 0, 1], 1: [1, '', '', 0, 1], 7: [7, 0, 1, 1.0, 2]}, {5: [5, '', '', 0, 1]}, {6: [6, '', '', 0, 1]}, {4: [4, '', '', 0, 1]}, {2: [2, '', '', 0, 1]}, {3: [3, '', '', 0, 1]}, {0: [0, '', '', 0, 1]}, {1: [1, '', '', 0, 1]}]
    """
    go = True
    #clust_nm = 'all'
    clust_idx = -1
    len_x = len(X)
    df_res = pd.DataFrame([],columns=['cluster_id', 'cluster_parent', 'cluster_members', 'cluster_table', 'cluster_size'])

    z=fastcluster.linkage(X, method=method)
    hac = HAC(z, parent = clust_idx)
    new_clusters = hac.top_n_good_clusters(nr_splits,min_size=min_size,max_extension=max_extension)
    
    df_res=df_res.append(pd.DataFrame(new_clusters).T)
    depth=depth-1
    
    while go:
        new_res = {}
        for c in new_clusters:
            hac = HAC(new_clusters[c]['cluster_table'],parent=c)
            interim_clusters = hac.top_n_good_clusters(nr_splits,min_size=min_size,max_extension=max_extension)
            new_res = {**new_res, **interim_clusters}
        df_res=df_res.append(pd.DataFrame(new_res).T)
        new_clusters=new_res
        depth=depth-1
        if depth<1:
            go = False
    
    col_nm={'cluster_size':'value','cluster_id':'id','cluster_parent':'parent'}
    df_res=df_res.rename(columns=col_nm)
    return df_res