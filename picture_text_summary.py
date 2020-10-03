from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import fastcluster
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster import hierarchy
from hac_tools import HAC

def cluster_summary_simple(clust_txt, model, clust_emb = [], top_n=1, text_if_empty='all'):
    if clust_txt == []:
        return text_if_empty
    else:
        df=pd.DataFrame()
        df['titles']=clust_txt
        if len(clust_emb)==0:
            clust_emb=model.encode(clust_txt, batch_size=16, 
                                        show_progress_bar=False, convert_to_numpy=True)
        clust_avg=np.mean(clust_emb, axis=0, keepdims=True)
        df['cluster_rank']=0.5*(1 + cosine_similarity(clust_emb,clust_avg))
        df1=df.sort_values('cluster_rank',ascending=False).copy()
        if top_n==1:
            return df1.iloc[0].titles
        else:
            return list(df1.head(top_n).titles.values)


def unroll_tree_map(X, method='single', depth=3, nr_splits=3,min_size=0.1,max_extension=1):
    go = True
    #clust_nm = 'all'
    clust_idx = -1
    len_x = len(X)
    res_d={clust_idx:
        {
        'cluster_id': clust_idx,
        'cluster_parent': 'all',
        'cluster_members': [],
        'cluster_table': {},
        'cluster_size': len(X),
        }
          }
    df_res = pd.DataFrame(res_d).T
    #col_nm = ['value', 'color', 'parent_idx', 'id_idx','parent_nm', \
    #         'id_nm', 'members_ids', 'members_txt', 'members_embeddings']

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
    
    df_res['color']=df_res['cluster_size']/len_x
    col_nm={'cluster_size':'value','cluster_id':'id','cluster_parent':'parent'}
    df_res=df_res.rename(columns=col_nm)
    return df_res