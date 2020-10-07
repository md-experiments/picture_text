from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import fastcluster
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster import hierarchy
from sentence_transformers import SentenceTransformer

from src.hac_tools import HAC
from src.treemap import build_tree_map
from src.utils import TimeClass

def sbert_encoder(text_list, pretrained_reference='distilbert-base-nli-stsb-mean-tokens'):
    """
    Helper function using sentence_transformers which simplifies the embedding call

    Args:
        text_list (list): list of strings to embed
        pretrained_reference (string, optional): the pretrained model to use inside SentenceTransformer, refer to https://www.sbert.net, defaults to 'distilbert-base-nli-stsb-mean-tokens'
    Returns:
        list of embeddings for each string
    """
    model = SentenceTransformer(pretrained_reference)
    text_embeddings = model.encode(text_list, batch_size=16, show_progress_bar=False, convert_to_numpy=True)
    return text_embeddings

class PictureText(object):
    """
    PictureText class for the build of treemaps from hierarchical clustering of text embeddings
    """
    def __init__(self, txt):
        """
        Initialize class

        Args:
            txt (list): List of strings to visualize
        """
        self.txt = txt
        self.txt_embeddings = None
        self.encoder = None
        self.linkage_table = None
        self.hac_method = None
        self.hac_metric = None

    def __call__(self, txt_embeddings=None, encoder=sbert_encoder, hac_method='ward', hac_metric='euclidean'):
        """
        Calls embeddings and generates HAC linkage table. Can either provide embeddings or an encoder.
        Call method can be triggered multiple times with updates to embeddings or linkage when relevant
        
        Args:
            txt_embeddings (list, optional): list of embeddings of the sentences clust_txt, assume ordering matches text, defaults to None
            encoder (object): encoder function used to generate txt_embeddings if none provided, defaults to sbert_encoder
            hac_method (string): HAC method used by fastcluster, defaults to 'ward'
            hac_metric (string): Distrance metric used by fastcluster, defaults to 'euclidean'

        >>> pt = PictureText(['txt','txt','txt','txt','txt','txt','txt'])
        >>> pt([[1], [3], [1], [3], [1], [3], [1]])
        Embeddings updated, external embeddings provided
        Linkage updated, using ward method and euclidean distances, time taken 0 secs
        >>> pt.txt_embeddings
        [[1], [3], [1], [3], [1], [3], [1]]
        >>> pt(encoder = lambda x: [[1]]*len(x))
        Embeddings updated, using encoder, time taken 0 secs
        Linkage updated, using ward method and euclidean distances, time taken 0 secs
        >>> pt.txt_embeddings
        [[1], [1], [1], [1], [1], [1], [1]]
        >>> X=[[x] for x in [1001,1000,1,10,99,100,101]]
        >>> pt(X)
        Embeddings updated, external embeddings provided
        Linkage updated, using ward method and euclidean distances, time taken 0 secs
        >>> pt.txt_embeddings
        [[1001], [1000], [1], [10], [99], [100], [101]]
        >>> pt.linkage_table
        array([[0.00000000e+00, 1.00000000e+00, 1.00000000e+00, 2.00000000e+00],
               [5.00000000e+00, 6.00000000e+00, 1.00000000e+00, 2.00000000e+00],
               [4.00000000e+00, 8.00000000e+00, 1.73205081e+00, 3.00000000e+00],
               [2.00000000e+00, 3.00000000e+00, 9.00000000e+00, 2.00000000e+00],
               [9.00000000e+00, 1.00000000e+01, 1.46398770e+02, 5.00000000e+00],
               [7.00000000e+00, 1.10000000e+01, 1.58601647e+03, 7.00000000e+00]])
        >>> pt.hac_method
        'ward'
        """
        # Set embeddings if those are missing or if we changed the embeddings
        if txt_embeddings:
            self.txt_embeddings = txt_embeddings
            self.linkage_table = None
            assert(len(self.txt_embeddings)==len(self.txt))
            print('Embeddings updated, external embeddings provided')
        # Calculate embeddings if those are missing and the encoder is unchanged do nothing
        elif (encoder == self.encoder):
            pass
        else:
            t = TimeClass()
            self.encoder = encoder
            self.txt_embeddings = self.encoder(self.txt)
            self.linkage_table = None
            secs, _ = t.take()
            assert(len(self.txt_embeddings)==len(self.txt))
            print(f'Embeddings updated, using encoder, time taken {secs} secs')

        # Generate linkage table or update it if parameters for HAC have changed
        if (self.linkage_table==None) or (hac_method!=self.hac_method) or (hac_metric!=self.hac_metric):
            t = TimeClass()
            self.hac_method = hac_method
            self.hac_metric = hac_metric
            self.linkage_table = fastcluster.linkage(self.txt_embeddings, method=hac_method, metric=hac_metric)
            secs, _ = t.take()
            print(f'Linkage updated, using {hac_method} method and {hac_metric} distances, time taken {secs} secs')

    def make_picture(self, 
                summarizer = None,
                layer_depth = 6,
                layer_min_size = 0.1,
                layer_max_extension = 1,
                treemap_average_score = None, 
                treemap_maxdepth=3,
                ):
        """
        Creates the HAC treemap picture of text

        Args:        
            summarizer (object): Summarizer function of the form summary, summary_quality = summarizer(list_text,list_embeddings) and returns a summary (string) and summary_quality (float), defaults None which uses cluster_summary_simple
            Used by hac_to_treemap:
                layer_depth (int, optional): Number of layers to return. This will be the number of drilldowns available in treemap, defaults to 6
                layer_min_size (float, optional): Minimal size for a cluster, as a % of total number of observations in X, defaults to 0.1
                layer_max_extension (float, optional): Percent extension to nr_splits if min_size not met by all clusters, defaults to 1.0
            Used by build_tree_map:
                treemap_average_score (float, optional): Score used as midpoint for plot colors, defaults to None which uses a weighted average of the summary_quality
                treemap_maxdepth (int, optional): Number of levels of hierarchy to show, min 2, defaults to 3
            
        Returns:
            Interactive plotly treemap
        """
        # Set summarizer
        if summarizer:
            self.summarizer = summarizer
        else:
            self.summarizer = self.cluster_summary_simple
        # Convert HAC linkage table into tree map form
        df_res = self.hac_to_treemap(self.linkage_table, depth=layer_depth,min_size=layer_min_size,max_extension=layer_max_extension,)
        # Get summaries for each cluster
        df_res['labels'], df_res['color']= zip(*df_res.apply(lambda x: \
            self.summarizer([np.array(self.txt[m]) for m in x['cluster_members']], \
                                [np.array(self.txt_embeddings[m]) for m in x['cluster_members']]), axis=1))
        # Calculate overall tree map average score
        if treemap_average_score:
            self.average_score = treemap_average_score
        else:
            self.average_score = df_res.apply(lambda x: x['color']*x['value'],axis=1).sum()/df_res.value.sum()
        print(f'Picture weighted average {round(self.average_score,2)}')
        # Draw tree map
        build_tree_map(df_res,maxdepth=treemap_maxdepth,average_score=self.average_score)

    def cluster_summary_simple(self,clust_txt,clust_embeddings,top_n=1, text_if_empty='blank'):
        """
        Returns a summary for a list of documents assuming they belong to the same cluster.
        Takes the document embedding closest to the average cluster embedding as summary. 
        Additionally, averages the similarity of all documents to the cluster average as a measure of quality of the cluster

        Args:
            clust_txt (list): list of sentences to summarize
            clust_embeddings (list): list of embeddings of the sentences clust_txt, assume ordering matches text
            top_n (int, optional): number of sentences to provide as summary, defaults to 1
            text_if_empty (string): Text to use as summary if empty list of documents provided, defaults to 'blank'

        Returns:
            summary_txt (str or list): summary sentence or sentences 
            centroid_similarity (float): average similarity to the centroid
        
        >>> pt = PictureText([])
        >>> pt.cluster_summary_simple(['txt1','txt2'], clust_embeddings = [[1,2],[4,5]])
        ('Txt2', 0.9965696683150038)
        >>> pt.cluster_summary_simple([], clust_embeddings = [], top_n=1, text_if_empty='blank')
        ('blank', 1)
        """
        assert(len(clust_txt)==len(clust_embeddings))

        if clust_txt == []:
            summary_txt = text_if_empty
            centroid_similarity = 1
        else:
            df=pd.DataFrame()
            df['titles']=clust_txt
            clust_avg=np.mean(clust_embeddings, axis=0, keepdims=True)
            df['cluster_rank']=0.5*(1 + cosine_similarity(clust_embeddings,clust_avg))
            df1=df.sort_values('cluster_rank',ascending=False).copy()
            centroid_similarity = df['cluster_rank'].mean()
            if top_n==1:
                summary_txt = str(df1.iloc[0].titles).title()
            else:
                summary_txt = list(df1.head(top_n).titles.values)
        return summary_txt, centroid_similarity

    def hac_to_treemap(self, linkage_table, depth=3, nr_splits=3,min_size=0.1,max_extension=1):
        """
        Starting from a list of vectors, performs HAC using fastcluster, then splits results into a 
        series of layers with each layer consisting of a roughly equivalent number of slices

        Args:
            linkage_table (list): Linkage table produced as an output of a HAC algorithm (fastcluster or scipy) 
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
        >>> pt = PictureText(['txt']*7)
        >>> pt(X)
        Embeddings updated, external embeddings provided
        Linkage updated, using ward method and euclidean distances, time taken 0 secs
        >>> df = pt.hac_to_treemap(pt.linkage_table)
        >>> df.drop('cluster_table',axis=1)
            id parent cluster_members value
        9    9   Full       [4, 5, 6]     3
        10  10   Full          [2, 3]     2
        7    7   Full          [0, 1]     2
        5    5      9             [5]     1
        6    6      9             [6]     1
        4    4      9             [4]     1
        2    2     10             [2]     1
        3    3     10             [3]     1
        0    0      7             [0]     1
        1    1      7             [1]     1
        >>> list(df['cluster_table'].values)
        [{4: [4, '', '', 0, 1], 5: [5, '', '', 0, 1], 6: [6, '', '', 0, 1], 8: [8, 5, 6, 1.0, 2], 9: [9, 4, 8, 1.7320508075688772, 3]}, {2: [2, '', '', 0, 1], 3: [3, '', '', 0, 1], 10: [10, 2, 3, 9.0, 2]}, {0: [0, '', '', 0, 1], 1: [1, '', '', 0, 1], 7: [7, 0, 1, 1.0, 2]}, {5: [5, '', '', 0, 1]}, {6: [6, '', '', 0, 1]}, {4: [4, '', '', 0, 1]}, {2: [2, '', '', 0, 1]}, {3: [3, '', '', 0, 1]}, {0: [0, '', '', 0, 1]}, {1: [1, '', '', 0, 1]}]
        """
        go = True
        clust_idx = 'Full'
        df_res = pd.DataFrame([],columns=['cluster_id', 'cluster_parent', 'cluster_members', 'cluster_table', 'cluster_size'])
        
        hac = HAC(linkage_table, parent = clust_idx)
        new_clusters = hac.top_n_good_clusters(nr_splits,min_size=min_size,max_extension=max_extension)
        
        df_res=df_res.append(pd.DataFrame(new_clusters).T)
        depth=depth-1
        
        while go and depth>0:
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