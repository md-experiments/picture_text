from utils import flatten_list
from scipy.cluster.hierarchy import to_tree
import matplotlib.pyplot as plt
from scipy.cluster import hierarchy
import fastcluster
import numpy as np

class HAC():
    def __init__(self, linkage_table, parent=None):
        """
        """
        if parent == None:
            self.parent = -1
        else:
            self.parent = parent
        if not isinstance(linkage_table, dict):
            self.linkage_table = linkage_table
            self.rootnode, self.nodelist = to_tree(self.linkage_table, rd=True)
            self.tbl = {i: [i, left_clust(nd), right_clust(nd), nd.dist, nd.count] for (i, nd) in enumerate(self.nodelist)}
        else:
            self.tbl = linkage_table
        self.tbl_clusters = list(self.tbl.keys())
        self.tbl_clusters.sort()

    def dendrogram(self):
        plt.figure()
        dn = hierarchy.dendrogram(self.linkage_table)

    def get_members(self, cluster_id):
        """
        Get list of member and cluster ids for a certain node / starting point
        """
        memb=[]
        get_idx=new_memb=[cluster_id]

        while len(new_memb)>0:
            memb = memb + new_memb
            new_memb = []
            for idx in get_idx:
                new_memb = new_memb + [m for m in self.tbl[idx][1:3] if m != '']
            get_idx = new_memb

        memb.sort()
        members = [m for m in memb if self.tbl[m][1] == '']
        clusters = [m for m in memb if not self.tbl[m][1] == '']
        # Contains the full list of members
        table = {m: self.tbl[m] for m in memb}
        return members, clusters, table

    def top_n_clusters(self,nr_clusters):
        clust_id=self.tbl_clusters[-nr_clusters:]
        clust_id=[c for c in clust_id if self.tbl[c][1]!='']
        top_n=[self.tbl[c] for c in clust_id if self.tbl[c][1]!='']
        
        child_id = flatten_list([t[1:3] for t in top_n])
        child_id = [c for c in child_id if c not in clust_id]
        child_size = [self.tbl[c][4] for c in child_id]
        total_size = sum(child_size)
        return child_id, child_size, total_size

    def top_n_good_clusters(self,nr_clusters,min_size=0.1,max_extension=1.0):
        """
        Returns the members, clusters and linkage tables of the top N clusters. 
        TBD Will look for a minimal size of each cluster and extend size of N until enough clusters are found that match the minimal size

        Args:
            nr_clusters (int): Number of clusters to return
            min_size (float): Minimal size of cluster
            max_extension (float): Relative proportion to extend search until all clusters are minimal size
        
        """
        
        nr_clusters = nr_clusters - 1
        max_num_clusters = int(nr_clusters * (1 + max_extension))
        check = True
        # While number of tiny clusters (< min_size) is non-zero keep increasing
        while check and nr_clusters < max_num_clusters:
            clust_id, clust_size, total_size=self.top_n_clusters(nr_clusters)
            nr_tiny_clusters = sum([c / total_size < min_size for c in clust_size]) 
            check = nr_tiny_clusters > 0
            nr_clusters = min(nr_clusters + nr_tiny_clusters, max_num_clusters)

        clust_id, clust_size, total_size=self.top_n_clusters(nr_clusters)
        
        res = {}
        for c in clust_id:
            m,_,t = self.get_members(c)
            res[c]={
                'cluster_id': c,
                'cluster_parent': self.parent,
                'cluster_members': m,
                'cluster_table': t,
                'cluster_size': len(m),
            }
        # Check counts still match and no datapoints lost
        assert(total_size==sum([res[c]['cluster_size'] for c in clust_id]))
        return res

def left_clust(nd):
    try:
        return nd.get_left().get_id()
    except:
        return ''
    
def right_clust(nd):
    try:
        return nd.get_right().get_id()
    except:
        return ''