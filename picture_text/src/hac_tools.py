from picture_text.src.utils import flatten_list
from scipy.cluster.hierarchy import to_tree
#import matplotlib.pyplot as plt
from scipy.cluster import hierarchy
import numpy as np

class HAC():
    def __init__(self, linkage_table, parent=None):
        """
        Instantiates a class, starting with a fastcluster or scipy HAC linkage table and helping the move to a treemap
        Alternatively this can also receive a ready linkage table or a subset thereof for the cases where only a part of the tree is being analysed

        Args:
            linkage_table (list or dict): 
                Linkage table produced as an output of a HAC algorithm (fastcluster or scipy) 
                OR
                a dictionary table subset thereof
            parent (int or string, optional): Parent ID value to be used as parent of this dataset

        >>> X=[[x] for x in [1001,1000,1,10,99,100,101]]
        >>> z=fastcluster.single(X)
        >>> hac = HAC(z) # Linkage table case
        >>> hac.tbl
        {0: [0, '', '', 0, 1], 1: [1, '', '', 0, 1], 2: [2, '', '', 0, 1], 3: [3, '', '', 0, 1], 4: [4, '', '', 0, 1], 5: [5, '', '', 0, 1], 6: [6, '', '', 0, 1], 7: [7, 0, 1, 1.0, 2], 8: [8, 5, 6, 1.0, 2], 9: [9, 4, 8, 1.0, 3], 10: [10, 2, 3, 9.0, 2], 11: [11, 9, 10, 89.0, 5], 12: [12, 7, 11, 899.0, 7]}
        >>> hac.tbl_clusters
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        >>> z2={8: [8, '', '', 0, 1], 9: [9, '', '', 0, 1], 20: [20, 8, 9, 2.0, 2]}
        >>> hac2=HAC(z2) # Dictionary case
        >>> hac2.tbl
        {8: [8, '', '', 0, 1], 9: [9, '', '', 0, 1], 20: [20, 8, 9, 2.0, 2]}
        >>> hac2.tbl_clusters
        [8, 9, 20]
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

    def dendrogram(self, **kwargs):
        """
        Create a dendrogram from the data in the class
        """
        #plt.figure()
        dn = hierarchy.dendrogram(self.linkage_table, **kwargs)

    def get_members(self, cluster_id):
        """
        Get list of member and cluster ids for a certain node id

        Args:
            cluster_id (int): Cluster ID to get members for

        Returns:
            members (list): list of original datapoints belonging to this cluster ID
            clusters (list): list of subclusters belonging to this cluster ID
            table (dict): full table with all members both cluster and datapoints
        
        >>> X=[[x] for x in [1001,1000,1,10,99,100,101]]
        >>> z=fastcluster.single(X)
        >>> hac = HAC(z)
        >>> m, c, t = hac.get_members(3)
        >>> m
        [3]
        >>> c
        []
        >>> t
        {3: [3, '', '', 0, 1]}
        >>> m, c, t = hac.get_members(12)
        >>> m
        [0, 1, 2, 3, 4, 5, 6]
        >>> c
        [7, 8, 9, 10, 11, 12]
        >>> t
        {0: [0, '', '', 0, 1], 1: [1, '', '', 0, 1], 2: [2, '', '', 0, 1], 3: [3, '', '', 0, 1], 4: [4, '', '', 0, 1], 5: [5, '', '', 0, 1], 6: [6, '', '', 0, 1], 7: [7, 0, 1, 1.0, 2], 8: [8, 5, 6, 1.0, 2], 9: [9, 4, 8, 1.0, 3], 10: [10, 2, 3, 9.0, 2], 11: [11, 9, 10, 89.0, 5], 12: [12, 7, 11, 899.0, 7]}
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
        """
        >>> X=[[x] for x in [1001,1000,1,10,99,100,101]]
        >>> z=fastcluster.single(X)
        >>> hac = HAC(z)
        >>> child_id, child_size, total_size = hac.top_n_clusters(3)
        >>> child_id
        [2, 3, 9, 7]
        >>> child_size
        [1, 1, 3, 2]
        >>> total_size
        7
        """
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
        Extends the number of clusters to a specified limit if very small clusters are found
        
        Args:
            nr_clusters (int): Number of clusters to return
            min_size (float, optional): Minimal size for a cluster, as a % of total number of observations in X,
                defaults to 0.1 (meaning the smallest cluster should be at least 10% of overall size)
            max_extension (float, optional): Percent extension to nr_splits if min_size not met by all clusters, defaults to 1.0
        Example: 
            - if nr_splits = 3, min_size = 0.1, max_extension=1 
            - max_extension = 1 means up to 100% increase in nr_splits, i.e. up to 6 splits in this case
            - only 1 out of 3 clusters initially are > 10% 
            - Initially this will add 2 more splits (3 - 1) to a total of 5 which is less then the max_extension allowance of 6
            - If again 2 of the 5 are under 10%, this would mean increasing number of splits to 7, however, the max is 6 so we end up with 6

        Returns:
            res (list): List of dictionaries containing details (ids, parent, members, table, size) of all relevant clusters found

        >>> X=[[x] for x in [1001,1000,1,10,99,100,101]]
        >>> z=fastcluster.single(X)
        >>> hac = HAC(z)
        >>> res = hac.top_n_good_clusters(3)
        >>> res
        {9: {'cluster_id': 9, 'cluster_parent': -1, 'cluster_members': [4, 5, 6], 'cluster_table': {4: [4, '', '', 0, 1], 5: [5, '', '', 0, 1], 6: [6, '', '', 0, 1], 8: [8, 5, 6, 1.0, 2], 9: [9, 4, 8, 1.0, 3]}, 'cluster_size': 3}, 10: {'cluster_id': 10, 'cluster_parent': -1, 'cluster_members': [2, 3], 'cluster_table': {2: [2, '', '', 0, 1], 3: [3, '', '', 0, 1], 10: [10, 2, 3, 9.0, 2]}, 'cluster_size': 2}, 7: {'cluster_id': 7, 'cluster_parent': -1, 'cluster_members': [0, 1], 'cluster_table': {0: [0, '', '', 0, 1], 1: [1, '', '', 0, 1], 7: [7, 0, 1, 1.0, 2]}, 'cluster_size': 2}}
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
    """
    Returns the id of the left cluster belonging to a node
    
    Args:
        nd (node): NOde from nodelist in scipy.cluster.hierarchy.to_tree
    Returns:
        id or '' if the node is just a datapoint and not a cluster
    """
    try:
        return nd.get_left().get_id()
    except:
        return ''
    
def right_clust(nd):
    """
    Returns the id of the right cluster belonging to a node
    
    Args:
        nd (node): NOde from nodelist in scipy.cluster.hierarchy.to_tree
    Returns:
        id or '' if the node is just a datapoint and not a cluster
    """
    try:
        return nd.get_right().get_id()
    except:
        return ''