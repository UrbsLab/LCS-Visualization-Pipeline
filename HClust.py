
'''
Methods related to Hierarchical Clustering
'''
import numpy as np
from fastcluster import linkage
import copy
import random

########CLUSTER TREE METHODS############################################################################################
def createClusterTree(linkage_matrix,instance_labels,instances):
    '''
    :param linkage_matrix: array of [[cluster_index_1,cluster_index_2,distance_between_clusters,# observations in new cluster],...]
    :param instance_labels: array of instance labels, in order given to the the mechanism that created the above linkage matrix
    :param instances: 2D numpy.ndarray of instances, where rows are the object to be clustered.
    :return: ClusterTree based on linkage matrix and instance_labels
    '''

    return ClusterTree(linkage_matrix,instance_labels,instances)

class ClusterTree:
    def __init__(self,linkage_matrix,instance_labels,instances):
        self.clusters = []
        self.instances = instances
        self.label_to_index_map = {}
        self.instance_labels = instance_labels
        self.significant_clusters = None
        self.significant_cluster_objs = None

        #Map instance labels to instance indices
        for index in range(len(instance_labels)):
            self.label_to_index_map[instance_labels[index]] = index

        #Populate initial instances
        for label in instance_labels:
            self.clusters.append(Cluster([self.label_to_index_map[label]],None,None,None,0))

        #Traverse linkage matrix to create tree
        cluster_rank = 1
        for row in linkage_matrix:
            cluster_index_1 = int(round(row[0]))
            cluster_index_2 = int(round(row[1]))
            distance_between_clusters = row[2]
            combined_instance_indices = self.clusters[cluster_index_1].instance_index_array+self.clusters[cluster_index_2].instance_index_array
            new_cluster = Cluster(combined_instance_indices,cluster_index_1,cluster_index_2,distance_between_clusters,cluster_rank)
            self.clusters.append(new_cluster)

            #Set sibling and parent indices
            self.clusters[cluster_index_1].parent_index = len(self.clusters) - 1
            self.clusters[cluster_index_2].parent_index = len(self.clusters) - 1
            self.clusters[cluster_index_1].sibling_index = cluster_index_2
            self.clusters[cluster_index_2].sibling_index = cluster_index_1

            cluster_rank += 1

    def rootCluster(self):
        return self.clusters[-1]

    def getChildrenOf(self,cluster):
        return (self.clusters[cluster.child_index_1],self.clusters[cluster.child_index_2])

    def twoMeansClusterIndexOf(self,cluster):
        subcluster1,subcluster2 = self.getChildrenOf(cluster)
        mean_1 = self.getMeanInstanceOfClusterInstances(subcluster1)
        mean_2 = self.getMeanInstanceOfClusterInstances(subcluster2)
        mean_total = self.getMeanInstanceOfClusterInstances(cluster)

        ss1 = 0
        for instance_index in subcluster1.instance_index_array:
            instance = self.instances[instance_index]
            ss1 += pow(np.linalg.norm(mean_1-instance),2)

        ss2 = 0
        for instance_index in subcluster2.instance_index_array:
            instance = self.instances[instance_index]
            ss2 += pow(np.linalg.norm(mean_2-instance),2)

        tts = 0
        for instance_index in cluster.instance_index_array:
            instance = self.instances[instance_index]
            tts += pow(np.linalg.norm(mean_total-instance),2)

        return (ss1+ss2)/tts

    def getMeanInstanceOfClusterInstances(self,cluster):
        instance_indices = cluster.instance_index_array
        num_attributes = self.instances[0].size
        num_instances = len(instance_indices)
        mean_array = np.empty(num_attributes)
        for index in instance_indices:
            mean_array += self.instances[index]
        return mean_array/num_instances

    def getSDInstanceOfClusterInstances(self,cluster):
        instance_indices = cluster.instance_index_array
        array = []
        for index in instance_indices:
            array.append(self.instances[index])
        array = np.transpose(np.array(array))
        std_array = []
        for a in array:
            std_array.append(np.std(a))
        return np.array(std_array)

    def getRandomInstances(self,mean_array,sd_array,num_samples):
        total = []
        for i in range(mean_array.size):
            a = np.random.normal(mean_array[i],sd_array[i],num_samples)
            total.append(a)
        return np.transpose(np.array(total))

    def indicesToLabels(self,array):
        l = []
        for i in array:
            b = []
            for a in i:
                b.append(self.instance_labels[a])
            l.append(b)
        return l

    def getNSignificantClusters(self,N,metric,method,p_value=0.05,sample_count=30,random_state=None):
        if self.significant_clusters == None:
            self.getSignificantClusters(metric,method,p_value,sample_count,random_state=random_state)

        copy_objs = copy.deepcopy(self.significant_cluster_objs)
        if N > len(copy_objs):
            N = len(copy_objs)

        while len(copy_objs) > N:
            #Find lowest rank cluster that also has a sibling
            min_rank = np.inf
            lowest_cluster = None
            for cluster in copy_objs:
                if cluster.height_tracker < min_rank:
                    #Find sibling, if it exists.
                    c_index = self.clusters[cluster.sibling_index].sibling_index
                    potential_sibling_cluster = None
                    for c in copy_objs:
                        if c.sibling_index == c_index:
                            potential_sibling_cluster = cluster
                    if potential_sibling_cluster != None:
                        min_rank = cluster.height_tracker
                        lowest_cluster = cluster

            #Get lowest rank cluster's sibling
            own_index = self.clusters[lowest_cluster.sibling_index].sibling_index
            sibling_cluster = None
            for cluster in copy_objs:
                if cluster.sibling_index == own_index:
                    sibling_cluster = cluster

            #Find lowest rank cluster's parent
            parent_cluster = copy.deepcopy(self.clusters[lowest_cluster.parent_index])

            #Remove siblings from copy. Add parent
            copy_objs.remove(lowest_cluster)
            copy_objs.remove(sibling_cluster)
            copy_objs.append(parent_cluster)

        #Convert copy into list of lists of labels
        sig_clusters = []
        colors = []
        for c in copy_objs:
            sig_clusters.append(c.instance_index_array)
            colors.append(c.hex)
        return self.indicesToLabels(sig_clusters),colors

    def getSignificantClusters(self,metric,method,p_value=0.05,sample_count=30,random_state=None):
        if random_state != None:
            random.seed(random_state)
            np.random.seed(random_state)

        if self.significant_clusters == None:
            self.significant_clusters = []
            self.significant_cluster_objs = []
            root_cluster = self.rootCluster()
            self.getSignificantClusters_Recur(root_cluster,p_value,sample_count,metric,method)

            colors = []
            for c in self.significant_cluster_objs:
                colors.append(c.hex)
            return self.indicesToLabels(self.significant_clusters),colors
        else:
            colors = []
            for c in self.significant_cluster_objs:
                colors.append(c.hex)
            return self.indicesToLabels(self.significant_clusters),colors

    def getSignificantClusters_Recur(self,cluster,p_value,sample_count,metric,method):
        if cluster.child_index_1 != None:
            #observed_CI = self.twoMeansClusterIndexOf(cluster)
            observed_CI = cluster.distance_between_children
            cluster_P_value = self.getPValue(cluster,sample_count,observed_CI,metric,method)
            significance_cutoff = p_value*(cluster.getNumInstances()-1)/(len(self.instances)-1)
            if cluster_P_value <= significance_cutoff:
                subcluster1,subcluster2 = self.getChildrenOf(cluster)
                self.getSignificantClusters_Recur(subcluster1,p_value,sample_count,metric,method)
                self.getSignificantClusters_Recur(subcluster2,p_value,sample_count,metric,method)
            elif cluster_P_value > significance_cutoff:
                self.significant_clusters.append(cluster.instance_index_array)
                self.significant_cluster_objs.append(cluster)
        else:
            self.significant_clusters.append(cluster.instance_index_array)
            self.significant_cluster_objs.append(cluster)

    def getPValue(self,cluster,sample_count,observed_CI,metric,method):
        num_below_observed = 0

        cluster_mean = self.getMeanInstanceOfClusterInstances(cluster)
        cluster_sd = self.getSDInstanceOfClusterInstances(cluster)

        for i in range(sample_count):
            #randomly sample instances based off of cluster instance mean and SD
            array = self.getRandomInstances(cluster_mean,cluster_sd,len(cluster.instance_index_array))
            labels = list(range(len(cluster.instance_index_array)))
            linkage_matrix = linkage(array,metric=metric,method=method)

            #Create Cluster Tree from linkage matrix and get CI of root cluster
            sub_tree = ClusterTree(linkage_matrix,labels,array)
            #sub_tree_ci = sub_tree.twoMeansClusterIndexOf(sub_tree.rootCluster())
            sub_tree_ci = sub_tree.rootCluster().distance_between_children

            #If CI of root cluster is less/greater than or equal to observed, increment
            #if sub_tree_ci <= observed_CI: #For 2 Means computation
            if sub_tree_ci >= observed_CI: #For linkage distance computation
                num_below_observed += 1
        return num_below_observed/sample_count

class Cluster:
    def __init__(self,instance_index_array,child_index_1,child_index_2,distance_between_chidren,height_tracker):
        '''
        :param instance_index_array: list of instance indices that make up this cluster
        :param child_index_1/2: index of children in ClusterTree's clusters attribute
        :param distance_between_chidren: linkage distance between children
        :param height_tracker: an integer that ranks the cluster's height on the tree. Lower rank => closer to leaves
        '''
        self.instance_index_array = instance_index_array
        self.child_index_1 = child_index_1
        self.child_index_2 = child_index_2
        self.distance_between_children = distance_between_chidren
        self.height_tracker = height_tracker
        self.parent_index = None
        self.sibling_index = None
        self.hex = self.randomHex()

    def getNumInstances(self):
        return len(self.instance_index_array)

    def randomHex(self):
        s = '#'
        for i in range(6):
            s += random.choice(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F'])
        return s

