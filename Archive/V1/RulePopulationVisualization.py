import sys
import os
import argparse
import pickle
import pandas as pd
import copy
import seaborn
import matplotlib.pyplot as plt
import networkx as nx
from Graph import Graph

import numpy as np
import random
from scipy.stats import pearsonr,spearmanr
import HClust
from fastcluster import linkage

####TO RUN THIS CODE, YOUR COMPUTER MUST HAVE R INSTALLED
# import rpy2.robjects as robjects
# from rpy2.robjects.packages import importr
# base = importr('base')
# pvclust = importr('pvclust')
# import rpy2.robjects as ro
# from rpy2.robjects.conversion import localconverter
# from rpy2.robjects import pandas2ri
#######


from networkx.algorithms.community.centrality import girvan_newman
from operator import itemgetter
import itertools

'''
model_file: assumes the model is an LCS and follows the interface of scikit-eLCS/XCS/ExSTraCS. Assumes the file is a 
            pickled file of the entire LCS object

training_file: assumes the training file given is the one that trained the above pickled model. Instance Labels


Example Run Command:
python RulePopulationVisualization.py --data-path /Users/robert/Desktop/vizData --model-file Model --training-file Hetero.csv --output-path /Users/robert/Desktop/vizOutputs --experiment-name test1
'''

def main(argv):
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--data-path', dest='data_path', type=str, help='path to directory containing datasets')
    parser.add_argument('--model-file', dest='model_file', type=str, help='pickled LCS model file name')
    parser.add_argument('--experiment-name', dest='experiment_name', type=str, help='name of experiment (no spaces)')
    parser.add_argument('--training-file', dest='training_file', type=str, help='training file name')
    parser.add_argument('--output-path', dest='output_path', type=str, help='path to output directory')
    parser.add_argument('--instance-label', dest='instance_label', type=str, default='')
    parser.add_argument('--class-label', dest='class_label', type=str, default='Class')
    parser.add_argument('--cluster-metric',dest='cluster_metric',type=str,default='default')

    options = parser.parse_args(argv[1:])
    data_path = options.data_path
    output_path = options.output_path
    experiment_name = options.experiment_name
    model_file = options.model_file
    training_file = options.training_file
    if options.instance_label == '':
        instance_label = 'None'
    else:
        instance_label = options.instance_label
    if options.class_label == '':
        class_label = 'None'
    else:
        class_label = options.class_label

    if options.cluster_metric == 'pearson':
        cluster_metric = 'pearson'
    elif options.cluster_metric == 'spearman':
        cluster_metric = 'spearman'
    else:
        cluster_metric = 'correlation'

    #Check arguments
    if not os.path.exists(data_path):
        raise Exception("Provided data_path does not exist")
    if not os.path.isfile(data_path+'/'+model_file):
        raise Exception("Provided model_file does not exist at given data_path")
    if not os.path.isfile(data_path+'/'+model_file):
        raise Exception("Provided model_file does not exist at given data_path")

    # if os.path.exists(output_path+'/'+experiment_name):
    #     raise Exception("Experiment Name must be unique")

    for char in experiment_name:
        if not char in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890':
            raise Exception('Experiment Name must be alphanumeric')

    if training_file[-4:] != '.csv':
        raise Exception('training_file must be a .csv file')

    # Create Experiment folder, with log and job folders
    if not os.path.exists(output_path + '/' + experiment_name):
        os.mkdir(output_path + '/' + experiment_name)

    #Unwrap data and set up variables
    file = open(data_path + '/' + model_file, 'rb')
    model = pickle.load(file)
    file.close()

    data = pd.read_csv(data_path+'/'+training_file,sep=',')
    num_train_instances = model.env.formatData.numTrainInstances
    if instance_label == 'None':
        data_features = data.drop(class_label,axis=1).values
        instances = list(range(num_train_instances))
    else:
        data_features = data.drop([class_label,instance_label], axis=1).values
        instances = data[instance_label].values
    data_phenotypes = data[class_label].values
    data_headers = data.drop(class_label,axis=1).columns.values

    rule_population = copy.copy(model.population.popSet)
    num_attributes = model.env.formatData.numAttributes
    num_rules = len(rule_population)

    #####CONTROL PANEL######
    do_rulepop_heatmap = True
    do_rulepop_network = True
    do_AT_comp = True
    do_exp_niche = True
    ########################

    #Heatmap and Clustermap of Rule Population
    if do_rulepop_heatmap:
        rule_specificity_array = []
        for instance in range(num_rules):
            a = []
            for attribute in range(num_attributes):
                a.append(0)
            rule_specificity_array.append(a)

        rule_index_count = 0
        for classifier in rule_population:
            for i in classifier.specifiedAttList:
                rule_specificity_array[rule_index_count][i] = 1
            rule_index_count+=1

        rule_df = pd.DataFrame(rule_specificity_array,columns=data_headers,index=list(range(num_rules)))

        if num_attributes > num_rules:
            rule_df = rule_df.T

        seaborn.heatmap(rule_df,cmap='plasma')
        plt.savefig(output_path+'/' + experiment_name + '/rulepopHeatmap.png')
        plt.close('all')

        if cluster_metric == 'spearman':
            seaborn.clustermap(rule_df,metric=spearmanDistance,cmap='plasma')
            plt.savefig(output_path + '/' + experiment_name + '/rulepopClustermapSpearman.png')
        elif cluster_metric == 'pearson':
            seaborn.clustermap(rule_df, metric=pearsonDistance,cmap='plasma')
            plt.savefig(output_path + '/' + experiment_name + '/rulepopClustermapPearson.png')
        else:
            seaborn.clustermap(rule_df, metric='correlation',cmap='plasma')
            plt.savefig(output_path + '/' + experiment_name + '/rulepopClustermapDefault.png')
        plt.close('all')

    #Rule specificity network
    if do_rulepop_network:
        attribute_acc_specificity_counts = model.get_final_attribute_specificity_list()
        acc_spec_dict = {}
        for header_index in range(len(data_headers)):
            acc_spec_dict[data_headers[header_index]] = attribute_acc_specificity_counts[header_index]
        attribute_cooccurrences = model.get_final_attribute_coocurrences(data_headers,len(data_headers))

        G = nx.Graph()
        edge_list = []
        weight_list = []
        for co in attribute_cooccurrences:
            G.add_edge(co[0],co[1],weight=co[3])
            edge_list.append((co[0],co[1]))
            weight_list.append(co[3])

        pos = nx.spring_layout(G,k=1)

        max_node_value = max(acc_spec_dict.values())
        for i in acc_spec_dict:
            acc_spec_dict[i] = acc_spec_dict[i]/max_node_value*1000

        max_weight_value = max(weight_list)
        for i in range(len(weight_list)):
            weight_list[i] = weight_list[i]/max_weight_value * 10


        nx.draw_networkx_nodes(G,pos,nodelist=acc_spec_dict.keys(),node_size=[v*1 for v in acc_spec_dict.values()],node_color='#FF3377')
        nx.draw_networkx_edges(G,pos,edge_color='#E0B8FF',edgelist=edge_list,width=[v*1 for v in weight_list])
        nx.draw_networkx_labels(G,pos)
        plt.axis('off')
        plt.savefig(output_path + '/' + experiment_name + '/rulespecificityGraph.png',dpi=300)
        plt.close('all')

    #AT clustered heatmap
    if do_AT_comp:
        #Get AT Scores for each instance
        AT_scores = model.get_attribute_tracking_scores(instance_labels=np.array(instances))

        #Normalize AT Scores and convert to np.ndarray and plot heatmap
        normalized_AT_scores = []
        for i in range(len(AT_scores)):
            normalized = AT_scores[i][1]
            max_score = max(normalized)
            for j in range(len(normalized)):
                if max_score != 0:
                    normalized[j] /= max_score
                else:
                    normalized[j] = 0
            normalized_AT_scores.append(np.array(normalized))
        normalized_AT_scores = np.array(normalized_AT_scores)

        AT_full_df = pd.DataFrame(normalized_AT_scores, columns=data_headers, index=instances)
        seaborn.heatmap(AT_full_df,cmap='plasma')
        plt.savefig(output_path + '/' + experiment_name + '/ATheatmap.png')
        plt.close('all')

        #Get feature linkage
        raw_sums = model.get_final_attribute_tracking_sums() #Sum of AT scores for each feature, used for clustering and ranking cluster relevance
        feature_linkage_rows = linkage(np.transpose(np.array([raw_sums])),metric='euclidean',method='ward')
        #feature_linkage_rows = linkage(np.transpose(normalized_AT_scores), metric='euclidean', method='ward')

        #Find Relevant Features via significant hclust
        cluster_tree_features = HClust.createClusterTree(feature_linkage_rows, data_headers,np.transpose(np.array([raw_sums])))
        #cluster_tree_features = HClust.createClusterTree(feature_linkage_rows, data_headers,np.transpose(normalized_AT_scores))
        feature_clusters = cluster_tree_features.getSignificantClusters(p_value=0.85, sample_count=30,metric='euclidean',method='ward')
        print(feature_clusters)

        num_expected_hetero_subgroups = min(len(feature_clusters),2)

        cluster_sums = list(np.zeros(len(feature_clusters)))
        count = 0
        for cluster_f in feature_clusters:
            for feature in cluster_f:
                f_index = np.where(data_headers == feature)[0][0]
                cluster_sums[count] += raw_sums[f_index]/len(cluster_f)
            count += 1

        cluster_sums_sorted = sorted(cluster_sums)
        cluster_sums_sorted.reverse()
        important_cluster_indices = []
        for i in range(num_expected_hetero_subgroups):
            important_cluster_indices.append(cluster_sums.index(cluster_sums_sorted[i]))

        important_features = []
        for c in important_cluster_indices:
            important_features.extend(feature_clusters[c])

        print(important_features)
        important_feature_indices = []
        for f in important_features:
            important_feature_indices.append(list(data_headers).index(f))

        #Filter AT_scores
        filtered_AT_scores = []
        for i in range(len(normalized_AT_scores)):
            filtered = []
            for z in important_feature_indices:
                filtered.append(normalized_AT_scores[i,z])
            filtered_AT_scores.append(filtered)

        #Filter headers
        new_headers = []
        for z in important_feature_indices:
            new_headers.append(data_headers[z])

        #Cluster instances
        AT_df = pd.DataFrame(filtered_AT_scores, columns=new_headers, index=instances)
        g = seaborn.clustermap(AT_df,metric='correlation',method='ward',cmap='plasma')
        plt.savefig(output_path + '/' + experiment_name + '/ATclustermap.png')
        plt.close('all')

        cluster_tree = HClust.createClusterTree(g.dendrogram_row.linkage,instances,AT_df.to_numpy())
        clusters = cluster_tree.getSignificantClusters(p_value=0.05,sample_count=100,metric='correlation',method='ward')
        color_dict = {}
        for cluster in clusters:
            random_color = randomHex()
            for instance_label in cluster:
                color_dict[instance_label] = random_color
        color_list = pd.Series(dict(sorted(color_dict.items())))

        g = seaborn.clustermap(AT_df, row_linkage=g.dendrogram_row.linkage,col_linkage=g.dendrogram_col.linkage,row_colors=color_list,cmap='plasma')
        plt.savefig(output_path + '/' + experiment_name + '/ATclustermapLabeled.png',dpi=300)
        plt.close('all')

    #Experimental niche map
    if do_exp_niche:
        popSet = []
        for classifier in model.population.popSet:
            if classifier.accuracy > 0.75:
                popSet.append(copy.deepcopy(classifier))

        rule_map = Graph(popSet)
        for train_instance_index in range(len(model.env.formatData.trainFormatted[0])):
            state = model.env.formatData.trainFormatted[0][train_instance_index]
            phenotype = model.env.formatData.trainFormatted[1][train_instance_index]
            correct_set_indices = []
            for rule_index in range(len(popSet)):
                rule = popSet[rule_index]
                if rule.match(model,state) and rule.phenotype == phenotype:
                    correct_set_indices.append(rule_index)
            #printCorrectSetAttrSpecList(correct_set_indices,popSet,num_attributes)
            for rule_index_1 in correct_set_indices:
                for rule_index_2 in correct_set_indices:
                    if rule_index_1 != rule_index_2:
                        rule_map.addOutgoingEdgeFromTo(rule_index_1,rule_index_2,1)

        #rule_map.subtractFromAllEdges(2)
        #rule_map.subtractFromAllEdges(int(rule_map.getMaxWeight()/5))

        G = nx.Graph()
        edge_list = []
        weight_list = []
        for rule in rule_map.rules:
            for outgoing in rule_map.rules[rule].outgoingEdges:
                if not G.has_edge(rule,outgoing) and rule_map.rules[rule].outgoingEdges[outgoing] > 0:
                    G.add_edge(rule,outgoing,weight=rule_map.rules[rule].outgoingEdges[outgoing])
                    edge_list.append((rule,outgoing))
                    weight_list.append(rule_map.rules[rule].outgoingEdges[outgoing])
        for v in rule_map.getVertices():
            G.add_node(v)
        pos = nx.spring_layout(G, k=1)

        #color nodes by class
        classes = rule_map.getClassDict()
        for classType in classes:
            newPos = {}
            for key in pos:
                if key in classes[classType]:
                    newPos[key] = pos[key]

            nx.draw_networkx_nodes(G, newPos,node_color=randomHex(),node_size=10,nodelist=classes[classType])

        #color nodes by community
        # comp = girvan_newman(G, most_valuable_edge=heaviest)
        # limited = itertools.takewhile(lambda c: len(c) <= 100, comp)
        # l = []
        # for communities in limited:
        #     l.append(tuple(sorted(c) for c in communities))
        #     #print(tuple(sorted(c) for c in communities))
        # groups = l[len(l) - 1]
        # for group in groups:
        #     newPos = {}
        #     for key in pos:
        #         if key in group:
        #             newPos[key] = pos[key]
        #     nx.draw_networkx_nodes(G, newPos, node_color=randomHex(), node_size=10, nodelist=group)

        # for group in groups:
        #     for rule in group:
        #         printRule(popSet[rule],num_attributes)
        #     print()

        #Draw Edges
        max_weight_value = max(weight_list)
        for i in range(len(weight_list)):
            weight_list[i] = weight_list[i] / max_weight_value * 2

        nx.draw_networkx_edges(G, pos, edge_color='#E0B8FF',edgelist=edge_list,width=[v*1 for v in weight_list])
        #nx.draw_networkx_labels(G, pos)
        plt.axis('off')
        plt.savefig(output_path + '/' + experiment_name + '/nicheGraph.png',dpi=500)
        plt.close('all')



def randomHex():
    s = '#'
    for i in range(6):
        s+=random.choice(['0','1','2','3','4','5','6','7','8','9','A','B','C','D','E','F'])
    return s

def heaviest(G):
    u,v,w = max(G.edges(data='weight'),key=itemgetter(2))
    return (u,v)

def printRule(classifier,numAttr):
    attributeCounter = 0

    for attribute in range(numAttr):
        if attribute in classifier.specifiedAttList:
            specifiedLocation = classifier.specifiedAttList.index(attribute)
            if not isinstance(classifier.condition[specifiedLocation],list):  # isDiscrete
                print(classifier.condition[specifiedLocation], end="\t")
            else:
                print("[", end="")
                print(
                    round(classifier.condition[specifiedLocation][0] * 10) / 10,
                    end=", ")
                print(
                    round(classifier.condition[specifiedLocation][1] * 10) / 10,
                    end="")
                print("]", end="\t")
        else:
            print("#", end="\t")
        attributeCounter += 1
    if not isinstance(classifier.phenotype,list):
        print(classifier.phenotype, end="\t")
    else:
        print("[", end="")
        print(round(classifier.phenotype[0] * 10) / 10, end=", ")
        print(round(classifier.phenotype[1] * 10) / 10, end="")
        print("]", end="\t")
    print()

def printCorrectSetAttrSpecList(correctSetIndices,popSet,numAttributes):
    attributeAccList = []
    for i in range(numAttributes):
        attributeAccList.append(0.0)
    for cli in correctSetIndices:
        cl = popSet[cli]
        for ref in cl.specifiedAttList:
            attributeAccList[ref] += cl.numerosity * cl.accuracy
    for a in range(len(attributeAccList)):
        if max(attributeAccList) != 0:
            attributeAccList[a]  = int(attributeAccList[a]/max(attributeAccList)*100)/100
        else:
            attributeAccList[a] = 0
    print(attributeAccList)

def spearmanDistance(u,v):
    return 1 - spearmanr(u,v)[0]

def pearsonDistance(u,v):
    return 1 - pearsonr(u,v)[0]

if __name__ == '__main__':
    sys.exit(main(sys.argv))

