import pandas as pd
import random
from skExSTraCS import ExSTraCS
import numpy as np
import os
import pickle
import copy
import seaborn
import matplotlib.pyplot as plt
from scipy.stats import pearsonr,spearmanr
from scipy.spatial.distance import euclidean
import networkx as nx
import sys
import math
import HClust
import csv

def main(argv):
    # CONTROL PANEL ####################################################################################################
    # Dataset Info
    datapath = 'Datasets/vivekHetero.txt'
    #datapath = 'Datasets/GAMETES/Files/Core/n2l2b2h2_0_EDM-1_01_hetLabel.txt'
    class_label = 'Class'
    instance_label = None
    group_label = 'Model'
    cv_count = 3

    # Experiment Info
    experiment_name = argv[1]

    # Training Info
    learning_iterations = 16000
    N = 1000
    nu = 1
    attribute_tracking_method = 'wh'
    use_expert_knowledge = None
    random_state = None

    # Analysis Info
    rulepop_clustering_method = 'pearson'

    ####################################################################################################################
    # Set random_seed
    if random_state == None:
        random_state = random.randint(0,1000000)

    # Create train and test sets
    if datapath[-1] == 't': #txt
        dataset = pd.read_csv(datapath, sep='\t')
    else: #csv
        dataset = pd.read_csv(datapath, sep=',')

    #Remove 'unnamed columns'
    dataset = dataset.loc[:, ~dataset.columns.str.contains('^Unnamed')]

    train_dfs,test_dfs = cv_partitioner(dataset,cv_count,class_label,random_state)

    # Create Experiment Folders
    if not os.path.exists('Outputs/' + experiment_name):
        os.mkdir('Outputs/' + experiment_name)
        os.mkdir('Outputs/' + experiment_name + '/models')
        os.mkdir('Outputs/' + experiment_name + '/visualizations')
        os.mkdir('Outputs/' + experiment_name + '/visualizations/rulepop')
        os.mkdir('Outputs/' + experiment_name + '/visualizations/at')

    # Write Metadata
    outfile = open('Outputs/' + experiment_name + '/metadata', mode='w')
    outfile.write('datapath: ' + str(datapath)+'\n')
    outfile.write('learning_iterations: '+str(learning_iterations)+'\n')
    outfile.write('N: ' + str(N)+'\n')
    outfile.write('nu: ' + str(nu)+'\n')
    outfile.write('attribute_tracking_method: ' + str(attribute_tracking_method)+'\n')
    outfile.close()

    for cv in range(cv_count):
        #Get Data Values################################################################################################
        if instance_label != None and group_label != None:
            train_dfs[cv].set_index([instance_label, group_label], inplace=True)
            test_dfs[cv].set_index([instance_label, group_label], inplace=True)
            use_group_label = group_label
            use_inst_label = instance_label
        elif instance_label != None and group_label == None:
            train_dfs[cv] = train_dfs[cv].assign(group=np.ones(train_dfs[cv].values.shape[0]))
            train_dfs[cv].set_index([instance_label, 'group'], inplace=True)
            test_dfs[cv] = test_dfs[cv].assign(group=np.ones(test_dfs[cv].values.shape[0]))
            test_dfs[cv].set_index([instance_label, 'group'], inplace=True)
            use_group_label = 'group'
            use_inst_label = instance_label
        elif instance_label == None and group_label != None:
            train_dfs[cv] = train_dfs[cv].assign(instance=np.array(list(range(train_dfs[cv].values.shape[0]))))
            train_dfs[cv].set_index([group_label, 'instance'], inplace=True)
            test_dfs[cv] = test_dfs[cv].assign(instance=np.array(list(range(train_dfs[cv].values.shape[0],train_dfs[cv].values.shape[0]+test_dfs[cv].values.shape[0]))))
            test_dfs[cv].set_index([group_label, 'instance'], inplace=True)
            use_group_label = group_label
            use_inst_label = 'instance'
        else:
            train_dfs[cv] = train_dfs[cv].assign(instance=np.array(list(range(train_dfs[cv].values.shape[0]))))
            train_dfs[cv] = train_dfs[cv].assign(group=np.ones(train_dfs[cv].values.shape[0]))
            train_dfs[cv].set_index(['group', 'instance'], inplace=True)
            test_dfs[cv] = test_dfs[cv].assign(instance=np.array(list(range(train_dfs[cv].values.shape[0],train_dfs[cv].values.shape[0]+test_dfs[cv].values.shape[0]))))
            test_dfs[cv] = test_dfs[cv].assign(group=np.ones(test_dfs[cv].values.shape[0]))
            test_dfs[cv].set_index(['group', 'instance'], inplace=True)
            use_group_label = 'group'
            use_inst_label = 'instance'

        train_data_features = train_dfs[cv].drop(class_label,axis=1).values
        train_data_phenotypes = train_dfs[cv][class_label].values
        train_instance_labels = train_dfs[cv].index.get_level_values(use_inst_label).tolist()
        train_group_labels = train_dfs[cv].index.get_level_values(use_group_label).tolist()

        test_data_features = test_dfs[cv].drop(class_label,axis=1).values
        test_data_phenotypes = test_dfs[cv][class_label].values
        test_instance_labels = test_dfs[cv].index.get_level_values(use_inst_label).tolist()
        test_group_labels = test_dfs[cv].index.get_level_values(use_group_label).tolist()

        data_headers = train_dfs[cv].drop(class_label, axis=1).columns.values

        group_colors = {}
        for group_name in set(test_group_labels+train_group_labels):
            random_color = randomHex()
            group_colors[group_name] = random_color

        # Train ExSTraCS model##########################################################################################
        if use_expert_knowledge != None:
            scores = use_expert_knowledge.fit(train_data_features,train_data_phenotypes).feature_importances_
            model = ExSTraCS(learning_iterations=learning_iterations,N=N,nu=nu,attribute_tracking_method=attribute_tracking_method,expert_knowledge=scores,rule_compaction=None,random_state=random_state)
        else:
            model = ExSTraCS(learning_iterations=learning_iterations,N=N,nu=nu,attribute_tracking_method=attribute_tracking_method,rule_compaction=None,random_state=random_state)
        model.fit(train_data_features, train_data_phenotypes)

        outfile = open('Outputs/' + experiment_name + '/models/model_' + str(cv), 'wb')
        pickle.dump(model,outfile)
        outfile.close()

        # Export Testing Accuracy
        outfile = open('Outputs/' + experiment_name + '/models/testingAcc_' + str(cv), mode='w')
        outfile.write(str(model.score(test_data_features,test_data_phenotypes)))

        #Rule Population Heatmap########################################################################################
        rule_specificity_array = []
        rule_population = copy.copy(model.population.popSet)
        num_attributes = model.env.formatData.numAttributes
        num_rules = len(rule_population)
        for instance in range(num_rules):
            a = []
            for attribute in range(num_attributes):
                a.append(0)
            rule_specificity_array.append(a)

        rule_index_count = 0
        for classifier in rule_population:
            for i in classifier.specifiedAttList:
                rule_specificity_array[rule_index_count][i] = 1
            rule_index_count += 1

        rule_df = pd.DataFrame(rule_specificity_array, columns=data_headers, index=list(range(num_rules)))

        if num_attributes > num_rules:
            rule_df = rule_df.T

        seaborn.heatmap(rule_df, cmap='plasma')
        plt.savefig('Outputs/' + experiment_name + '/visualizations/rulepop/rulepopHeatmap_'+str(cv)+'.png')
        plt.close('all')

        #Rule Population Clustermap
        if rulepop_clustering_method == 'spearman':
            seaborn.clustermap(rule_df,metric=spearmanDistance,method='ward',cmap='plasma')
            plt.savefig('Outputs/' + experiment_name + '/visualizations/rulepop/rulepopClusterMapSpearman_'+str(cv)+'.png')
        elif rulepop_clustering_method == 'pearson':
            seaborn.clustermap(rule_df, metric=pearsonDistance,method='ward',cmap='plasma')
            plt.savefig('Outputs/' + experiment_name + '/visualizations/rulepop/rulepopClusterMapPearson_' + str(cv) + '.png')
        else:
            #When using this, make sure you have no individual vectors that have all equal values
            seaborn.clustermap(rule_df, metric='correlation',method='ward',cmap='plasma')
            plt.savefig('Outputs/' + experiment_name + '/visualizations/rulepop/rulepopClusterMapDefault_' + str(cv) + '.png')
        plt.close('all')

        #Rule Specificity Network#######################################################################################
        attribute_acc_specificity_counts = model.get_final_attribute_specificity_list()
        acc_spec_dict = {}
        for header_index in range(len(data_headers)):
            acc_spec_dict[data_headers[header_index]] = attribute_acc_specificity_counts[header_index]
        attribute_cooccurrences = model.get_final_attribute_coocurrences(data_headers, len(data_headers))

        G = nx.Graph()
        edge_list = []
        weight_list = []
        for co in attribute_cooccurrences:
            G.add_edge(co[0], co[1], weight=co[3])
            edge_list.append((co[0], co[1]))
            weight_list.append(co[3])

        pos = nx.spring_layout(G, k=1)

        max_node_value = max(acc_spec_dict.values())
        for i in acc_spec_dict:
            acc_spec_dict[i] = acc_spec_dict[i] / max_node_value * 1000

        max_weight_value = max(weight_list)
        for i in range(len(weight_list)):
            weight_list[i] = weight_list[i] / max_weight_value * 10

        nx.draw_networkx_nodes(G, pos, nodelist=acc_spec_dict.keys(), node_size=[v * 1 for v in acc_spec_dict.values()],
                               node_color='#FF3377')
        nx.draw_networkx_edges(G, pos, edge_color='#E0B8FF', edgelist=edge_list, width=[v * 1 for v in weight_list])
        nx.draw_networkx_labels(G, pos)
        plt.axis('off')
        plt.savefig('Outputs/' + experiment_name + '/visualizations/rulepop/rulepopGraph_' + str(cv) + '.png', dpi=300)
        plt.close('all')

        #AT Scores Heatmap##############################################################################################
        # Get AT Scores for each instance
        AT_scores = model.get_attribute_tracking_scores(instance_labels=np.array(train_instance_labels))

        # Normalize AT Scores and convert to np.ndarray and plot heatmap
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

        AT_full_df = pd.DataFrame(normalized_AT_scores, columns=data_headers, index=train_instance_labels)
        seaborn.heatmap(AT_full_df, cmap='plasma')
        plt.savefig('Outputs/' + experiment_name + '/visualizations/at/ATHeatmap_' + str(cv) + '.png')
        plt.close('all')

        #AT Score Clustermap############################################################################################
        g = seaborn.clustermap(AT_full_df, metric=pearsonDistance, method='ward', cmap='plasma')
        cluster_tree = HClust.createClusterTree(g.dendrogram_row.linkage, train_instance_labels, AT_full_df.to_numpy())
        clusters = cluster_tree.getSignificantClusters(p_value=0.05, sample_count=100, metric='correlation',method='ward')
        color_dict = {}
        for cluster in clusters:
            random_color = randomHex()
            for inst_label in cluster:
                color_dict[inst_label] = random_color
        color_list = pd.Series(dict(sorted(color_dict.items())))

        group_dict = {}
        for i in range(len(train_group_labels)):
            if train_group_labels[i] in group_dict:
                group_dict[train_group_labels[i]].append(train_instance_labels[i])
            else:
                group_dict[train_group_labels[i]] = [train_instance_labels[i]]

        group_color_dict = {}
        for group in group_dict:
            random_color = group_colors[group]
            for inst_label in group_dict[group]:
                group_color_dict[inst_label] = random_color
        group_list = pd.Series(dict(sorted(group_color_dict.items())))

        combo_list = pd.concat([group_list,color_list],axis=1)

        g = seaborn.clustermap(AT_full_df, row_linkage=g.dendrogram_row.linkage, col_linkage=g.dendrogram_col.linkage,row_colors=combo_list, cmap='plasma')
        plt.savefig('Outputs/' + experiment_name + '/visualizations/at/ATClustermap_' + str(cv) + '.png', dpi=300)
        plt.close('all')

        #Export .csv of AT cluster makeup and specificity and penetrance################################################


        #Cluster Test Instances#########################################################################################
        #Pair test instances with a prediction
        test_predictions = model.predict(test_data_features)
        predicted_instances = np.insert(test_data_features,num_attributes,test_predictions,1)

        #Get Test Instance AT Scores: Eventually build this out to be a part of the package
        test_attribute_tracking_sums = [[0]*num_attributes for i in range(len(predicted_instances))]
        at_counter = 0
        for instance in predicted_instances:
            state = instance[:-1]
            model.population.makeEvalMatchSet(model,state)
            correct_set = []
            for matched_rule_index in model.population.matchSet:
                if model.population.popSet[matched_rule_index].phenotype == instance[-1]:
                    correct_set.append(matched_rule_index)
            model.population.clearSets()

            #external AT update
            if attribute_tracking_method == 'add':
                for ref in correct_set:
                    for each in model.population.popSet[ref].specifiedAttList:
                        test_attribute_tracking_sums[at_counter][each] += model.population.popSet[ref].accuracy
            elif attribute_tracking_method == 'wh':
                temp_att_track = [0] * num_attributes
                for ref in correct_set:
                    for each in model.population.popSet[ref].specifiedAttList:
                        temp_att_track[each] += model.population.popSet[ref].accuracy
                for attribute_index in range(len(temp_att_track)):
                    test_attribute_tracking_sums[at_counter][attribute_index] += model.attribute_tracking_beta * (temp_att_track[attribute_index] - test_attribute_tracking_sums[at_counter][attribute_index])
            at_counter += 1

        #Get AT scores in return format
        ret_list = []
        for i in range(len(predicted_instances)):
            ret_list.append([test_instance_labels[i],test_attribute_tracking_sums[i]])

        # Normalize Test AT Scores and convert to np.ndarray
        normalized_test_AT_scores = []
        for i in range(len(ret_list)):
            normalized = ret_list[i][1]
            max_score = max(normalized)
            for j in range(len(normalized)):
                if max_score != 0:
                    normalized[j] /= max_score
                else:
                    normalized[j] = 0
            normalized_test_AT_scores.append(np.array(normalized))
        normalized_test_AT_scores = np.array(normalized_test_AT_scores)

        AT_test_df = pd.DataFrame(normalized_test_AT_scores, columns=data_headers, index=test_instance_labels)

        # Cluster Test Instances Into Original Clusters and Export Significant Cluster##################################
        test_clusters = {}
        for i in range(len(test_instance_labels)):
            test_instance_label = test_instance_labels[i]
            test_score = normalized_test_AT_scores[i]
            color = get_closest_cluster_color(clusters,normalized_AT_scores,train_instance_labels,test_score,color_dict)
            if color in test_clusters:
                test_clusters[color].append(test_instance_label)
            else:
                test_clusters[color] = [test_instance_label]

        with open('Outputs/' + experiment_name + '/visualizations/at/ATClusters_' + str(cv) + '.csv',mode='w') as file:
            writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            for cluster in clusters:
                exp_color = color_dict[cluster[0]]
                writer.writerow(['ClusterID: '+exp_color])
                writer.writerow(['Instance Label','True Group','Test/Train','']+list(data_headers)+['Actual '+class_label,'Predicted '+class_label])
                for exp_traininstance in cluster:
                    exp_index = train_instance_labels.index(exp_traininstance)
                    exp_traingroup = train_group_labels[exp_index]
                    exp_state = train_data_features[exp_index].tolist()
                    exp_phenotype = train_data_phenotypes[exp_index]
                    writer.writerow([exp_traininstance,exp_traingroup,'Train','']+exp_state+[exp_phenotype,'N/A'])
                if exp_color in test_clusters:
                    for exp_testinstance in test_clusters[exp_color]:
                        exp_index = test_instance_labels.index(exp_testinstance)
                        exp_testgroup = test_group_labels[exp_index]
                        exp_state = test_data_features[exp_index].tolist()
                        exp_phenotype = [test_data_phenotypes[exp_index],test_predictions[exp_index]]
                        writer.writerow([exp_testinstance, exp_testgroup, 'Test', ''] + exp_state + exp_phenotype)
                writer.writerow([])

        # Cluster Test Instances Into New Clusters and Export###########################################################
        # Merge test AT scores with train AT scores
        AT_new_df = pd.concat([AT_test_df,AT_full_df],ignore_index=False)
        complete_instance_labels = test_instance_labels+train_instance_labels
        complete_group_labels = test_group_labels+train_group_labels

        #Generate Clustermap
        g = seaborn.clustermap(AT_new_df, metric=pearsonDistance, method='ward', cmap='plasma')
        cluster_tree = HClust.createClusterTree(g.dendrogram_row.linkage, complete_instance_labels, AT_new_df.to_numpy())
        clusters = cluster_tree.getSignificantClusters(p_value=0.05, sample_count=100, metric='correlation',method='ward')

        color_dict = {}
        for cluster in clusters:
            random_color = randomHex()
            for inst_label in cluster:
                color_dict[inst_label] = random_color
        color_list = pd.Series(dict(sorted(color_dict.items())))

        group_dict = {}
        for i in range(len(complete_group_labels)):
            if complete_group_labels[i] in group_dict:
                group_dict[complete_group_labels[i]].append(complete_instance_labels[i])
            else:
                group_dict[complete_group_labels[i]] = [complete_instance_labels[i]]

        group_color_dict = {}
        for group in group_dict:
            random_color = group_colors[group]
            for inst_label in group_dict[group]:
                group_color_dict[inst_label] = random_color
        group_list = pd.Series(dict(sorted(group_color_dict.items())))

        type_color_dict = {}
        for inst_label in test_instance_labels:
            type_color_dict[inst_label] = 'yellow'
        for inst_label in train_instance_labels:
            type_color_dict[inst_label] = 'black'
        type_list = pd.Series(dict(sorted(type_color_dict.items())))

        combo_list = pd.concat([type_list,group_list, color_list], axis=1)

        g = seaborn.clustermap(AT_new_df, row_linkage=g.dendrogram_row.linkage, col_linkage=g.dendrogram_col.linkage,row_colors=combo_list, cmap='plasma')
        plt.savefig('Outputs/' + experiment_name + '/visualizations/at/ATFullClustermap_' + str(cv) + '.png', dpi=300)
        plt.close('all')

        with open('Outputs/' + experiment_name + '/visualizations/at/ATreClusters_' + str(cv) + '.csv',mode='w') as file:
            writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            for cluster in clusters:
                exp_color = color_dict[cluster[0]]
                writer.writerow(['ClusterID: '+exp_color])
                writer.writerow(['Instance Label','True Group','Test/Train','']+list(data_headers)+['Actual '+class_label,'Predicted '+class_label])
                for exp_instance in cluster:
                    if exp_instance in train_instance_labels:
                        exp_index = train_instance_labels.index(exp_instance)
                        exp_traingroup = train_group_labels[exp_index]
                        exp_state = train_data_features[exp_index].tolist()
                        exp_phenotype = train_data_phenotypes[exp_index]
                        writer.writerow([exp_instance,exp_traingroup,'Train','']+exp_state+[exp_phenotype,'N/A'])

                for exp_instance in cluster:
                    if exp_instance in test_instance_labels:
                        exp_index = test_instance_labels.index(exp_instance)
                        exp_testgroup = test_group_labels[exp_index]
                        exp_state = test_data_features[exp_index].tolist()
                        exp_phenotype = [test_data_phenotypes[exp_index],test_predictions[exp_index]]
                        writer.writerow([exp_instance,exp_testgroup,'Test','']+exp_state+exp_phenotype)
                writer.writerow([])



# Helper Methods #######################################################################################################
def get_closest_cluster_color(cluster_labels,cluster_instances,train_instances,test_instance,color_dict):
    '''
    :param cluster_labels: list of lists: sublist is labels of instances in a cluster
    :param cluster_instances: numpy array of instances, where rows are instances, in order of train_instances
    :param train_instances: list of train labels
    :param test_instance: numpy array of the instance to be compared to
    :param color_dict: {instance_label:color,...} of original clusters
    :return: hex code of closest cluster color
    #Uses a naive average pearson distance
    '''
    distances = []
    for cluster in cluster_labels:
        avg_distance = 0
        for inst_label in cluster:
            index = train_instances.index(inst_label)
            cluster_instance = cluster_instances[index]
            avg_distance += pearsonDistance(test_instance,cluster_instance)
        avg_distance /= len(cluster)
        distances.append(avg_distance)

    #Return color of arbitrary instance in closest cluster
    return color_dict[cluster_labels[distances.index(min(distances))][0]]

def cv_partitioner(td, cv_partitions, outcomeLabel, randomSeed):
    # Shuffle instances to avoid potential biases
    td = td.sample(frac=1, random_state=randomSeed).reset_index(drop=True)

    # Temporarily convert data frame to list of lists (save header for later)
    header = list(td.columns.values)
    datasetList = list(list(x) for x in zip(*(td[x].values.tolist() for x in td.columns)))

    # Handle Special Variables for Nominal Outcomes
    outcomeIndex = td.columns.get_loc(outcomeLabel)
    classList = []
    for each in datasetList:
        if each[outcomeIndex] not in classList:
            classList.append(each[outcomeIndex])

    # Initialize partitions
    partList = []
    for x in range(cv_partitions):
        partList.append([])

    # Stratified Partitioning Method-----------------------
    byClassRows = [[] for i in range(len(classList))]  # create list of empty lists (one for each class)
    for row in datasetList:
        # find index in classList corresponding to the class of the current row.
        cIndex = classList.index(row[outcomeIndex])
        byClassRows[cIndex].append(row)

    for classSet in byClassRows:
        currPart = 0
        counter = 0
        for row in classSet:
            partList[currPart].append(row)
            counter += 1
            currPart = counter % cv_partitions

    train_dfs = []
    test_dfs = []
    for part in range(0, cv_partitions):
        testList = partList[part]  # Assign testing set as the current partition

        trainList = []
        tempList = []
        for x in range(0, cv_partitions):
            tempList.append(x)
        tempList.pop(part)

        for v in tempList:  # for each training partition
            trainList.extend(partList[v])

        train_dfs.append(pd.DataFrame(trainList, columns=header))
        test_dfs.append(pd.DataFrame(testList, columns=header))

    return train_dfs, test_dfs

def spearmanDistance(u,v):
    if len(set(u)) == 1 and len(set(v)) == 1 and u[0] == v[0]: #Prevent NaN values
        return 0
    elif len(set(u)) == 1 and len(set(v)) == 1 and u[0] != v[0]:
        return 1
    elif len(set(u)) == 1 or len(set(v)) == 1:
        return euclidean(u,v)/math.sqrt(len(v)) #normalized euclidean distance
    return 1 - spearmanr(u,v)[0]

def pearsonDistance(u,v):
    if len(set(u)) == 1 and len(set(v)) == 1 and u[0] == v[0]:  # Prevent NaN values
        return 0
    elif len(set(u)) == 1 and len(set(v)) == 1 and u[0] != v[0]:
        return 1
    elif len(set(u)) == 1 or len(set(v)) == 1:
        return euclidean(u, v) / math.sqrt(len(v))  # normalized euclidean distance
    return 1 - pearsonr(u,v)[0]

def randomHex():
    s = '#'
    for i in range(6):
        s+=random.choice(['0','1','2','3','4','5','6','7','8','9','A','B','C','D','E','F'])
    return s

if __name__ == "__main__":
    sys.exit(main(sys.argv))