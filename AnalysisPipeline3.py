import sys
import os
import argparse
import time
import random
import pandas as pd
import numpy as np
import copy
import pickle
from scipy.stats import pearsonr,spearmanr
from scipy.spatial.distance import euclidean
from skExSTraCS import ExSTraCS
import math
import csv
import seaborn
import HClust
import networkx as nx
import matplotlib.pyplot as plt
from statistics import mean

'''
Sample Run Commands:
#MP6 problem
python AnalysisPipeline3.py --d Datasets/mp6_full.csv --o Outputs --e mp6v3 --inst Instance --group Group --iter 10000 --N 500 --nu 10

#MP11 problem
python AnalysisPipeline3.py --d Datasets/mp11_full.csv --o Outputs --e mp11v3 --inst Instance --group Group --iter 10000 --N 1000 --nu 10

#MP20 problem
python AnalysisPipeline3.py --d Datasets/mp20_full.csv --o Outputs --e mp20v3 --inst Instance --group Group --iter 30000 --N 2000 --nu 10

#rule 1
python AnalysisPipeline3.py --d Datasets/one.txt --o Outputs --e onev3 --group Model --iter 20000 --N 1000 --nu 1
'''

def main(argv):
    # Parse arguments
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--d', dest='data_path', type=str, help='path to directory containing datasets')
    parser.add_argument('--o', dest='output_path', type=str, help='path to output directory')
    parser.add_argument('--e', dest='experiment_name', type=str, help='name of experiment (no spaces)')

    parser.add_argument('--class', dest='class_label', type=str, default="Class")
    parser.add_argument('--inst', dest='instance_label', type=str, default="None")
    parser.add_argument('--group', dest='group_label', type=str, default="None")

    parser.add_argument('--cv', dest='cv_partitions', type=int, help='number of CV partitions', default=3)

    parser.add_argument('--iter', dest='learning_iterations', type=int, default=16000)
    parser.add_argument('--N', dest='N', type=int, default=1000)
    parser.add_argument('--nu', dest='nu', type=int, default=1)
    parser.add_argument('--at-method', dest='attribute_tracking_method', type=str, default='wh')
    parser.add_argument('--random-state',dest='random_state',type=str,default='None')

    parser.add_argument('--rulepop-method', dest='rulepop_clustering_method', type=str, default='pearson')

    options = parser.parse_args(argv[1:])
    data_path = options.data_path
    output_path = options.output_path
    experiment_name = options.experiment_name
    if options.class_label == 'None':
        class_label = None
    else:
        class_label = options.class_label
    if options.instance_label == 'None':
        instance_label = None
    else:
        instance_label = options.instance_label
    if options.group_label == 'None':
        group_label = None
    else:
        group_label = options.group_label
    cv_count = options.cv_partitions
    learning_iterations = options.learning_iterations
    N = options.N
    nu = options.nu
    attribute_tracking_method = options.attribute_tracking_method
    if options.random_state == 'None':
        random_state = random.randint(0, 1000000)
    else:
        random_state = options.random_state
    rulepop_clustering_method = options.rulepop_clustering_method

    # Create experiment folders and check path validity
    if not os.path.exists(data_path):
        raise Exception("Provided data_path does not exist")

    for char in experiment_name:
        if not char in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890':
            raise Exception('Experiment Name must be alphanumeric')

    experiment_path = output_path + '/' + experiment_name
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    if not os.path.exists(experiment_path):
        os.mkdir(experiment_path)

    # Write Metadata
    outfile = open(experiment_path + '/metadata', mode='w')
    outfile.write('data_path: ' + str(data_path) + '\n')
    outfile.write('learning_iterations: ' + str(learning_iterations) + '\n')
    outfile.write('N: ' + str(N) + '\n')
    outfile.write('nu: ' + str(nu) + '\n')
    outfile.write('attribute_tracking_method: ' + str(attribute_tracking_method) + '\n')
    outfile.write('rulepop_clustering_method: ' + str(rulepop_clustering_method) + '\n')
    outfile.close()

    # Read in data
    if data_path[-1] == 't':  # txt
        dataset = pd.read_csv(data_path, sep='\t')
    elif data_path[-1] == 'v':  # csv
        dataset = pd.read_csv(data_path, sep=',')
    elif data_path[-1] == 'z':  # txt.gz
        dataset = pd.read_csv(data_path, sep='\t', compression='gzip')
    else:
        raise Exception('Unrecognized File Type')

    # Random seed
    random.seed(random_state)
    np.random.seed(random_state)

    # Remove 'unnamed columns'
    dataset = dataset.loc[:, ~dataset.columns.str.contains('^Unnamed')]

    # Add Instance and Group Label Columns if necessary
    visualize_true_clusters = True
    if instance_label == None:
        dataset = dataset.assign(instance=np.array(list(range(dataset.values.shape[0]))))
    if group_label == None:
        dataset = dataset.assign(group=np.ones(dataset.values.shape[0]))
        visualize_true_clusters = False

    train_dfs, test_dfs = cv_partitioner(dataset, cv_count, class_label, random_state)
    ####################################################################################################################
    # Create cv information
    cv_info = []
    tt_inst = []
    for cv in range(cv_count):
        if instance_label != None and group_label != None:
            train_dfs[cv].set_index([instance_label, group_label], inplace=True)
            test_dfs[cv].set_index([instance_label, group_label], inplace=True)
            use_group_label = group_label
            use_inst_label = instance_label
        elif instance_label != None and group_label == None:
            train_dfs[cv].set_index([instance_label, 'group'], inplace=True)
            test_dfs[cv].set_index([instance_label, 'group'], inplace=True)
            use_group_label = 'group'
            use_inst_label = instance_label
        elif instance_label == None and group_label != None:
            train_dfs[cv].set_index([group_label, 'instance'], inplace=True)
            test_dfs[cv].set_index([group_label, 'instance'], inplace=True)
            use_group_label = group_label
            use_inst_label = 'instance'
        else:
            train_dfs[cv].set_index(['group', 'instance'], inplace=True)
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

        cv_info.append([train_data_features,train_data_phenotypes,train_instance_labels,train_group_labels,test_data_features,test_data_phenotypes,test_instance_labels,test_group_labels,use_inst_label,use_group_label])
        tt_inst += train_instance_labels

    #Group Colors and Instance Labels
    full_df = copy.deepcopy(dataset)
    if instance_label != None and group_label != None:
        full_df.set_index([instance_label, group_label], inplace=True)
        use_group_label = group_label
        use_inst_label = instance_label
    elif instance_label != None and group_label == None:
        full_df = full_df.assign(group=np.ones(full_df.values.shape[0]))
        full_df.set_index([instance_label, 'group'], inplace=True)
        use_group_label = 'group'
        use_inst_label = instance_label
    elif instance_label == None and group_label != None:
        full_df = full_df.assign(instance=np.array(list(range(full_df.values.shape[0]))))
        full_df.set_index([group_label, 'instance'], inplace=True)
        use_group_label = group_label
        use_inst_label = 'instance'
    else:
        full_df = full_df.assign(instance=np.array(list(range(full_df.values.shape[0]))))
        full_df = full_df.assign(group=np.ones(full_df.values.shape[0]))
        full_df.set_index(['group', 'instance'], inplace=True)
        use_group_label = 'group'
        use_inst_label = 'instance'

    data_features = full_df.drop(class_label, axis=1).values
    data_phenotypes = full_df[class_label].values
    data_headers = full_df.drop(class_label, axis=1).columns.values
    full_instance_labels = full_df.index.get_level_values(use_inst_label).tolist()
    full_group_labels = full_df.index.get_level_values(use_group_label).tolist()
    group_colors = {}
    for group_name in set(full_group_labels):
        random_color = randomHex()
        group_colors[group_name] = random_color

    ####################################################################################################################
    #Run CVs
    for cv in range(cv_count):
        train_data_features = cv_info[cv][0]
        train_data_phenotypes = cv_info[cv][1]
        train_instance_labels = cv_info[cv][2]
        train_group_labels = cv_info[cv][3]
        test_data_features = cv_info[cv][4]
        test_data_phenotypes = cv_info[cv][5]
        test_instance_labels = cv_info[cv][6]
        test_group_labels = cv_info[cv][7]
        inst_label = cv_info[cv][8]
        group_label = cv_info[cv][9]

        # Create CV directory
        if not os.path.exists(experiment_path + '/CV_' + str(cv)):
            os.mkdir(experiment_path + '/CV_' + str(cv))

        # Train ExSTraCS Model
        model = ExSTraCS(learning_iterations=learning_iterations, N=N, nu=nu,attribute_tracking_method=attribute_tracking_method, rule_compaction=None,random_state=random_state)
        model.fit(train_data_features, train_data_phenotypes)

        outfile = open(experiment_path + '/CV_' + str(cv) + '/model', 'wb')
        pickle.dump(model, outfile)
        outfile.close()

        # Export Testing Accuracy for each instance
        predicted_data_phenotypes = model.predict(test_data_features)
        equality = np.equal(predicted_data_phenotypes,test_data_phenotypes)
        with open(experiment_path + '/CV_' + str(cv) + '/instTestingAccuracy.csv', mode='w') as file:
            writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            writer.writerow([inst_label,'isCorrect'])
            for i in range(len(test_instance_labels)):
                writer.writerow([test_instance_labels[i], 1 if equality[i] else 0])
        file.close()

        # Export Aggregate Testing Accuracy
        outfile = open(experiment_path + '/CV_' + str(cv) + '/testingAccuracy.txt', mode='w')
        outfile.write(str(model.score(test_data_features, test_data_phenotypes)))
        outfile.close()

        # Save train and testing datasets into csvs
        with open(experiment_path + '/CV_' + str(cv) + '/trainDataset.csv', mode='w') as file:
            writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(list(data_headers) + [class_label, inst_label, group_label])
            for i in range(len(train_instance_labels)):
                writer.writerow(list(train_data_features[i]) + [train_data_phenotypes[i]] + [train_instance_labels[i]] + [train_group_labels[i]])
        file.close()

        with open(experiment_path + '/CV_' + str(cv) + '/testDataset.csv', mode='w') as file:
            writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(list(data_headers) + [class_label, inst_label, group_label])
            for i in range(len(test_instance_labels)):
                writer.writerow(list(test_data_features[i]) + [test_data_phenotypes[i]] + [test_instance_labels[i]] + [test_group_labels[i]])
        file.close()

        # Get AT Scores for each instance
        AT_scores = model.get_attribute_tracking_scores(instance_labels=np.array(train_instance_labels))

        # Normalize AT Scores
        normalized_AT_scores = []
        for i in range(len(AT_scores)):
            normalized = AT_scores[i][1]
            max_score = max(normalized)
            for j in range(len(normalized)):
                if max_score != 0:
                    normalized[j] /= max_score
                else:
                    normalized[j] = 0
            normalized_AT_scores.append(list(normalized))

        # Save Normalized AT Scores
        with open(experiment_path + '/CV_' + str(cv) + '/normalizedATScores.csv', mode='w') as file:
            writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            writer.writerow([inst_label]+list(data_headers))
            for i in range(len(train_instance_labels)):
                writer.writerow([train_instance_labels[i]]+normalized_AT_scores[i])
        file.close()

        # Plot CV unlabelled clustermap
        AT_full_df_cv = pd.DataFrame(normalized_AT_scores, columns=data_headers, index=train_instance_labels)
        try:
            g = seaborn.clustermap(AT_full_df_cv, metric='correlation', method='ward', cmap='plasma')
        except:
            print('AT Clustermap default pearson failed. Trying slower own Pearson method instead')
            g = seaborn.clustermap(AT_full_df_cv, metric=pearsonDistance, method='ward', cmap='plasma')

        g = seaborn.clustermap(AT_full_df_cv, row_linkage=g.dendrogram_row.linkage, col_linkage=g.dendrogram_col.linkage, cmap='plasma')
        plt.savefig(experiment_path + '/CV_' + str(cv) + '/ATclustermap.png',dpi=300)
        plt.close('all')

    ####################################################################################################################
    #CV Composite Analysis
    if not os.path.exists(experiment_path + '/Composite'):
        os.mkdir(experiment_path + '/Composite')
        os.mkdir(experiment_path + '/Composite/rulepop')
        os.mkdir(experiment_path + '/Composite/rulepop/ruleclusters')
        os.mkdir(experiment_path + '/Composite/at')
        os.mkdir(experiment_path + '/Composite/at/atclusters')

    ####################################################################################################################
    # Merge AT scores and create heatmap
    merged_AT_dict = {}
    merged_AT_dict_count = {}
    for cv in range(cv_count):
        inst_label = cv_info[cv][8]
        partial_AT_scores = pd.read_csv(experiment_path + '/CV_' + str(cv) + '/normalizedATScores.csv')
        partial_scores = partial_AT_scores.drop(inst_label,axis=1).values
        partial_labels = partial_AT_scores[inst_label].values
        for i in range(len(partial_labels)):
            if partial_labels[i] in merged_AT_dict:
                merged_AT_dict[partial_labels[i]] += partial_scores[i]
                merged_AT_dict_count[partial_labels[i]] += 1
            else:
                merged_AT_dict[partial_labels[i]] = partial_scores[i]
                merged_AT_dict_count[partial_labels[i]] = 1
    merged_AT = []
    for label in full_instance_labels:
        merged_AT.append(merged_AT_dict[label]/merged_AT_dict_count[label]) #renormalize
    merged_AT = np.array(merged_AT)
    AT_full_df = pd.DataFrame(merged_AT, columns=data_headers, index=full_instance_labels)

    seaborn.heatmap(AT_full_df, cmap='plasma')
    plt.savefig(experiment_path + '/Composite/at/ATHeatmap.png')
    plt.close('all')

    # Merge Instance Test scores
    merged_test_dict = {}
    merged_test_dict_count = {}
    for cv in range(cv_count):
        inst_label = cv_info[cv][8]
        partial_test_scores = pd.read_csv(experiment_path + '/CV_' + str(cv) + '/instTestingAccuracy.csv')
        partial_scores = partial_test_scores.drop(inst_label,axis=1).values
        partial_labels = partial_test_scores[inst_label].values
        for i in range(len(partial_labels)):
            if partial_labels[i] in merged_test_dict:
                merged_test_dict[partial_labels[i]] += partial_scores[i][0]
                merged_test_dict_count[partial_labels[i]] += 1
            else:
                merged_test_dict[partial_labels[i]] = partial_scores[i][0]
                merged_test_dict_count[partial_labels[i]] = 1
    merged_test = []
    for label in full_instance_labels:
        merged_test.append(merged_test_dict[label]/merged_test_dict_count[label]) #renormalize
    merged_test = np.array(merged_test)

    #AT Clustermaps and CSV Analysis
    try:
        g = seaborn.clustermap(AT_full_df, metric='correlation', method='ward', cmap='plasma')
    except:
        print('AT Clustermap default pearson failed. Trying slower own Pearson method instead')
        g = seaborn.clustermap(AT_full_df, metric=pearsonDistance, method='ward', cmap='plasma')

    cluster_tree = HClust.createClusterTree(g.dendrogram_row.linkage, full_instance_labels, AT_full_df.to_numpy())
    clusters, colors = cluster_tree.getSignificantClusters(p_value=0.05, sample_count=100, metric='correlation',method='ward', random_state=random_state)

    for cluster_count in reversed(range(1, len(clusters) + 1)):
        if not os.path.exists(experiment_path + '/Composite/at/atclusters/' + str(cluster_count) + '_clusters'):
            os.mkdir(experiment_path + '/Composite/at/atclusters/' + str(cluster_count) + '_clusters')

        subclusters, colors = cluster_tree.getNSignificantClusters(cluster_count, p_value=0.05, sample_count=100, metric='correlation', method='ward',random_state=random_state)

        # Clustermaps
        color_dict = {}
        color_count = 0
        for cluster in subclusters:
            random_color = colors[color_count]
            for inst_label in cluster:
                color_dict[inst_label] = random_color
            color_count += 1
        color_list = pd.Series(dict(sorted(color_dict.items())))

        group_dict = {}
        for i in range(len(full_group_labels)):
            if full_group_labels[i] in group_dict:
                group_dict[full_group_labels[i]].append(full_instance_labels[i])
            else:
                group_dict[full_group_labels[i]] = [full_instance_labels[i]]

        group_color_dict = {}
        for group in group_dict:
            random_color = group_colors[group]
            for inst_label in group_dict[group]:
                group_color_dict[inst_label] = random_color
        group_list = pd.Series(dict(sorted(group_color_dict.items())))

        if visualize_true_clusters:
            combo_list = pd.concat([group_list, color_list], axis=1)
            combo_list.columns = ['True Clusters','Found Clusters']
        else:
            combo_list = pd.Series.to_frame(color_list)
            combo_list.columns = ['Found Clusters']

        g = seaborn.clustermap(AT_full_df, row_linkage=g.dendrogram_row.linkage, col_linkage=g.dendrogram_col.linkage,row_colors=combo_list, cmap='plasma')
        plt.savefig(experiment_path + '/Composite/at/atclusters/' + str(cluster_count) + '_clusters/ATclustermap.png',dpi=300)
        plt.close('all')

        with open(experiment_path + '/Composite/at/atclusters/' + str(cluster_count) + '_clusters/labeledDataset.csv',mode='w') as file:
            writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(['Instance','Cluster'] + list(data_headers) + [class_label])
            for cluster in subclusters:
                exp_color = color_dict[cluster[0]]
                for exp_instance in cluster:
                    exp_index = full_instance_labels.index(exp_instance)
                    exp_state = data_features[exp_index].tolist()
                    exp_phenotype = data_phenotypes[exp_index]
                    writer.writerow([exp_instance, exp_color] + exp_state + [exp_phenotype])
        file.close()

        with open(experiment_path + '/Composite/at/atclusters/' + str(cluster_count) + '_clusters/clusters.csv', mode='w') as file:
            writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            for cluster in subclusters:
                exp_color = color_dict[cluster[0]]
                writer.writerow(['ClusterID: ' + exp_color])
                writer.writerow(['Instance Label', 'True Group', ''] + list(data_headers) + [class_label])
                at_sums = np.array([0.0]*len(data_headers))
                group_makeup = {}
                class_makeup = {}

                test_score_sum = 0
                for exp_instance in cluster:
                    exp_index = full_instance_labels.index(exp_instance)
                    exp_group = full_group_labels[exp_index]
                    exp_state = data_features[exp_index].tolist()
                    exp_phenotype = data_phenotypes[exp_index]
                    writer.writerow([exp_instance, exp_group, ''] + exp_state + [exp_phenotype])
                    test_score_sum += merged_test[exp_index]

                    at_sums += merged_AT[exp_index]
                    if exp_group in group_makeup:
                        group_makeup[exp_group] += 1/len(cluster)
                    else:
                        group_makeup[exp_group] = 1/len(cluster)
                    if exp_phenotype in class_makeup:
                        class_makeup[exp_phenotype] += 1/len(cluster)
                    else:
                        class_makeup[exp_phenotype] = 1/len(cluster)

                writer.writerow(["Cluster Testing Accuracy: " + str(test_score_sum / len(cluster))])

                writer.writerow(['AT Sums:'])
                ks = []
                vs = []
                for k, v in sorted(dict(zip(list(data_headers),list(at_sums))).items(), key=lambda item: item[1]):
                    ks.append(k)
                    vs.append(v)
                writer.writerow(list(reversed(ks)))
                writer.writerow(list(reversed(vs)))

                writer.writerow(['True Group Composition:'])
                ks = []
                vs = []
                for k,v in sorted(group_makeup.items(), key=lambda item: item[1]):
                    ks.append(k)
                    vs.append(v)
                writer.writerow(list(reversed(ks)))
                writer.writerow(list(reversed(vs)))

                writer.writerow(['True Class Composition:'])
                ks = []
                vs = []
                for k, v in sorted(class_makeup.items(), key=lambda item: item[1]):
                    ks.append(k)
                    vs.append(v)
                writer.writerow(list(reversed(ks)))
                writer.writerow(list(reversed(vs)))

                writer.writerow([])
        file.close()
    ####################################################################################################################
    # Merge Rule Population
    merged_population = []
    models = []
    for cv in range(cv_count):
        file = open(experiment_path + '/CV_' + str(cv) + '/model', 'rb')
        model = pickle.load(file)
        models.append(model)
        for rule in model.population.popSet:
            shouldAdd = True
            for merged_rule in merged_population:
                if rule.equals(merged_rule):
                    shouldAdd = False
                    if not isinstance(merged_rule.numerosity,list):
                        merged_rule.numerosity = [merged_rule.numerosity,rule.numerosity]
                        merged_rule.accuracy = [merged_rule.accuracy,rule.accuracy]
                        merged_rule.initTimeStamp = [merged_rule.initTimeStamp, rule.initTimeStamp]
                    else:
                        merged_rule.numerosity.append(rule.numerosity)
                        merged_rule.accuracy.append(rule.accuracy)
                        merged_rule.initTimeStamp.append(rule.initTimeStamp)
            if shouldAdd:
                merged_population.append(rule)

    for rule in merged_population:
        if isinstance(rule.numerosity,list):
            rule.numerosity = int(mean(rule.numerosity))
            rule.accuracy = mean(rule.accuracy)
            rule.initTimeStamp = int(mean(rule.initTimeStamp))

    num_rules = len(merged_population)
    rule_specificity_array = []
    for inst in range(num_rules):
        a = []
        for attribute in range(len(data_headers)):
            a.append(0)
        rule_specificity_array.append(a)

    rule_index_count = 0
    for classifier in merged_population:
        for i in classifier.specifiedAttList:
            rule_specificity_array[rule_index_count][i] = 1
        rule_index_count += 1

    rule_df = pd.DataFrame(rule_specificity_array, columns=data_headers, index=list(range(num_rules)))

    seaborn.heatmap(rule_df, cmap='plasma')
    plt.savefig(experiment_path + '/Composite/rulepop/rulepopHeatmap.png')
    plt.close('all')

    # Rule Population Clustermaps
    if rulepop_clustering_method == 'spearman':
        metric = spearmanDistance
    elif rulepop_clustering_method == 'pearson':
        metric = pearsonDistance
    else:
        metric = 'correlation'

    if rulepop_clustering_method == 'pearson':
        try:
            r = seaborn.clustermap(rule_df, metric='correlation', method='ward', cmap='plasma')
        except:
            r = seaborn.clustermap(rule_df, metric=metric, method='ward', cmap='plasma')
    else:
        r = seaborn.clustermap(rule_df, metric=metric, method='ward', cmap='plasma')
    rule_cluster_tree = HClust.createClusterTree(r.dendrogram_row.linkage, list(range(num_rules)), rule_df.to_numpy())

    if rulepop_clustering_method == 'pearson':
        try:
            rule_clusters, rule_colors = rule_cluster_tree.getSignificantClusters(p_value=0.05, sample_count=100, metric='correlation',method='ward', random_state=random_state)
        except:
            rule_clusters, rule_colors = rule_cluster_tree.getSignificantClusters(p_value=0.05, sample_count=100, metric=metric,method='ward', random_state=random_state)
    else:
        rule_clusters, rule_colors = rule_cluster_tree.getSignificantClusters(p_value=0.05, sample_count=100,metric=metric, method='ward',random_state=random_state)

    for rule_cluster_count in reversed(range(1, len(rule_clusters) + 1)):
        if not os.path.exists(experiment_path + '/Composite/rulepop/ruleclusters/' + str(rule_cluster_count) + '_clusters'):
            os.mkdir(experiment_path + '/Composite/rulepop/ruleclusters/' + str(rule_cluster_count) + '_clusters')

        rule_subclusters, rule_colors = rule_cluster_tree.getNSignificantClusters(rule_cluster_count, p_value=0.05,sample_count=100, metric=metric,method='ward',random_state=random_state)
        rule_color_dict = {}
        rule_color_count = 0
        for cluster in rule_subclusters:
            random_color = rule_colors[rule_color_count]
            for inst_label in cluster:
                rule_color_dict[inst_label] = random_color
            rule_color_count += 1
        rule_color_list = pd.Series(dict(sorted(rule_color_dict.items())))
        rule_color_list = pd.Series.to_frame(rule_color_list)
        rule_color_list.columns = ['Found Clusters']

        seaborn.clustermap(rule_df, row_linkage=r.dendrogram_row.linkage, col_linkage=r.dendrogram_col.linkage,row_colors=rule_color_list, cmap='plasma')
        plt.savefig(experiment_path + '/Composite/rulepop/ruleclusters/' + str(rule_cluster_count) + '_clusters/ruleClustermap.png', dpi=300)
        plt.close('all')

        with open(experiment_path + '/Composite/rulepop/ruleclusters/' + str(rule_cluster_count) + '_clusters/ruleClusters.csv', mode='w') as file:
            writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            for rule_cluster in rule_subclusters:
                exp_color = rule_color_dict[rule_cluster[0]]
                writer.writerow(['ClusterID: ' + exp_color])
                writer.writerow(list(data_headers) + [class_label, 'Accuracy', 'Numerosity', 'Specificity', 'Init Timestamp'])

                spec_sum = np.array([0.0] * len(data_headers))
                acc_spec_sum = np.array([0.0] * len(data_headers))
                acc_sum = 0
                numerosity_sum = 0
                init_ts_sum = 0
                specificity_sum = 0

                for inst_index in rule_cluster:
                    rule = merged_population[inst_index]
                    condition = []
                    condition_counter = 0
                    for attr_index in range(len(data_headers)):
                        if attr_index in rule.specifiedAttList:
                            condition.append(rule.condition[condition_counter])
                            spec_sum[attr_index] += 1
                            acc_spec_sum[attr_index] += rule.accuracy
                            condition_counter += 1
                        else:
                            condition.append('#')
                    writer.writerow(condition + [rule.phenotype, rule.accuracy, rule.numerosity,len(rule.specifiedAttList) / len(data_headers), rule.initTimeStamp])

                    acc_sum += rule.accuracy * rule.numerosity
                    numerosity_sum += rule.numerosity
                    init_ts_sum += rule.initTimeStamp * rule.numerosity
                    specificity_sum += len(rule.specifiedAttList) / len(data_headers) * rule.numerosity

                writer.writerow(['Rule Specificity Sums'])
                ks = []
                vs = []
                for k, v in sorted(dict(zip(list(data_headers), list(spec_sum))).items(), key=lambda item: item[1]):
                    ks.append(k)
                    vs.append(v)
                writer.writerow(list(reversed(ks)))
                writer.writerow(list(reversed(vs)))

                writer.writerow(['Rule Accuracy Weighted Specificity Sums'])
                ks = []
                vs = []
                for k, v in sorted(dict(zip(list(data_headers), list(acc_spec_sum))).items(), key=lambda item: item[1]):
                    ks.append(k)
                    vs.append(v)
                writer.writerow(list(reversed(ks)))
                writer.writerow(list(reversed(vs)))

                writer.writerow(['Avg Accuracy', 'Avg Init Timestamp', 'Avg Specificity'])
                writer.writerow(
                    [acc_sum / numerosity_sum, init_ts_sum / numerosity_sum, specificity_sum / numerosity_sum])

                writer.writerow([])
        file.close()

    ####################################################################################################################
    # Rule Specificity Network
    attribute_acc_specificity_counts = np.zeros(len(data_headers))
    merged_attribute_cooccurrences = []
    for model in models:
        attribute_acc_specificity_counts += np.array(model.get_final_attribute_specificity_list())
        attribute_cooccurrences = model.get_final_attribute_coocurrences(data_headers, len(data_headers))
        if merged_attribute_cooccurrences == []:
            merged_attribute_cooccurrences = attribute_cooccurrences
        else:
            for index in range(len(attribute_cooccurrences)):
                pair = attribute_cooccurrences[index]
                shouldAdd = True
                for index2 in range(len(merged_attribute_cooccurrences)):
                    pair2 = merged_attribute_cooccurrences[index2]
                    if (pair[0] == pair2[0] and pair[1] == pair2[1]) or (pair[0] == pair2[1] and pair[1] == pair2[0]):
                        merged_attribute_cooccurrences[index2] += attribute_cooccurrences[index]
                        shouldAdd = False
                if shouldAdd:
                    merged_attribute_cooccurrences.append(attribute_cooccurrences[index])
    acc_spec_dict = {}
    for header_index in range(len(data_headers)):
        acc_spec_dict[data_headers[header_index]] = attribute_acc_specificity_counts[header_index]

    G = nx.Graph()
    edge_list = []
    weight_list = []
    for co in merged_attribute_cooccurrences:
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

    nx.draw_networkx_nodes(G, pos, nodelist=acc_spec_dict.keys(), node_size=[v * 1 for v in acc_spec_dict.values()],node_color='#FF3377')
    nx.draw_networkx_edges(G, pos, edge_color='#E0B8FF', edgelist=edge_list, width=[v * 1 for v in weight_list])
    nx.draw_networkx_labels(G, pos)
    plt.axis('off')
    plt.savefig(experiment_path + '/Composite/rulepop/rulepopGraph.png', dpi=300)
    plt.close('all')
    ####################################################################################################################

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

def randomHex():
    s = '#'
    for i in range(6):
        s+=random.choice(['0','1','2','3','4','5','6','7','8','9','A','B','C','D','E','F'])
    return s

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

if __name__ == '__main__':
    sys.exit(main(sys.argv))