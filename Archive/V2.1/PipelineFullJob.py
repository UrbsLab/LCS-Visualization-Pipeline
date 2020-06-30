import sys
import os
import pickle
from skExSTraCS import ExSTraCS
import networkx as nx
import Utilities
import seaborn
import matplotlib.pyplot as plt
import copy
import pandas as pd
import numpy as np
import HClust
import csv

def job(experiment_path,cv):
    # Unpack Data
    file = open(experiment_path + '/pickledCV_' + str(cv), 'rb')
    rawData = pickle.load(file)
    file.close()

    save = rawData[cv]
    data_features = save[0]
    data_phenotypes = save[1]
    instance_labels = save[2]
    group_labels = save[3]
    data_headers = save[4]

    group_colors = rawData[-2]
    learning_iterations = rawData[-1][0]
    N = rawData[-1][1]
    nu = rawData[-1][2]
    attribute_tracking_method = rawData[-1][3]
    rulepop_clustering_method = rawData[-1][4]
    random_state = rawData[-1][5]
    class_label = rawData[-1][6]
    group_label = rawData[-1][7]
    inst_label = rawData[-1][8]

    # Create CV directories
    if not os.path.exists(experiment_path + 'CV_' + str(cv)):
        os.mkdir(experiment_path + '/Full')
        os.mkdir(experiment_path + '/Full/training')
        os.mkdir(experiment_path + '/Full/visualizations')
        os.mkdir(experiment_path + '/Full/visualizations/rulepop')
        os.mkdir(experiment_path + '/Full/visualizations/rulepop/ruleclusters')
        os.mkdir(experiment_path + '/Full/visualizations/at')
        os.mkdir(experiment_path + '/Full/visualizations/at/atclusters')

    # Train ExSTraCS Model###############################################################################################
    model = ExSTraCS(learning_iterations=learning_iterations, N=N, nu=nu,attribute_tracking_method=attribute_tracking_method, rule_compaction=None,random_state=random_state)
    model.fit(data_features, data_phenotypes)

    outfile = open(experiment_path + '/Full/training/model', 'wb')
    pickle.dump(model, outfile)
    outfile.close()

    # Rule Population Heatmap########################################################################################
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
    plt.savefig(experiment_path + '/Full/visualizations/rulepop/rulepopHeatmap.png')
    plt.close('all')

    # Rule Population Clustermaps
    if rulepop_clustering_method == 'spearman':
        metric = Utilities.spearmanDistance
    elif rulepop_clustering_method == 'pearson':
        metric = Utilities.pearsonDistance
    else:
        metric = 'correlation'

    r = seaborn.clustermap(rule_df, metric=metric, method='ward', cmap='plasma')
    rule_cluster_tree = HClust.createClusterTree(r.dendrogram_row.linkage,list(range(num_rules)),rule_df.to_numpy())
    rule_clusters, rule_colors = rule_cluster_tree.getSignificantClusters(p_value=0.05,sample_count=100,metric=metric,method='ward',random_state=random_state)
    for rule_cluster_count in reversed(range(1,len(rule_clusters)+1)):
        if not os.path.exists(experiment_path + '/Full/visualizations/rulepop/ruleclusters/'+str(rule_cluster_count)+'_clusters'):
            os.mkdir(experiment_path + '/Full/visualizations/rulepop/ruleclusters/'+str(rule_cluster_count)+'_clusters')

        rule_subclusters, rule_colors = rule_cluster_tree.getNSignificantClusters(rule_cluster_count,p_value=0.05, sample_count=100,metric=metric, method='ward',random_state=random_state)
        rule_color_dict = {}
        rule_color_count = 0
        for cluster in rule_subclusters:
            random_color = rule_colors[rule_color_count]
            for inst_label in cluster:
                rule_color_dict[inst_label] = random_color
            rule_color_count += 1
        rule_color_list = pd.Series(dict(sorted(rule_color_dict.items())))

        seaborn.clustermap(rule_df, row_linkage=r.dendrogram_row.linkage, col_linkage=r.dendrogram_col.linkage,row_colors=rule_color_list, cmap='plasma')
        plt.savefig(experiment_path + '/Full/visualizations/rulepop/ruleclusters/'+str(rule_cluster_count)+'_clusters/ruleClustermap.png',dpi=300)
        plt.close('all')

        with open(experiment_path + '/Full/visualizations/rulepop/ruleclusters/'+str(rule_cluster_count)+'_clusters/ruleClusters.csv', mode='w') as file:
            writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            for rule_cluster in rule_subclusters:
                exp_color = rule_color_dict[rule_cluster[0]]
                writer.writerow(['ClusterID: ' + exp_color])
                writer.writerow(list(data_headers)+[class_label,'Accuracy','Numerosity','Specificity','Init Timestamp'])

                spec_sum = np.array([0.0] * len(data_headers))
                acc_spec_sum = np.array([0.0] * len(data_headers))
                acc_sum = 0
                numerosity_sum = 0
                init_ts_sum = 0
                specificity_sum = 0

                for inst_index in rule_cluster:
                    rule = model.population.popSet[inst_index]
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
                    writer.writerow(condition+[rule.phenotype,rule.accuracy,rule.numerosity,len(rule.specifiedAttList)/len(data_headers),rule.initTimeStamp])

                    acc_sum += rule.accuracy*rule.numerosity
                    numerosity_sum += rule.numerosity
                    init_ts_sum += rule.initTimeStamp*rule.numerosity
                    specificity_sum += len(rule.specifiedAttList)/len(data_headers)*rule.numerosity

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

                writer.writerow(['Avg Accuracy','Avg Init Timestamp','Avg Specificity'])
                writer.writerow([acc_sum/numerosity_sum,init_ts_sum/numerosity_sum,specificity_sum/numerosity_sum])

                writer.writerow([])
        file.close()

    # Rule Specificity Network#######################################################################################
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

    nx.draw_networkx_nodes(G, pos, nodelist=acc_spec_dict.keys(), node_size=[v * 1 for v in acc_spec_dict.values()],node_color='#FF3377')
    nx.draw_networkx_edges(G, pos, edge_color='#E0B8FF', edgelist=edge_list, width=[v * 1 for v in weight_list])
    nx.draw_networkx_labels(G, pos)
    plt.axis('off')
    plt.savefig(experiment_path + '/Full/visualizations/rulepop/rulepopGraph.png', dpi=300)
    plt.close('all')

    # AT Scores Heatmap##############################################################################################
    # Get AT Scores for each instance
    AT_scores = model.get_attribute_tracking_scores(instance_labels=np.array(instance_labels))

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

    AT_full_df = pd.DataFrame(normalized_AT_scores, columns=data_headers, index=instance_labels)
    seaborn.heatmap(AT_full_df, cmap='plasma')
    plt.savefig(experiment_path + '/Full/visualizations/at/ATHeatmap.png')
    plt.close('all')

    g = seaborn.clustermap(AT_full_df, metric=Utilities.pearsonDistance, method='ward', cmap='plasma')
    cluster_tree = HClust.createClusterTree(g.dendrogram_row.linkage, instance_labels, AT_full_df.to_numpy())
    clusters, colors = cluster_tree.getSignificantClusters(p_value=0.05, sample_count=100, metric='correlation',method='ward',random_state=random_state)

    # Generate maps and csvs for each cluster configurations#############################################################
    for cluster_count in reversed(range(1, len(clusters) + 1)):
        if not os.path.exists(experiment_path + '/Full/visualizations/at/atclusters/' + str(cluster_count) + '_clusters'):
            os.mkdir(experiment_path + '/Full/visualizations/at/atclusters/' + str(cluster_count) + '_clusters')

        subclusters, colors = cluster_tree.getNSignificantClusters(cluster_count, p_value=0.05, sample_count=100,metric='correlation', method='ward',random_state=random_state)

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
        for i in range(len(group_labels)):
            if group_labels[i] in group_dict:
                group_dict[group_labels[i]].append(instance_labels[i])
            else:
                group_dict[group_labels[i]] = [instance_labels[i]]

        group_color_dict = {}
        for group in group_dict:
            random_color = group_colors[group]
            for inst_label in group_dict[group]:
                group_color_dict[inst_label] = random_color
        group_list = pd.Series(dict(sorted(group_color_dict.items())))

        combo_list = pd.concat([group_list, color_list], axis=1)

        g = seaborn.clustermap(AT_full_df, row_linkage=g.dendrogram_row.linkage, col_linkage=g.dendrogram_col.linkage,row_colors=combo_list, cmap='plasma')
        plt.savefig(experiment_path + '/Full/visualizations/at/atclusters/' + str(cluster_count) + '_clusters/ATclustermap.png',dpi=300)
        plt.close('all')

        with open(experiment_path + '/Full/visualizations/at/atclusters/' + str(cluster_count) + '_clusters/labeledDataset.csv',mode='w') as file:
            writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(['Instance','Cluster'] + list(data_headers) + [class_label])
            for cluster in subclusters:
                exp_color = color_dict[cluster[0]]
                for exp_instance in cluster:
                    exp_index = instance_labels.index(exp_instance)
                    exp_state = data_features[exp_index].tolist()
                    exp_phenotype = data_phenotypes[exp_index]
                    writer.writerow([exp_instance, exp_color] + exp_state + [exp_phenotype])
        file.close()

        with open(experiment_path + '/Full/visualizations/at/atclusters/' + str(cluster_count) + '_clusters/clusters.csv', mode='w') as file:
            writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            for cluster in subclusters:
                exp_color = color_dict[cluster[0]]
                writer.writerow(['ClusterID: ' + exp_color])
                writer.writerow(['Instance Label', 'True Group', ''] + list(data_headers) + [class_label])
                at_sums = np.array([0.0]*len(data_headers))
                group_makeup = {}
                class_makeup = {}
                for exp_instance in cluster:
                    exp_index = instance_labels.index(exp_instance)
                    exp_group = group_labels[exp_index]
                    exp_state = data_features[exp_index].tolist()
                    exp_phenotype = data_phenotypes[exp_index]
                    writer.writerow([exp_instance, exp_group, ''] + exp_state + [exp_phenotype])

                    at_sums += normalized_AT_scores[exp_index]
                    if exp_group in group_makeup:
                        group_makeup[exp_group] += 1/len(cluster)
                    else:
                        group_makeup[exp_group] = 1/len(cluster)
                    if exp_phenotype in class_makeup:
                        class_makeup[exp_phenotype] += 1/len(cluster)
                    else:
                        class_makeup[exp_phenotype] = 1/len(cluster)

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
    os.remove(experiment_path + '/pickledCV_' + str(cv))

if __name__ == '__main__':
    job(sys.argv[1],sys.argv[2])