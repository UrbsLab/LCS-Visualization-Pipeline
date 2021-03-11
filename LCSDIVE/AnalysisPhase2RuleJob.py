import sys
import time
import pickle
import seaborn
from statistics import mean
import csv
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist, pdist, squareform
import os
import math
import numpy as np
import pandas as pd
import HClust
from sklearn.metrics import silhouette_score
import matplotlib as mpl

from Utilities import find_elbow

def job(experiment_path, rule_height_factor):
    job_start_time = time.time()

    # Load information
    file = open(experiment_path + '/phase1pickle', 'rb')
    phase1_pickle = pickle.load(file)
    file.close()

    cv_count = phase1_pickle[9]
    full_info = phase1_pickle[1]

    data_headers = full_info[2]
    class_label = phase1_pickle[8]
    random_state = phase1_pickle[7]

    cmap = mpl.colors.ListedColormap(['xkcd:navy blue','xkcd:yellow'])

    # Merge Rule Population
    merged_population = []
    for cv in range(cv_count):
        file = open(experiment_path + '/CV_' + str(cv) + '/model', 'rb')
        model = pickle.load(file)
        for rule in model.population.popSet:
            merged_population.append(rule)

    num_rules = len(merged_population)
    rule_specificity_array = []
    micro_to_macro_rule_index_map = {}
    micro_rule_index_count = 0
    macro_rule_index_count = 0
    for inst in range(num_rules):
        rule = merged_population[inst]
        a = []
        for attribute in range(len(data_headers)):
            if attribute in rule.specifiedAttList:
                a.append(1)
            else:
                a.append(0)
        for microclassifier in range(rule.numerosity):
            rule_specificity_array.append(a)
            micro_to_macro_rule_index_map[micro_rule_index_count] = macro_rule_index_count
            micro_rule_index_count += 1
        macro_rule_index_count += 1
    rule_specificity_array = np.array(rule_specificity_array)

    rule_df = pd.DataFrame(rule_specificity_array, columns=data_headers, index=list(range(micro_rule_index_count)))

    plt.figure(figsize=((10 / math.sqrt(rule_height_factor), 10 * math.sqrt(rule_height_factor))))
    h = seaborn.heatmap(rule_df, cmap=cmap, cbar_kws={'ticks':[0,1]})
    h.tick_params(left=False,labelleft=False)
    if rule_specificity_array.shape[1] <= 11:
        h.set_xticklabels(h.get_xmajorticklabels(),fontsize='xx-large')
    elif rule_specificity_array.shape[1] <= 20:
        h.set_xticklabels(h.get_xmajorticklabels(),fontsize='x-large')
    if rule_specificity_array.shape[1] >= 20:
        plt.xticks(rotation=90)
    plt.xlabel('Features',fontsize='xx-large')
    plt.ylabel('Rules',fontsize='xx-large')
    plt.tight_layout()
    plt.savefig(experiment_path + '/Composite/rulepop/rulepopHeatmap.png')
    plt.close('all')

    # Rule Population Clustermaps
    r = seaborn.clustermap(rule_df, metric='sqeuclidean', method='ward', cmap=cmap, cbar_kws={'ticks':[0,1]})
    rule_cluster_tree = HClust.createClusterTree(r.dendrogram_row.linkage, list(range(micro_rule_index_count)),rule_df.to_numpy())

    rule_clusters, rule_colors = rule_cluster_tree.getSignificantClusters(p_value=0.05, sample_count=100,metric='sqeuclidean', method='ward',random_state=random_state)

    rule_distortions = []

    precomputed_distances = squareform(pdist(rule_specificity_array, metric='sqeuclidean'))
    silhouettes = []

    for rule_cluster_count in reversed(range(1, len(rule_clusters) + 1)):
        if not os.path.exists(
                experiment_path + '/Composite/rulepop/ruleclusters/' + str(rule_cluster_count) + '_clusters'):
            os.mkdir(experiment_path + '/Composite/rulepop/ruleclusters/' + str(rule_cluster_count) + '_clusters')

        rule_subclusters, rule_colors = rule_cluster_tree.getNSignificantClusters(rule_cluster_count, p_value=0.05,sample_count=100, metric='sqeuclidean',method='ward',random_state=random_state)

        # Elbow Method
        centroids = []
        for cluster in rule_subclusters:
            centroid = np.zeros(len(data_headers))
            for inst_label_index in cluster:
                centroid += rule_specificity_array[inst_label_index]
            centroid /= len(cluster)
            centroids.append(centroid)
        centroids = np.array(centroids)
        rule_distortions.append(sum(np.min(cdist(rule_specificity_array, centroids, 'sqeuclidean'), axis=1)))

        # Silhouette Method
        s_counter = 0
        new_l = [0] * micro_rule_index_count
        for cluster in rule_subclusters:
            for inst_label_index in cluster:
                new_l[inst_label_index] = s_counter
            s_counter += 1
        if rule_cluster_count != 1:
            silhouettes.append(silhouette_score(precomputed_distances, new_l, metric='precomputed'))
        else:
            silhouettes.append(0)

        # Clustermaps
        rule_color_dict = {}
        rule_color_count = 0
        for cluster in rule_subclusters:
            random_color = rule_colors[rule_color_count]
            for inst_label in cluster:
                rule_color_dict[inst_label] = random_color
            rule_color_count += 1
        rule_color_list = pd.Series(dict(sorted(rule_color_dict.items())))
        rule_color_list = pd.Series.to_frame(rule_color_list)
        rule_color_list.columns = ['Found']

        c = seaborn.clustermap(rule_df, row_linkage=r.dendrogram_row.linkage, col_linkage=r.dendrogram_col.linkage,row_colors=rule_color_list, cmap=cmap, cbar_kws={'ticks':[0,1]}, figsize=(10 / math.sqrt(rule_height_factor), 10 * math.sqrt(rule_height_factor)))
        c.ax_heatmap.tick_params(right=False,labelright=False)
        if rule_specificity_array.shape[1] <= 11:
            c.ax_heatmap.set_xticklabels(c.ax_heatmap.get_xmajorticklabels(), fontsize='xx-large')
        elif rule_specificity_array.shape[1] <= 20:
            c.ax_heatmap.set_xticklabels(c.ax_heatmap.get_xmajorticklabels(),fontsize='x-large')
        c.ax_heatmap.set_xlabel('Features',fontsize='xx-large')
        c.ax_heatmap.set_ylabel('Rules',fontsize='xx-large',rotation=-90,labelpad=20)
        if rule_specificity_array.shape[1] >= 20:
            plt.setp(c.ax_heatmap.get_xticklabels(),rotation=90)
        plt.tight_layout()
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

                covered_macro_rule_indices = []
                for inst_index in rule_cluster:
                    macro_rule_index = micro_to_macro_rule_index_map[inst_index]
                    if not macro_rule_index in covered_macro_rule_indices:
                        covered_macro_rule_indices.append(macro_rule_index)
                        rule = merged_population[macro_rule_index]
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
                        writer.writerow(condition + [rule.phenotype, rule.accuracy, rule.numerosity,
                                                     len(rule.specifiedAttList) / len(data_headers),
                                                     rule.initTimeStamp])

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
                writer.writerow([acc_sum / numerosity_sum, init_ts_sum / numerosity_sum, specificity_sum / numerosity_sum])

                writer.writerow([])
        file.close()

    # Plot Rule Elbow Plot
    rule_distortions.reverse()
    optimal_elbow = find_elbow(rule_distortions)
    plt.plot(range(1, len(rule_clusters) + 1), rule_distortions, 'bx-')
    plt.xlabel('Number of Clusters',fontsize='xx-large')
    plt.ylabel('Distortion',fontsize='xx-large')
    plt.axvline(x=optimal_elbow, color='xkcd:sky', linestyle='--',label='Elbow at: '+str(optimal_elbow))
    plt.legend(fontsize='xx-large')
    plt.savefig(experiment_path + '/Composite/rulepop/' + str(optimal_elbow) + 'optimalClusters.png',dpi=300)
    plt.close('all')

    # Plot AT Silhouette Plot
    silhouettes.reverse()
    optimal_silhouette = np.argmax(np.array(silhouettes))+1
    plt.plot(range(1, len(rule_clusters) + 1), silhouettes, 'bx-')
    plt.xlabel('Number of Clusters', fontsize='xx-large')
    plt.ylabel('Silhouette Score', fontsize='xx-large')
    plt.axvline(x=optimal_silhouette, color='xkcd:sky', linestyle='--', label='Max at: ' + str(optimal_silhouette))
    plt.legend(fontsize='xx-large')
    plt.savefig(experiment_path + '/Composite/rulepop/' + str(optimal_silhouette) + 'optimalSilhouette.png', dpi=300)
    plt.close('all')

    # Save Runtime
    runtime_file = open(experiment_path + '/Composite/rulepop/runtime.txt', 'w')
    runtime_file.write(str(time.time() - job_start_time))
    runtime_file.close()

    # Print completion
    print("Rule phase 2 complete")

if __name__ == '__main__':
    job(sys.argv[1],float(sys.argv[2]))