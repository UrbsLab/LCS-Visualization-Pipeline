import sys
import time
import pickle
from Utilities import pearsonDistance, find_elbow
import numpy as np
import pandas as pd
import math
import HClust
import os
import csv
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
import seaborn

def job(experiment_path, at_height_factor):
    job_start_time = time.time()

    # Load information
    file = open(experiment_path + '/phase1pickle', 'rb')
    phase1_pickle = pickle.load(file)
    file.close()

    cv_count = phase1_pickle[9]
    cv_info = phase1_pickle[0]
    full_info = phase1_pickle[1]
    data_features = full_info[0]
    data_phenotypes = full_info[1]
    data_headers = full_info[2]
    full_instance_labels = full_info[3]
    full_group_labels = full_info[4]
    group_colors = full_info[5]
    class_label = phase1_pickle[8]
    random_state = phase1_pickle[7]

    visualize_true_clusters = phase1_pickle[2]

    # Merge AT scores and create heatmap
    merged_AT_dict = {}
    merged_AT_dict_count = {}
    for cv in range(cv_count):
        inst_label = cv_info[cv][8]
        partial_AT_scores = pd.read_csv(experiment_path + '/CV_' + str(cv) + '/normalizedATScores.csv')
        partial_scores = partial_AT_scores.drop(inst_label, axis=1).values
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
        merged_AT.append(merged_AT_dict[label] / merged_AT_dict_count[label])  # renormalize
    merged_AT = np.array(merged_AT)
    AT_full_df = pd.DataFrame(merged_AT, columns=data_headers, index=full_instance_labels)

    plt.figure(figsize=((10 / math.sqrt(at_height_factor), 10 * math.sqrt(at_height_factor))))
    seaborn.heatmap(AT_full_df, cmap='plasma')
    plt.savefig(experiment_path + '/Composite/at/ATHeatmap.png')
    plt.close('all')

    # Merge Instance Test scores
    merged_test_dict = {}
    merged_test_dict_count = {}
    for cv in range(cv_count):
        inst_label = cv_info[cv][8]
        partial_test_scores = pd.read_csv(experiment_path + '/CV_' + str(cv) + '/instTestingAccuracy.csv')
        partial_scores = partial_test_scores.drop(inst_label, axis=1).values
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
        merged_test.append(merged_test_dict[label] / merged_test_dict_count[label])  # renormalize
    merged_test = np.array(merged_test)

    # AT Clustermaps and CSV Analysis
    try:
        g = seaborn.clustermap(AT_full_df, metric='correlation', method='ward', cmap='plasma')
    except:
        print('AT Clustermap default pearson failed. Trying slower own Pearson method instead')
        g = seaborn.clustermap(AT_full_df, metric=pearsonDistance, method='ward', cmap='plasma')

    cluster_tree = HClust.createClusterTree(g.dendrogram_row.linkage, full_instance_labels, AT_full_df.to_numpy())
    clusters, colors = cluster_tree.getSignificantClusters(p_value=0.05, sample_count=100, metric='correlation',method='ward', random_state=random_state)

    AT_distortions = []

    for cluster_count in reversed(range(1, len(clusters) + 1)):
        if not os.path.exists(experiment_path + '/Composite/at/atclusters/' + str(cluster_count) + '_clusters'):
            os.mkdir(experiment_path + '/Composite/at/atclusters/' + str(cluster_count) + '_clusters')

        subclusters, colors = cluster_tree.getNSignificantClusters(cluster_count, p_value=0.05, sample_count=100,metric='correlation', method='ward',random_state=random_state)

        # Elbow Method
        centroids = []
        for cluster in subclusters:
            centroid = np.zeros(len(data_headers))
            for inst_label in cluster:
                index = full_instance_labels.index(inst_label)
                centroid += merged_AT[index]
            centroid /= len(cluster)
            centroids.append(centroid)
        centroids = np.array(centroids)
        AT_distortions.append(sum(np.min(cdist(merged_AT, centroids, 'euclidean'), axis=1)))

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
            combo_list.columns = ['True Clusters', 'Found Clusters']
        else:
            combo_list = pd.Series.to_frame(color_list)
            combo_list.columns = ['Found Clusters']

        g = seaborn.clustermap(AT_full_df, row_linkage=g.dendrogram_row.linkage, col_linkage=g.dendrogram_col.linkage,row_colors=combo_list, cmap='plasma',figsize=(10 / math.sqrt(at_height_factor), 10 * math.sqrt(at_height_factor)))
        plt.savefig(experiment_path + '/Composite/at/atclusters/' + str(cluster_count) + '_clusters/ATclustermap.png',dpi=300)
        plt.close('all')

        with open(experiment_path + '/Composite/at/atclusters/' + str(cluster_count) + '_clusters/labeledDataset.csv',mode='w') as file:
            writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(['Instance', 'Cluster'] + list(data_headers) + [class_label])
            for cluster in subclusters:
                exp_color = color_dict[cluster[0]]
                for exp_instance in cluster:
                    exp_index = full_instance_labels.index(exp_instance)
                    exp_state = data_features[exp_index].tolist()
                    exp_phenotype = data_phenotypes[exp_index]
                    writer.writerow([exp_instance, exp_color] + exp_state + [exp_phenotype])
        file.close()

        with open(experiment_path + '/Composite/at/atclusters/' + str(cluster_count) + '_clusters/clusters.csv',mode='w') as file:
            writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            for cluster in subclusters:
                exp_color = color_dict[cluster[0]]
                writer.writerow(['ClusterID: ' + exp_color])
                writer.writerow(['Instance Label', 'True Group', ''] + list(data_headers) + [class_label])
                at_sums = np.array([0.0] * len(data_headers))
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
                        group_makeup[exp_group] += 1 / len(cluster)
                    else:
                        group_makeup[exp_group] = 1 / len(cluster)
                    if exp_phenotype in class_makeup:
                        class_makeup[exp_phenotype] += 1 / len(cluster)
                    else:
                        class_makeup[exp_phenotype] = 1 / len(cluster)

                writer.writerow(["Cluster Testing Accuracy: " + str(test_score_sum / len(cluster))])

                writer.writerow(['AT Sums:'])
                ks = []
                vs = []
                for k, v in sorted(dict(zip(list(data_headers), list(at_sums))).items(), key=lambda item: item[1]):
                    ks.append(k)
                    vs.append(v)
                writer.writerow(list(reversed(ks)))
                writer.writerow(list(reversed(vs)))

                writer.writerow(['True Group Composition:'])
                ks = []
                vs = []
                for k, v in sorted(group_makeup.items(), key=lambda item: item[1]):
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

    # Plot AT Elbow Plot
    AT_distortions.reverse()
    plt.plot(range(1, len(clusters) + 1), AT_distortions, 'bx-')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Distortion')
    plt.title('The Elbow Method using Distortion')
    plt.savefig(experiment_path + '/Composite/at/' + str(find_elbow(AT_distortions)) + 'optimalClusters.png', dpi=300)
    plt.close('all')

    # Save Runtime
    runtime_file = open(experiment_path + '/Composite/at/runtime.txt', 'w')
    runtime_file.write(str(time.time() - job_start_time))
    runtime_file.close()

    # Print completion
    print("AT phase 2 complete")

if __name__ == '__main__':
    job(sys.argv[1],float(sys.argv[2]))