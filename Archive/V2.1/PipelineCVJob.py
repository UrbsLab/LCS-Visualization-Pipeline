import sys
import pickle
import os
from skExSTraCS import ExSTraCS
import numpy as np
import pandas as pd
import HClust
import seaborn
import Utilities
import matplotlib.pyplot as plt
import csv

def job(experiment_path,cv):
    #Unpack Data
    file = open(experiment_path+'/pickledCV_'+str(cv), 'rb')
    rawData = pickle.load(file)
    file.close()

    save = rawData[cv]
    train_data_features = save[0]
    train_data_phenotypes = save[1]
    train_instance_labels = save[2]
    train_group_labels = save[3]
    test_data_features = save[4]
    test_data_phenotypes = save[5]
    test_instance_labels = save[6]
    test_group_labels = save[7]
    data_headers = save[8]

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

    #Create CV directories
    if not os.path.exists(experiment_path+'/CV_'+str(cv)):
        os.mkdir(experiment_path + '/CV_' + str(cv))
        os.mkdir(experiment_path + '/CV_' + str(cv) + '/training')
        os.mkdir(experiment_path + '/CV_' + str(cv) + '/atclusters')

    #Train ExSTraCS Model###############################################################################################
    model = ExSTraCS(learning_iterations=learning_iterations,N=N,nu=nu,attribute_tracking_method=attribute_tracking_method,rule_compaction=None,random_state=random_state)
    model.fit(train_data_features, train_data_phenotypes)

    outfile = open(experiment_path + '/CV_' + str(cv) + '/training/model', 'wb')
    pickle.dump(model, outfile)
    outfile.close()

    # Export Testing Accuracy
    outfile = open(experiment_path + '/CV_' + str(cv) + '/training/testingAccuracy.txt', mode='w')
    outfile.write(str(model.score(test_data_features, test_data_phenotypes)))
    outfile.close()

    # Save train and testing datasets into csvs
    with open(experiment_path + '/CV_' + str(cv) + '/training/trainDataset.csv', mode='w') as file:
        writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(list(data_headers)+['Class',inst_label,'True '+str(group_label)])
        for i in range(len(train_instance_labels)):
            writer.writerow(list(train_data_features[i])+[train_data_phenotypes[i]]+[train_instance_labels[i]]+[train_group_labels[i]])

    with open(experiment_path + '/CV_' + str(cv) + '/training/testDataset.csv', mode='w') as file:
        writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(list(data_headers)+['Class',inst_label,'True '+str(group_label)])
        for i in range(len(test_instance_labels)):
            writer.writerow(list(test_data_features[i])+[test_data_phenotypes[i]]+[test_instance_labels[i]]+[test_group_labels[i]])

    #Cluster Training AT Scores#########################################################################################
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

    #Get AT training clusters
    AT_full_df = pd.DataFrame(normalized_AT_scores, columns=data_headers, index=train_instance_labels)
    g = seaborn.clustermap(AT_full_df, metric='correlation', method='ward', cmap='plasma')
    print('clustermap done')
    #g = seaborn.clustermap(AT_full_df, metric=Utilities.pearsonDistance, method='ward', cmap='plasma')
    cluster_tree = HClust.createClusterTree(g.dendrogram_row.linkage, train_instance_labels, AT_full_df.to_numpy())
    clusters, colors = cluster_tree.getSignificantClusters(p_value=0.05, sample_count=100, metric='correlation',method='ward',random_state=random_state)

    #Get Normalized AT scores for testing instances#####################################################################
    # Pair test instances with a prediction
    num_attributes = len(data_headers)
    test_predictions = model.predict(test_data_features)
    predicted_instances = np.insert(test_data_features, num_attributes, test_predictions, 1)

    # Get Test Instance AT Scores: Eventually build this out to be a part of the package
    test_attribute_tracking_sums = [[0] * num_attributes for i in range(len(predicted_instances))]
    at_counter = 0
    for instance in predicted_instances:
        state = instance[:-1]
        model.population.makeEvalMatchSet(model, state)
        correct_set = []
        for matched_rule_index in model.population.matchSet:
            if model.population.popSet[matched_rule_index].phenotype == instance[-1]:
                correct_set.append(matched_rule_index)
        model.population.clearSets()

        # external AT update
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

    # Get AT scores in return format
    ret_list = []
    for i in range(len(predicted_instances)):
        ret_list.append([test_instance_labels[i], test_attribute_tracking_sums[i]])

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

    #Generate maps and csvs for each cluster configurations#############################################################
    for cluster_count in reversed(range(1, len(clusters)+1)):
        if not os.path.exists(experiment_path + '/CV_' + str(cv) + '/atclusters/'+str(cluster_count)+'_clusters'):
            os.mkdir(experiment_path + '/CV_' + str(cv) + '/atclusters/'+str(cluster_count)+'_clusters')

        subclusters, colors = cluster_tree.getNSignificantClusters(cluster_count, p_value=0.05, sample_count=100,metric='correlation', method='ward',random_state=random_state)

        #Clustermaps
        color_dict = {}
        color_count = 0
        for cluster in subclusters:
            random_color = colors[color_count]
            for inst_label in cluster:
                color_dict[inst_label] = random_color
            color_count += 1
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

        combo_list = pd.concat([group_list, color_list], axis=1)

        g = seaborn.clustermap(AT_full_df, row_linkage=g.dendrogram_row.linkage, col_linkage=g.dendrogram_col.linkage,row_colors=combo_list, cmap='plasma')
        plt.savefig(experiment_path + '/CV_' + str(cv) + '/atclusters/'+str(cluster_count)+'_clusters/ATclustermap.png', dpi=300)
        plt.close('all')

        #Test Instances
        test_clusters = {}
        for i in range(len(test_instance_labels)):
            test_instance_label = test_instance_labels[i]
            test_score = normalized_test_AT_scores[i]
            color = get_closest_cluster_color(clusters, normalized_AT_scores, train_instance_labels, test_score, color_dict)
            if color in test_clusters:
                test_clusters[color].append(test_instance_label)
            else:
                test_clusters[color] = [test_instance_label]

        with open(experiment_path + '/CV_' + str(cv) + '/atclusters/'+str(cluster_count)+'_clusters/clusters.csv', mode='w') as file:
            writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            for cluster in subclusters:
                exp_color = color_dict[cluster[0]]
                writer.writerow(['ClusterID: ' + exp_color])
                writer.writerow(['Instance Label', 'True Group', 'Test/Train', ''] + list(data_headers) + ['Actual ' + class_label,'Predicted ' + class_label])

                at_sums = np.array([0.0] * len(data_headers))
                group_makeup = {}
                class_makeup = {}
                at_sums_test = np.array([0.0] * len(data_headers))
                group_makeup_test = {}
                class_makeup_test = {}
                at_sums_train = np.array([0.0] * len(data_headers))
                group_makeup_train = {}
                class_makeup_train = {}

                if exp_color in test_clusters:
                    total_size = len(cluster) + len(test_clusters[exp_color])
                else:
                    total_size = len(cluster)

                for exp_traininstance in cluster:
                    exp_index = train_instance_labels.index(exp_traininstance)
                    exp_traingroup = train_group_labels[exp_index]
                    exp_state = train_data_features[exp_index].tolist()
                    exp_phenotype = train_data_phenotypes[exp_index]
                    writer.writerow([exp_traininstance, exp_traingroup, 'Train', ''] + exp_state + [exp_phenotype, 'N/A'])

                    at_sums += normalized_AT_scores[exp_index]
                    if exp_traingroup in group_makeup:
                        group_makeup[exp_traingroup] += 1 / total_size
                    else:
                        group_makeup[exp_traingroup] = 1 / total_size
                    if exp_phenotype in class_makeup:
                        class_makeup[exp_phenotype] += 1 / total_size
                    else:
                        class_makeup[exp_phenotype] = 1 / total_size

                    at_sums_train += normalized_AT_scores[exp_index]
                    if exp_traingroup in group_makeup_train:
                        group_makeup_train[exp_traingroup] += 1 / len(cluster)
                    else:
                        group_makeup_train[exp_traingroup] = 1 / len(cluster)
                    if exp_phenotype in class_makeup_train:
                        class_makeup_train[exp_phenotype] += 1 / len(cluster)
                    else:
                        class_makeup_train[exp_phenotype] = 1 / len(cluster)
                if exp_color in test_clusters:
                    testing_accuracy = 0
                    for exp_testinstance in test_clusters[exp_color]:
                        exp_index = test_instance_labels.index(exp_testinstance)
                        exp_testgroup = test_group_labels[exp_index]
                        exp_state = test_data_features[exp_index].tolist()
                        exp_phenotype = [test_data_phenotypes[exp_index], test_predictions[exp_index]]
                        writer.writerow([exp_testinstance, exp_testgroup, 'Test', ''] + exp_state + exp_phenotype)

                        if test_data_phenotypes[exp_index] == test_predictions[exp_index]:
                            testing_accuracy += 1

                        at_sums += normalized_test_AT_scores[exp_index]
                        if exp_testgroup in group_makeup:
                            group_makeup[exp_testgroup] += 1 / total_size
                        else:
                            group_makeup[exp_testgroup] = 1 / total_size
                        if exp_phenotype[0] in class_makeup:
                            class_makeup[exp_phenotype[0]] += 1 / total_size
                        else:
                            class_makeup[exp_phenotype[0]] = 1 / total_size

                        at_sums_test += normalized_test_AT_scores[exp_index]
                        if exp_testgroup in group_makeup_test:
                            group_makeup_test[exp_testgroup] += 1 / len(test_clusters[exp_color])
                        else:
                            group_makeup_test[exp_testgroup] = 1 / len(test_clusters[exp_color])
                        if exp_phenotype[0] in class_makeup_test:
                            class_makeup_test[exp_phenotype[0]] += 1 / len(test_clusters[exp_color])
                        else:
                            class_makeup_test[exp_phenotype[0]] = 1 / len(test_clusters[exp_color])

                    writer.writerow(["Cluster Testing Accuracy: "+str(testing_accuracy/len(test_clusters[exp_color]))])

                writer.writerow(['AT Sums:'])
                ks = []
                vs = []
                for k, v in sorted(dict(zip(list(data_headers), list(at_sums))).items(), key=lambda item: item[1]):
                    ks.append(k)
                    vs.append(v)
                writer.writerow(list(reversed(ks)))
                writer.writerow(list(reversed(vs)))

                writer.writerow(['AT Sums Training Only:'])
                ks = []
                vs = []
                for k, v in sorted(dict(zip(list(data_headers), list(at_sums_train))).items(), key=lambda item: item[1]):
                    ks.append(k)
                    vs.append(v)
                writer.writerow(list(reversed(ks)))
                writer.writerow(list(reversed(vs)))

                if exp_color in test_clusters:
                    writer.writerow(['AT Sums Testing Only:'])
                    ks = []
                    vs = []
                    for k, v in sorted(dict(zip(list(data_headers), list(at_sums_test))).items(), key=lambda item: item[1]):
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

                writer.writerow(['True Group Composition Training Only:'])
                ks = []
                vs = []
                for k, v in sorted(group_makeup_train.items(), key=lambda item: item[1]):
                    ks.append(k)
                    vs.append(v)
                writer.writerow(list(reversed(ks)))
                writer.writerow(list(reversed(vs)))

                if exp_color in test_clusters:
                    writer.writerow(['True Group Composition Testing Only:'])
                    ks = []
                    vs = []
                    for k, v in sorted(group_makeup_test.items(), key=lambda item: item[1]):
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

                writer.writerow(['True Class Composition Training Only:'])
                ks = []
                vs = []
                for k, v in sorted(class_makeup_train.items(), key=lambda item: item[1]):
                    ks.append(k)
                    vs.append(v)
                writer.writerow(list(reversed(ks)))
                writer.writerow(list(reversed(vs)))
                writer.writerow([])

                if exp_color in test_clusters:
                    writer.writerow(['True Class Composition Testing Only:'])
                    ks = []
                    vs = []
                    for k, v in sorted(class_makeup_test.items(), key=lambda item: item[1]):
                        ks.append(k)
                        vs.append(v)
                    writer.writerow(list(reversed(ks)))
                    writer.writerow(list(reversed(vs)))

                writer.writerow([])
        file.close()
    os.remove(experiment_path + '/pickledCV_' + str(cv))

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
            avg_distance += Utilities.pearsonDistance(test_instance,cluster_instance)
        avg_distance /= len(cluster)
        distances.append(avg_distance)

    #Return color of arbitrary instance in closest cluster
    return color_dict[cluster_labels[distances.index(min(distances))][0]]


if __name__ == '__main__':
    job(sys.argv[1],sys.argv[2])