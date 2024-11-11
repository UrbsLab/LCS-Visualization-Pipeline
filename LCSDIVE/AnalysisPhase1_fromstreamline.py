import sys
import os
import argparse
import time
import random
import pandas as pd
import numpy as np
import copy
import pickle
import AnalysisPhase1_fromstreamlineJob
import glob

'''
Sample Run Code
python AnalysisPhase1_fromstreamline.py --s /home/bandheyh/STREAMLINE/lcs/ --e lcs --d demodata --o Outputs/ --cluster 0
'''

def main(argv):
    # Parse arguments
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--s', dest='streamline_path', type=str, help='path to output directory')
    parser.add_argument('--o', dest='output_path', type=str, help='path to output directory')
    parser.add_argument('--e', dest='experiment_name', type=str, help='name of streamline experiment')
    parser.add_argument('--d', dest='dataset_name', type=str, help='dataset name')

    parser.add_argument('--cluster', dest='do_cluster', type=int, default=1)
    parser.add_argument('--m1', dest='memory1', type=int, default=2)
    parser.add_argument('--m2', dest='memory2', type=int, default=3)

    options = parser.parse_args(argv[1:])
    streamline_path = options.streamline_path
    experiment_name = options.experiment_name

    # Check experiment folders and check path validity
    if not os.path.exists(streamline_path):
        raise Exception("Provided streamline_path does not exist")

    for char in experiment_name:
        if not char in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890_':
            raise Exception('Experiment Name must be alphanumeric')

    dataset_name = options.dataset_name
    data_path = streamline_path + '/' + experiment_name + '/' + dataset_name + '/CVDatasets/' 
    model_path = streamline_path + '/' + experiment_name + '/' + dataset_name + '/models/pickledModels/'

    print(streamline_path + '/' + experiment_name + '/' + dataset_name)
    if not os.path.exists(streamline_path + '/' + experiment_name + '/' + dataset_name):
        raise Exception("Provided Dataset does not exist in STREAMLINE Run")


    with open(streamline_path + '/' + experiment_name + '/' + 'metadata.pickle', 'rb') as file:
        metadata = pickle.load(file)

    with open(streamline_path + '/' + experiment_name + '/' + 'algInfo.pickle', 'rb') as file:
        algInfo = pickle.load(file)

    class_label = metadata["Class Label"]
    instance_label = metadata["Instance Label"]
    cv_count = metadata["CV Partitions"]
    random_state = metadata["Random Seed"]
    if not algInfo["ExSTraCS"][0] == True:
        raise Exception("ExSTraCS not run in STREAMLINE")
    
    group_label = None
    visualize_true_clusters = False

    output_path = options.output_path

    experiment_path = output_path + '/' + experiment_name
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    if not os.path.exists(experiment_path):
        os.mkdir(experiment_path)
        os.mkdir(output_path + '/' + experiment_name + '/jobs')
        os.mkdir(output_path + '/' + experiment_name + '/logs')

    do_cluster = options.do_cluster
    memory1 = options.memory1
    memory2 = options.memory2

    # Random seed
    random.seed(random_state)
    np.random.seed(random_state)

    # Read in train and test dfs
    instance_label_map = None
    train_dfs = []
    test_dfs = []
    for i in range(cv_count):
        dataset_train = None
        dataset_test = None
        for file in glob.glob(data_path+'/*_'+str(i)+'_Train.csv'):
            dataset_train = pd.read_csv(file, sep=',')
            dataset_train = dataset_train.loc[:, ~dataset_train.columns.str.contains('^Unnamed')]
            dataset_train = dataset_train.assign(group=np.ones(dataset_train.values.shape[0]))
        for file in glob.glob(data_path+'/*_'+str(i)+'_Test.csv'):
            dataset_test = pd.read_csv(file, sep=',')
            dataset_test = dataset_test.loc[:, ~dataset_test.columns.str.contains('^Unnamed')]
            dataset_test = dataset_test.assign(group=np.ones(dataset_test.values.shape[0]))

        if instance_label == None:
            if i == 0:
                train_len = dataset_train.values.shape[0]
                test_len = dataset_test.values.shape[0]
                dataset_train = dataset_train.assign(instance=np.array(list(range(train_len))))
                dataset_test = dataset_test.assign(instance=np.array(list(range(train_len,train_len+test_len))))
                train_dfs.append(dataset_train)
                test_dfs.append(dataset_test)

                train_values = dataset_train.values
                test_values = dataset_test.values

                instance_label_map = np.concatenate((train_values,test_values),axis=0)
            else:
                train_labels = []
                train_values = dataset_train.values
                for row in train_values:
                    for row2 in instance_label_map:
                        if np.array_equal(row,row2[1:]) and not row2[0] in train_labels:
                            train_labels.append(row2[0])
                dataset_train = dataset_train.assign(instance=np.array(train_labels))
                train_dfs.append(dataset_train)

                test_labels = []
                test_values = dataset_test.values
                for row in test_values:
                    for row2 in instance_label_map:
                        if np.array_equal(row, row2[1:]) and not row2[0] in train_labels and not row2[0] in test_labels:
                            test_labels.append(row2[0])
                dataset_test = dataset_test.assign(instance=np.array(test_labels))
                test_dfs.append(dataset_test)
        else:
            train_dfs.append(dataset_train)
            test_dfs.append(dataset_test)


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

        train_data_features = train_dfs[cv].drop(class_label, axis=1).values
        train_data_phenotypes = train_dfs[cv][class_label].values
        train_instance_labels = train_dfs[cv].index.get_level_values(use_inst_label).tolist()
        train_group_labels = train_dfs[cv].index.get_level_values(use_group_label).tolist()

        test_data_features = test_dfs[cv].drop(class_label, axis=1).values
        test_data_phenotypes = test_dfs[cv][class_label].values
        test_instance_labels = test_dfs[cv].index.get_level_values(use_inst_label).tolist()
        test_group_labels = test_dfs[cv].index.get_level_values(use_group_label).tolist()
        cv_header_save = np.array(list(train_dfs[cv].drop(class_label, axis=1).columns))

        cv_info.append(
            [train_data_features, train_data_phenotypes, train_instance_labels, train_group_labels, test_data_features,
             test_data_phenotypes, test_instance_labels, test_group_labels, use_inst_label, use_group_label,cv_header_save])
        tt_inst += train_instance_labels

    # Find Maximal Data Headers across all CVs
    data_headers = []
    for train_df in train_dfs:
        data_headers.extend(list(train_df.columns))
    data_headers = list(set(data_headers))
    data_headers.remove(class_label)
    data_headers = np.array(data_headers)

    # Find Maximal Original Dataset
    data_features = pd.DataFrame()
    data_headers_left = list(copy.deepcopy(data_headers))
    for i in range(len(train_dfs)):
        full = pd.concat([train_dfs[i],test_dfs[i]])
        cv_headers = list(full.columns)
        for feature_name in cv_headers:
            if feature_name in data_headers_left:
                data_headers_left.remove(feature_name)
                data_features[feature_name] = full[feature_name]
    data_features = data_features.values

    # Get data phenotypes, instance labels, group labels
    full_df = pd.concat([train_dfs[0],test_dfs[0]])
    data_phenotypes = full_df[class_label].values
    full_instance_labels = full_df.index.get_level_values(use_inst_label).tolist()
    full_group_labels = full_df.index.get_level_values(use_group_label).tolist()

    # Group Color Map
    group_colors = {}
    for group_name in set(full_group_labels):
        random_color = randomHex()
        group_colors[group_name] = random_color

    # Export
    full_info = [data_features,data_phenotypes,data_headers,full_instance_labels,full_group_labels,group_colors]
    phase1_pickle = [cv_info, full_info, visualize_true_clusters,'0','0','0','0',random_state,class_label,cv_count,'0','0',model_path]
    outfile = open(experiment_path+'/phase1pickle', 'wb')
    pickle.dump(phase1_pickle, outfile)
    outfile.close()

    # Phase 1 Analysis
    for cv in range(cv_count):
        if do_cluster == 1:
            submitClusterJob(cv,experiment_path,memory1,memory2)
        else:
            submitLocalJob(cv,experiment_path)

    ####################################################################################################################
def submitLocalJob(cv,experiment_path):
    AnalysisPhase1_fromstreamlineJob.job(experiment_path,cv)

def submitClusterJob(cv,experiment_path,memory1,memory2):
    job_ref = str(time.time())
    job_name = experiment_path + '/jobs/' + job_ref + '_run.sh'
    sh_file = open(job_name, 'w')
    sh_file.write('#!/bin/bash\n')
    sh_file.write('#BSUB -J ' + job_ref + '\n')
    sh_file.write('#BSUB -o ' + experiment_path + '/logs/' + job_ref + '.o\n')
    sh_file.write('#BSUB -e ' + experiment_path + '/logs/' + job_ref + '.e\n')

    this_file_path = os.path.dirname(os.path.realpath(__file__))
    sh_file.write('python ' + this_file_path + '/AnalysisPhase1_fromstreamlineJob.py ' + experiment_path + " " + str(cv) + '\n')
    sh_file.close()
    os.system('bsub -q i2c2_normal -R "rusage[mem='+str(memory1)+'G]" -M '+str(memory2)+'G < ' + job_name)
    ####################################################################################################################

def randomHex():
    s = '#'
    for i in range(6):
        s+=random.choice(['0','1','2','3','4','5','6','7','8','9','A','B','C','D','E','F'])
    return s

if __name__ == '__main__':
    sys.exit(main(sys.argv))
