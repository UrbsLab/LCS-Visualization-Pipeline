import sys
import os
import argparse
import time
import random
import pandas as pd
import numpy as np
import copy
import pickle
import AnalysisPhase1_pretrainedJob
import glob

'''Sample Run Code
python AnalysisPhase1_pretrained.py --o /Users/robert/Desktop/outputs/test1/mp6/viz-outputs --e test1 --d /Users/robert/Desktop/outputs/test1/mp6/CVDatasets --m /Users/robert/Desktop/outputs/test1/mp6/training/pickledModels --inst Instance --cv 3 --cluster 0
python AnalysisPhase1_pretrained.py --o /Users/robert/Desktop/outputs/test1/mp11/viz-outputs --e test1 --d /Users/robert/Desktop/outputs/test1/mp11/CVDatasets --m /Users/robert/Desktop/outputs/test1/mp11/training/pickledModels --inst Instance --cv 3 --cluster 0


'''

def main(argv):
    # Parse arguments
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--d', dest='data_path', type=str, help='path to directory containing presplit train/test datasets ending with _CV_Test/Train.csv')
    parser.add_argument('--m', dest='model_path', type=str, help='path to directory containing pretrained ExSTraCS Models labeled ExStraCS_CV')
    parser.add_argument('--o', dest='output_path', type=str, help='path to output directory')
    parser.add_argument('--e', dest='experiment_name', type=str, help='name of experiment (no spaces)')

    parser.add_argument('--class', dest='class_label', type=str, default="Class")
    parser.add_argument('--inst', dest='instance_label', type=str, default="None")

    parser.add_argument('--cv', dest='cv_partitions', type=int, help='number of CV partitions', default=3)
    parser.add_argument('--random-state',dest='random_state',type=str,default='None')

    parser.add_argument('--cluster', dest='do_cluster', type=int, default=1)
    parser.add_argument('--m1', dest='memory1', type=int, default=2)
    parser.add_argument('--m2', dest='memory2', type=int, default=3)

    options = parser.parse_args(argv[1:])
    data_path = options.data_path
    model_path = options.model_path
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
    group_label = None
    visualize_true_clusters = False

    cv_count = options.cv_partitions

    if options.random_state == 'None':
        random_state = random.randint(0, 1000000)
    else:
        random_state = options.random_state
    do_cluster = options.do_cluster
    memory1 = options.memory1
    memory2 = options.memory2

    # Create experiment folders and check path validity
    if not os.path.exists(data_path):
        raise Exception("Provided data_path does not exist")

    for char in experiment_name:
        if not char in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890_':
            raise Exception('Experiment Name must be alphanumeric')

    experiment_path = output_path + '/' + experiment_name
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    if not os.path.exists(experiment_path):
        os.mkdir(experiment_path)
        os.mkdir(output_path + '/' + experiment_name + '/jobs')
        os.mkdir(output_path + '/' + experiment_name + '/logs')

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

        cv_info.append(
            [train_data_features, train_data_phenotypes, train_instance_labels, train_group_labels, test_data_features,
             test_data_phenotypes, test_instance_labels, test_group_labels, use_inst_label, use_group_label])
        tt_inst += train_instance_labels


    full_df = pd.concat([train_dfs[0],test_dfs[0]])
    data_features = full_df.drop(class_label, axis=1).values
    data_phenotypes = full_df[class_label].values
    data_headers = full_df.drop(class_label, axis=1).columns.values
    full_instance_labels = full_df.index.get_level_values(use_inst_label).tolist()
    full_group_labels = full_df.index.get_level_values(use_group_label).tolist()
    group_colors = {}
    for group_name in set(full_group_labels):
        random_color = randomHex()
        group_colors[group_name] = random_color
    full_info = [data_features,data_phenotypes,data_headers,full_instance_labels,full_group_labels,group_colors]

    phase1_pickle = [cv_info, full_info, visualize_true_clusters,'0','0','0','0',random_state,class_label,cv_count,'0','0',model_path]
    outfile = open(experiment_path+'/phase1pickle', 'wb')
    pickle.dump(phase1_pickle, outfile)
    outfile.close()

    for cv in range(cv_count):
        if do_cluster == 1:
            submitClusterJob(cv,experiment_path,memory1,memory2)
        else:
            submitLocalJob(cv,experiment_path)

    ####################################################################################################################
def submitLocalJob(cv,experiment_path):
    AnalysisPhase1_pretrainedJob.job(experiment_path,cv)

def submitClusterJob(cv,experiment_path,memory1,memory2):
    job_ref = str(time.time())
    job_name = experiment_path + '/jobs/' + job_ref + '_run.sh'
    sh_file = open(job_name, 'w')
    sh_file.write('#!/bin/bash\n')
    sh_file.write('#BSUB -J ' + job_ref + '\n')
    sh_file.write('#BSUB -o ' + experiment_path + '/logs/' + job_ref + '.o\n')
    sh_file.write('#BSUB -e ' + experiment_path + '/logs/' + job_ref + '.e\n')

    this_file_path = os.path.dirname(os.path.realpath(__file__))
    sh_file.write('python ' + this_file_path + '/AnalysisPhase1_pretrainedJob.py ' + experiment_path + " " + str(cv) + '\n')
    sh_file.close()
    os.system('bsub -q i2c2_normal -R "rusage[mem='+str(memory1)+'G]" -M '+str(memory2)+'G < ' + job_name)
    ####################################################################################################################

def cv_partitioner(td, cv_partitions, outcomeLabel, randomSeed,method,match_label):
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

    if method == 'stratified':
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
    elif method == 'matched':
        match_index = td.columns.get_loc(match_label)
        match_list = []
        for row in datasetList:
            if row[match_index] not in match_list:
                match_list.append(row[match_index])

        byMatchRows = [[] for i in range(len(match_list))]  # create list of empty lists (one for each match group)
        for row in datasetList:
            # find index in matchList corresponding to the matchset of the current row.
            mIndex = match_list.index(row[match_index])
            row.pop(match_index)  # remove match column from partition output
            byMatchRows[mIndex].append(row)

        currPart = 0
        counter = 0
        for matchSet in byMatchRows:  # Go through each unique set of matched instances
            for row in matchSet:  # put all of the instances
                partList[currPart].append(row)
            # move on to next matchset being placed in the next partition.
            counter += 1
            currPart = counter % cv_partitions

        header.pop(match_index)  # remove match column from partition output

    # Create Output
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

if __name__ == '__main__':
    sys.exit(main(sys.argv))