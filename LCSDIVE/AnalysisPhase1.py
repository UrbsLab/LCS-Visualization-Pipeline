import sys
import os
import argparse
import time
import random
import pandas as pd
import numpy as np
import copy
import pickle
import AnalysisPhase1Job

'''Sample Run Code
python AnalysisPhase1.py --d ../Datasets/mp6_full.csv --o ../Outputs --e mp6 --inst Instance --group Group --iter 20000 --N 500 --nu 10 --cluster 0
python AnalysisPhase1.py --d ../Datasets/mp11_full.csv --o ../Outputs --e mp11 --inst Instance --group Group --iter 20000 --N 1000 --nu 10 --cluster 0
python AnalysisPhase1.py --d ../Datasets/mp20_full.csv --o ../Outputs --e mp20 --inst Instance --group Group --iter 100000 --N 2000 --nu 10 --cluster 0
'''

def main(argv):
    # Parse arguments
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--d', dest='data_path', type=str, help='path to the dataset')
    parser.add_argument('--o', dest='output_path', type=str, help='path to output directory')
    parser.add_argument('--e', dest='experiment_name', type=str, help='name of experiment (no spaces)')

    parser.add_argument('--class', dest='class_label', type=str, default="Class")
    parser.add_argument('--inst', dest='instance_label', type=str, default="None")
    parser.add_argument('--group', dest='group_label', type=str, default="None")
    parser.add_argument('--match', dest='match_label', type=str, default="None")

    parser.add_argument('--cv', dest='cv_partitions', type=int, help='number of CV partitions', default=3)

    parser.add_argument('--iter', dest='learning_iterations', type=int, default=16000)
    parser.add_argument('--N', dest='N', type=int, default=1000)
    parser.add_argument('--nu', dest='nu', type=int, default=1)
    parser.add_argument('--at-method', dest='attribute_tracking_method', type=str, default='wh')
    parser.add_argument('--rc', dest='rule_compaction_method', type=str, default='None')
    parser.add_argument('--random-state',dest='random_state',type=str,default='None')
    parser.add_argument('--fssample', dest='feature_selection_sample_size', type=int, default=1000)

    parser.add_argument('--cluster', dest='do_cluster', type=int, default=1)
    parser.add_argument('--m1', dest='memory1', type=int, default=2)
    parser.add_argument('--m2', dest='memory2', type=int, default=3)

    options = parser.parse_args(argv[1:])
    data_path = options.data_path
    output_path = options.output_path
    experiment_name = options.experiment_name
    feature_selection_sample_size = options.feature_selection_sample_size
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
    if options.match_label == 'None':
        match_label = None
        cv_method = 'stratified'
    else:
        match_label = options.match_label
        cv_method = 'matched'

    cv_count = options.cv_partitions
    learning_iterations = options.learning_iterations
    N = options.N
    nu = options.nu
    attribute_tracking_method = options.attribute_tracking_method

    if options.rule_compaction_method == 'None':
        rule_compaction_method = None
    else:
        rule_compaction_method = options.rule_compaction_method

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

    # Write Metadata
    outfile = open(experiment_path + '/trainingMetadata', mode='w')
    outfile.write('data_path: ' + str(data_path) + '\n')
    outfile.write('CV: ' + str(cv_count) + '\n')
    outfile.write('learning_iterations: ' + str(learning_iterations) + '\n')
    outfile.write('N: ' + str(N) + '\n')
    outfile.write('nu: ' + str(nu) + '\n')
    outfile.write('attribute_tracking_method: ' + str(attribute_tracking_method) + '\n')
    outfile.write('random_state: ' + str(random_state) + '\n')
    outfile.write('multisurf sample size: ' + str(feature_selection_sample_size) + '\n')
    outfile.write('rule compaction method: ' + str(options.rule_compaction_method) + '\n')
    outfile.close()

    # Read in data
    if data_path[-1] == 't':  # txt
        dataset = pd.read_csv(data_path, sep='\t')
    elif data_path[-1] == 'v':  # csv
        dataset = pd.read_csv(data_path, sep=',')
    elif data_path[-1] == 'z':  # .txt.gz
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

    train_dfs, test_dfs = cv_partitioner(dataset, cv_count, class_label, random_state,cv_method,match_label)

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

    # Group Colors and Instance Labels
    full_df = copy.deepcopy(dataset)
    if match_label != None:
        full_df = full_df.drop(match_label,axis=1)
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
    full_info = [data_features,data_phenotypes,data_headers,full_instance_labels,full_group_labels,group_colors]

    phase1_pickle = [cv_info, full_info, visualize_true_clusters,learning_iterations,N,nu,attribute_tracking_method,random_state,class_label,cv_count,feature_selection_sample_size,rule_compaction_method]
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
    AnalysisPhase1Job.job(experiment_path,cv)

def submitClusterJob(cv,experiment_path,memory1,memory2):
    job_ref = str(time.time())
    job_name = experiment_path + '/jobs/' + job_ref + '_run.sh'
    sh_file = open(job_name, 'w')
    sh_file.write('#!/bin/bash\n')
    sh_file.write('#BSUB -J ' + job_ref + '\n')
    sh_file.write('#BSUB -o ' + experiment_path + '/logs/' + job_ref + '.o\n')
    sh_file.write('#BSUB -e ' + experiment_path + '/logs/' + job_ref + '.e\n')

    this_file_path = os.path.dirname(os.path.realpath(__file__))
    sh_file.write('python ' + this_file_path + '/AnalysisPhase1Job.py ' + experiment_path + " " + str(cv) + '\n')
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