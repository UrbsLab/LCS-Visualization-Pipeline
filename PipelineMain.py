import sys
import os
import argparse
import time
import random
import pandas as pd
import numpy as np
import copy
import pickle
import Utilities
import PipelineCVJob
import PipelineFullJob

'''
Sample Run Commands:
#MP6 problem
python PipelineMain.py --d Datasets/mp6_full.csv --o Outputs --e mp6ruletest --inst Instance --group Group --iter 10000 --N 500 --nu 10

#1 Locus 2 Model Heterogeneity
python PipelineMain.py --d Datasets/one.txt --o Outputs --e ruleOne --group Model --iter 20000 --N 1000 --nu 1
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

    parser.add_argument('--rulepop-method', dest='rulepop_clustering_method', type=str, default='default')

    parser.add_argument('--run-method', dest='run_method',type=str,default='local')

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
        random_state = random.randint(0,1000000)
    else:
        random_state = options.random_state
    rulepop_clustering_method = options.rulepop_clustering_method
    run_method = options.run_method

    # Create experiment folders and check path validity
    if not os.path.exists(data_path):
        raise Exception("Provided data_path does not exist")

    for char in experiment_name:
        if not char in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890':
            raise Exception('Experiment Name must be alphanumeric')

    experiment_path = output_path+'/'+experiment_name
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    if not os.path.exists(experiment_path):
        os.mkdir(experiment_path)
        os.mkdir(experiment_path + '/jobs')
        os.mkdir(experiment_path + '/logs')

    # Write Metadata
    outfile = open(experiment_path + '/metadata', mode='w')
    outfile.write('data_path: ' + str(data_path) + '\n')
    outfile.write('learning_iterations: ' + str(learning_iterations) + '\n')
    outfile.write('N: ' + str(N) + '\n')
    outfile.write('nu: ' + str(nu) + '\n')
    outfile.write('attribute_tracking_method: ' + str(attribute_tracking_method) + '\n')
    outfile.write('rulepop_clustering_method: ' + str(rulepop_clustering_method) + '\n')
    outfile.close()

    #Read in data
    if data_path[-1] == 't': #txt
        dataset = pd.read_csv(data_path, sep='\t')
    else: #csv
        dataset = pd.read_csv(data_path, sep=',')

    #Random seed
    random.seed(random_state)
    np.random.seed(random_state)

    # Remove 'unnamed columns'
    dataset = dataset.loc[:, ~dataset.columns.str.contains('^Unnamed')]
    train_dfs, test_dfs = cv_partitioner(dataset, cv_count, class_label, random_state)

    # Create and save cv information
    to_save = []
    for cv in range(cv_count):
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

        save = [train_data_features,train_data_phenotypes,train_instance_labels,train_group_labels,test_data_features,test_data_phenotypes,test_instance_labels,test_group_labels,data_headers]
        to_save.append(save)

    #Save full dataset info
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
    instance_labels = full_df.index.get_level_values(use_inst_label).tolist()
    group_labels = full_df.index.get_level_values(use_group_label).tolist()

    data_headers = full_df.drop(class_label, axis=1).columns.values
    save = [data_features,data_phenotypes,instance_labels,group_labels,data_headers]
    to_save.append(save)

    group_colors = {}
    for group_name in set(group_labels):
        random_color = Utilities.randomHex()
        group_colors[group_name] = random_color
    to_save.append(group_colors)

    to_save.append([learning_iterations,N,nu,attribute_tracking_method,rulepop_clustering_method,random_state,class_label,use_group_label,use_inst_label])

    for i in range(cv_count+1):
        outfile = open(experiment_path+'/pickledCV_'+str(i), 'wb')
        pickle.dump(to_save,outfile)
        outfile.close()

    #Run parallel jobs
    for cv in range(cv_count+1):
        if cv != cv_count:
            if run_method == 'local':
                run_local_cv(experiment_path,cv)
            elif run_method == 'cluster':
                run_cluster_cv(experiment_path,cv)
        else:
            if run_method == 'local':
                run_local_full(experiment_path,cv)
            elif run_method == 'cluster':
                run_cluster_full(experiment_path,cv)

def run_local_cv(experiment_path,cv):
    PipelineCVJob.job(experiment_path,cv)

def run_cluster_cv(experiment_path,cv):
    job_ref = str(time.time())
    job_name = experiment_path + '/jobs/' + job_ref + '_run.sh'
    sh_file = open(job_name, 'w')
    sh_file.write('#!/bin/bash\n')
    sh_file.write('#BSUB -J ' + job_ref + '\n')
    sh_file.write('#BSUB -o ' + experiment_path + '/logs/' + job_ref + '.o\n')
    sh_file.write('#BSUB -e ' + experiment_path + '/logs/' + job_ref + '.e\n')

    this_file_path = os.path.dirname(os.path.realpath(__file__))
    sh_file.write('python ' + this_file_path + '/PipelineCVJob.py ' + experiment_path + " " + str(cv) + '\n')
    sh_file.close()
    os.system('bsub < ' + job_name)

def run_local_full(experiment_path,cv):
    PipelineFullJob.job(experiment_path,cv)

def run_cluster_full(experiment_path,cv):
    job_ref = str(time.time())
    job_name = experiment_path + '/jobs/' + job_ref + '_run.sh'
    sh_file = open(job_name, 'w')
    sh_file.write('#!/bin/bash\n')
    sh_file.write('#BSUB -J ' + job_ref + '\n')
    sh_file.write('#BSUB -o ' + experiment_path + '/logs/' + job_ref + '.o\n')
    sh_file.write('#BSUB -e ' + experiment_path + '/logs/' + job_ref + '.e\n')

    this_file_path = os.path.dirname(os.path.realpath(__file__))
    sh_file.write('python ' + this_file_path + '/PipelineFullJob.py ' + experiment_path + " " + str(cv) + '\n')
    sh_file.close()
    os.system('bsub < ' + job_name)

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

if __name__ == '__main__':
    sys.exit(main(sys.argv))