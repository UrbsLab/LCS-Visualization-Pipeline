import sys
import os
import argparse
import time
import AnalysisPhase2ATJob
import AnalysisPhase2RuleJob
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import math
import pickle

'''Sample Run Code
#MP6 problem
python AnalysisPhase2.py --o Outputs --e mp6v3 --cluster 0

#MP11 problem
python AnalysisPhase2.py --o Outputs --e mp11v3

#MP20 problem
python AnalysisPhase2.py --o Outputs --e mp20v3

#CHOP Dataset
python AnalysisPhase2.py --o /home/robertzh/visualizations/vizoutputs --e choptest1
'''

def main(argv):
    # Parse arguments
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--o', dest='output_path', type=str, help='path to output directory')
    parser.add_argument('--e', dest='experiment_name', type=str, help='name of experiment (no spaces)')
    parser.add_argument('--rulepop-method', dest='rulepop_clustering_method', type=str, default='pearson')
    parser.add_argument('--rheight', dest='rule_height_factor', type=float, default=1)
    parser.add_argument('--aheight', dest='at_height_factor', type=float, default=1)
    parser.add_argument('--nspace', dest='network_space_factor', type=float, default=1)
    parser.add_argument('--cluster', dest='do_cluster', type=int, default=1)
    options = parser.parse_args(argv[1:])

    output_path = options.output_path
    experiment_name = options.experiment_name
    rulepop_clustering_method = options.rulepop_clustering_method
    rule_height_factor = options.rule_height_factor
    at_height_factor = options.at_height_factor
    network_space_factor = options.network_space_factor
    do_cluster = options.do_cluster
    experiment_path = output_path + '/' + experiment_name

    # CV Composite Analysis
    if not os.path.exists(experiment_path + '/Composite'):
        os.mkdir(experiment_path + '/Composite')
        os.mkdir(experiment_path + '/Composite/rulepop')
        os.mkdir(experiment_path + '/Composite/rulepop/ruleclusters')
        os.mkdir(experiment_path + '/Composite/at')
        os.mkdir(experiment_path + '/Composite/at/atclusters')

    if do_cluster == 1:
        submitClusterATJob(experiment_path,at_height_factor)
        submitClusterRuleJob(experiment_path,rulepop_clustering_method,rule_height_factor)
    else:
        submitLocalATJob(experiment_path,at_height_factor)
        submitLocalRuleJob(experiment_path,rulepop_clustering_method,rule_height_factor)

    ####################################################################################################################
    # Load information
    file = open(experiment_path + '/phase1pickle', 'rb')
    phase1_pickle = pickle.load(file)
    file.close()

    full_info = phase1_pickle[1]
    data_headers = full_info[2]
    cv_count = phase1_pickle[9]

    models = []
    for cv in range(cv_count):
        file = open(experiment_path + '/CV_' + str(cv) + '/model', 'rb')
        model = pickle.load(file)
        models.append(model)

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

    pos = nx.spring_layout(G, k=1, scale=network_space_factor)

    to_save = [acc_spec_dict,edge_list,weight_list,pos]
    outfile = open(experiment_path + '/Composite/rulepop/networkpickle', 'wb')
    pickle.dump(to_save, outfile)
    outfile.close()

    max_node_value = max(acc_spec_dict.values())
    for i in acc_spec_dict:
        acc_spec_dict[i] = math.pow(acc_spec_dict[i] / max_node_value, 3) * 1000  # Cubic Node Size Function

    max_weight_value = max(weight_list)
    for i in range(len(weight_list)):
        weight_list[i] = math.pow(weight_list[i] / max_weight_value, 3) * 10  # Cubic Weight Function

    nx.draw_networkx_nodes(G, pos=pos, nodelist=acc_spec_dict.keys(), node_size=[v * 1 for v in acc_spec_dict.values()], node_color='#FF3377')
    nx.draw_networkx_edges(G, pos=pos, edge_color='#E0B8FF', edgelist=edge_list, width=[v * 1 for v in weight_list])
    nx.draw_networkx_labels(G, pos=pos)
    plt.axis('off')
    plt.savefig(experiment_path + '/Composite/rulepop/rulepopGraph.png', dpi=300)
    plt.close('all')


    ####################################################################################################################

def submitLocalATJob(experiment_path,at_height_factor):
    AnalysisPhase2ATJob.job(experiment_path,at_height_factor)

def submitLocalRuleJob(experiment_path,rulepop_clustering_method,rule_height_factor):
    AnalysisPhase2RuleJob.job(experiment_path,rulepop_clustering_method,rule_height_factor)

def submitClusterATJob(experiment_path,at_height_factor):
    job_ref = str(time.time())
    job_name = experiment_path + '/jobs/' + job_ref + '_run.sh'
    sh_file = open(job_name, 'w')
    sh_file.write('#!/bin/bash\n')
    sh_file.write('#BSUB -J ' + job_ref + '\n')
    sh_file.write('#BSUB -o ' + experiment_path + '/logs/' + job_ref + '.o\n')
    sh_file.write('#BSUB -e ' + experiment_path + '/logs/' + job_ref + '.e\n')

    this_file_path = os.path.dirname(os.path.realpath(__file__))
    sh_file.write('python ' + this_file_path + '/AnalysisPhase2ATJob.py ' + experiment_path + ' '+ str(at_height_factor)+'\n')
    sh_file.close()
    os.system('bsub < ' + job_name)

def submitClusterRuleJob(experiment_path,rulepop_clustering_method,rule_height_factor):
    job_ref = str(time.time())
    job_name = experiment_path + '/jobs/' + job_ref + '_run.sh'
    sh_file = open(job_name, 'w')
    sh_file.write('#!/bin/bash\n')
    sh_file.write('#BSUB -J ' + job_ref + '\n')
    sh_file.write('#BSUB -o ' + experiment_path + '/logs/' + job_ref + '.o\n')
    sh_file.write('#BSUB -e ' + experiment_path + '/logs/' + job_ref + '.e\n')

    this_file_path = os.path.dirname(os.path.realpath(__file__))
    sh_file.write('python ' + this_file_path + '/AnalysisPhase2RuleJob.py ' + experiment_path + " " + str(rulepop_clustering_method) + ' '+str(rule_height_factor)+'\n')
    sh_file.close()
    os.system('bsub < ' + job_name)

if __name__ == '__main__':
    sys.exit(main(sys.argv))



