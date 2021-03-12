import sys
import os
import argparse
import time
from . import AnalysisPhase2ATJob
from . import AnalysisPhase2RuleJob
from . import AnalysisPhase2NetworkJob
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import math
import pickle

'''Sample Run Code
python AnalysisPhase2.py --o ../Outputs --e mp6 --cluster 0
python AnalysisPhase2.py --o ../Outputs --e mp11 --cluster 0
python AnalysisPhase2.py --o ../Outputs --e mp20 --cluster 0

python AnalysisPhase2.py --o /Users/robert/Desktop/outputs/test1/mp6/viz-outputs --e test1 --cluster 0
python AnalysisPhase2.py --o /Users/robert/Desktop/outputs/test1/mp11/viz-outputs --e test1 --cluster 0
'''

def main(argv):
    # Parse arguments
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--o', dest='output_path', type=str, help='path to output directory')
    parser.add_argument('--e', dest='experiment_name', type=str, help='name of experiment (no spaces)')
    parser.add_argument('--rheight', dest='rule_height_factor', type=float, default=1)
    parser.add_argument('--aheight', dest='at_height_factor', type=float, default=1)
    parser.add_argument('--cluster', dest='do_cluster', type=int, default=1)
    parser.add_argument('--am1', dest='at_memory1', type=int, default=2)
    parser.add_argument('--am2', dest='at_memory2', type=int, default=3)
    parser.add_argument('--rm1', dest='rule_memory1', type=int, default=5)
    parser.add_argument('--rm2', dest='rule_memory2', type=int, default=6)
    parser.add_argument('--nm1', dest='network_memory1', type=int, default=2)
    parser.add_argument('--nm2', dest='network_memory2', type=int, default=3)
    parser.add_argument('--dorule', dest='do_rule', type=int, default=1)

    options = parser.parse_args(argv[1:])

    output_path = options.output_path
    experiment_name = options.experiment_name
    rule_height_factor = options.rule_height_factor
    at_height_factor = options.at_height_factor
    do_cluster = options.do_cluster
    at_memory1 = options.at_memory1
    at_memory2 = options.at_memory2
    rule_memory1 = options.rule_memory1
    rule_memory2 = options.rule_memory2
    network_memory1 = options.network_memory1
    network_memory2 = options.network_memory2
    do_rule_cluster = options.do_rule
    experiment_path = output_path + '/' + experiment_name

    # CV Composite Analysis
    if not os.path.exists(experiment_path + '/Composite'):
        os.mkdir(experiment_path + '/Composite')
        os.mkdir(experiment_path + '/Composite/rulepop')
        os.mkdir(experiment_path + '/Composite/rulepop/ruleclusters')
        os.mkdir(experiment_path + '/Composite/at')
        os.mkdir(experiment_path + '/Composite/at/atclusters')

    if do_cluster == 1:
        submitClusterATJob(experiment_path,at_height_factor,at_memory1,at_memory2)
        submitClusterNetworkJob(experiment_path,network_memory1,network_memory2)
        if do_rule_cluster == 1:
            submitClusterRuleJob(experiment_path,rule_height_factor,rule_memory1,rule_memory2)
    else:
        submitLocalATJob(experiment_path,at_height_factor)
        if do_rule_cluster == 1:
            submitLocalRuleJob(experiment_path,rule_height_factor)
        submitLocalNetworkJob(experiment_path)

    ####################################################################################################################

def submitLocalATJob(experiment_path,at_height_factor):
    AnalysisPhase2ATJob.job(experiment_path,at_height_factor)

def submitLocalRuleJob(experiment_path,rule_height_factor):
    AnalysisPhase2RuleJob.job(experiment_path,rule_height_factor)

def submitLocalNetworkJob(experiment_path):
    AnalysisPhase2NetworkJob.job(experiment_path)

def submitClusterATJob(experiment_path,at_height_factor,m1,m2):
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
    os.system('bsub -q i2c2_normal -R "rusage[mem='+str(m1)+'G]" -M '+str(m2)+'G < ' + job_name)

def submitClusterRuleJob(experiment_path,rule_height_factor,m1,m2):
    job_ref = str(time.time())
    job_name = experiment_path + '/jobs/' + job_ref + '_run.sh'
    sh_file = open(job_name, 'w')
    sh_file.write('#!/bin/bash\n')
    sh_file.write('#BSUB -J ' + job_ref + '\n')
    sh_file.write('#BSUB -o ' + experiment_path + '/logs/' + job_ref + '.o\n')
    sh_file.write('#BSUB -e ' + experiment_path + '/logs/' + job_ref + '.e\n')

    this_file_path = os.path.dirname(os.path.realpath(__file__))
    sh_file.write('python ' + this_file_path + '/AnalysisPhase2RuleJob.py ' + experiment_path + " " + str(rule_height_factor)+'\n')
    sh_file.close()
    os.system('bsub -q i2c2_normal -R "rusage[mem='+str(m1)+'G]" -M '+str(m2)+'G < ' + job_name)

def submitClusterNetworkJob(experiment_path,m1,m2):
    job_ref = str(time.time())
    job_name = experiment_path + '/jobs/' + job_ref + '_run.sh'
    sh_file = open(job_name, 'w')
    sh_file.write('#!/bin/bash\n')
    sh_file.write('#BSUB -J ' + job_ref + '\n')
    sh_file.write('#BSUB -o ' + experiment_path + '/logs/' + job_ref + '.o\n')
    sh_file.write('#BSUB -e ' + experiment_path + '/logs/' + job_ref + '.e\n')

    this_file_path = os.path.dirname(os.path.realpath(__file__))
    sh_file.write('python ' + this_file_path + '/AnalysisPhase2NetworkJob.py ' + experiment_path + '\n')
    sh_file.close()
    os.system('bsub -q i2c2_normal -R "rusage[mem=' + str(m1) + 'G]" -M ' + str(m2) + 'G < ' + job_name)


if __name__ == '__main__':
    sys.exit(main(sys.argv))

