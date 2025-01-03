
import os
import sys
import shutil

def main(argv):
    datasets = ['d1','d2','d3','d4','d5','d6','d7','d8','d9','d10','d11','d12','d13','d14','d15','d16','d17','d18','d19',
                'd20','d21','d21a','d22','d23','dietadj','dietadjmatched','epionly','mp6','mp11','mp20','mp37','mp70']

    for dataset in datasets:
        for cv_i in range(10):
            os.remove(dataset + '/CV_' + str(cv_i) + '/model')
            os.remove(dataset + '/CV_' + str(cv_i) + '/trainDataset.csv')
            os.remove(dataset + '/CV_' + str(cv_i) + '/testDataset.csv')
        for cluster_i in range(101,600):
            if os.path.exists(dataset + '/Composite/at/atclusters/'+str(cluster_i)+'_clusters'):
                shutil.rmtree(dataset + '/Composite/at/atclusters/'+str(cluster_i)+'_clusters/')
        for cluster_i in range(51,600):
            if os.path.exists(dataset + '/Composite/rulepop/ruleclusters/'+str(cluster_i)+'_clusters'):
                shutil.rmtree(dataset + '/Composite/rulepop/ruleclusters/'+str(cluster_i)+'_clusters/')

if __name__ == '__main__':
    sys.exit(main(sys.argv))
