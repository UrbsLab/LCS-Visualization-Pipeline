import pickle
import networkx as nx
import matplotlib.pyplot as plt
import math
import sys
import numpy as np

def job(experiment_path):
    file = open(experiment_path + '/phase1pickle', 'rb')
    phase1_pickle = pickle.load(file)
    file.close()

    full_info = phase1_pickle[1]
    data_headers = full_info[2]
    cv_count = phase1_pickle[9]
    cv_info = phase1_pickle[0]

    models = []
    for cv in range(cv_count):
        file = open(experiment_path + '/CV_' + str(cv) + '/model', 'rb')
        model = pickle.load(file)
        models.append(model)

    # Rule Specificity Network
    attribute_acc_specificity_counts = np.zeros(len(data_headers))
    merged_attribute_cooccurrences = []
    cv_counter = 0
    for model in models:
        cv_headers = cv_info[cv_counter][10]
        cv_attribute_map = {}
        for feature_name in cv_headers:
            cv_attribute_map[feature_name] = np.where(data_headers == feature_name)[0][0]

        cv_spec_list = model.get_final_attribute_specificity_list()
        transformed_cv_spec_list = [0]*len(data_headers)
        for c in range(len(cv_headers)):
            feature_name = cv_headers[c]
            transformed_cv_spec_list[cv_attribute_map[feature_name]] = cv_spec_list[c]

        attribute_acc_specificity_counts += np.array(transformed_cv_spec_list)
        attribute_cooccurrences = model.get_final_attribute_coocurrences(cv_headers, len(cv_headers))
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
        cv_counter += 1

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

    pos = nx.spring_layout(G, k=1)

    to_save = [acc_spec_dict, edge_list, weight_list, pos]
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

if __name__ == '__main__':
    job(sys.argv[1])