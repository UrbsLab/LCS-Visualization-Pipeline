import pickle
from skExSTraCS import ExSTraCS
import os
import numpy as np
import csv
import time
import sys
from skrebate import MultiSURF

def job(experiment_path,cv):
    job_start_time = time.time()

    file = open(experiment_path+'/phase1pickle', 'rb')
    phase1_pickle =  pickle.load(file)
    file.close()

    cv_info = phase1_pickle[0]
    learning_iterations = phase1_pickle[3]
    N = phase1_pickle[4]
    nu = phase1_pickle[5]
    attribute_tracking_method = phase1_pickle[6]
    random_state = phase1_pickle[7]
    class_label = phase1_pickle[8]
    feature_selection_sample_size = phase1_pickle[10]
    rule_compaction_method = phase1_pickle[11]
    data_headers = phase1_pickle[1][2]

    train_data_features = cv_info[cv][0]
    train_data_phenotypes = cv_info[cv][1]
    train_instance_labels = cv_info[cv][2]
    train_group_labels = cv_info[cv][3]
    test_data_features = cv_info[cv][4]
    test_data_phenotypes = cv_info[cv][5]
    test_instance_labels = cv_info[cv][6]
    test_group_labels = cv_info[cv][7]
    inst_label = cv_info[cv][8]
    group_label = cv_info[cv][9]

    # Create CV directory
    if not os.path.exists(experiment_path + '/CV_' + str(cv)):
        os.mkdir(experiment_path + '/CV_' + str(cv))

    #MultiSURF Feature Scoring
    merged = np.insert(train_data_features, train_data_features.shape[1], train_data_phenotypes, 1)
    rb_sample = np.random.choice(merged.shape[0], min(feature_selection_sample_size,merged.shape[0]), replace=False)
    new_data = []
    for i in rb_sample:
        new_data.append(merged[i])
    new_data = np.array(new_data)
    data_featuresR = np.delete(new_data, -1, axis=1)
    data_phenotypesR = new_data[:, -1]
    featureimportance_model = MultiSURF()
    featureimportance_model.fit(data_featuresR, data_phenotypesR)
    scores = featureimportance_model.feature_importances_

    # Train ExSTraCS Model
    model = ExSTraCS(learning_iterations=learning_iterations, N=N, nu=nu,attribute_tracking_method=attribute_tracking_method,
                     rule_compaction=rule_compaction_method,random_state=random_state,do_correct_set_subsumption=False,expert_knowledge=scores)
    model.fit(train_data_features, train_data_phenotypes)

    outfile = open(experiment_path + '/CV_' + str(cv) + '/model', 'wb')
    pickle.dump(model, outfile)
    outfile.close()

    # Export Testing Accuracy for each instance
    predicted_data_phenotypes = model.predict(test_data_features)
    equality = np.equal(predicted_data_phenotypes, test_data_phenotypes)
    with open(experiment_path + '/CV_' + str(cv) + '/instTestingAccuracy.csv', mode='w') as file:
        writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerow([inst_label, 'isCorrect'])
        for i in range(len(test_instance_labels)):
            writer.writerow([test_instance_labels[i], 1 if equality[i] else 0])
    file.close()

    # Export Aggregate Testing Accuracy
    outfile = open(experiment_path + '/CV_' + str(cv) + '/testingAccuracy.txt', mode='w')
    outfile.write(str(model.score(test_data_features, test_data_phenotypes)))
    outfile.close()

    # Save train and testing datasets into csvs
    with open(experiment_path + '/CV_' + str(cv) + '/trainDataset.csv', mode='w') as file:
        writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(list(data_headers) + [class_label, inst_label, group_label])
        for i in range(len(train_instance_labels)):
            writer.writerow(list(train_data_features[i]) + [train_data_phenotypes[i]] + [train_instance_labels[i]] + [
                train_group_labels[i]])
    file.close()

    with open(experiment_path + '/CV_' + str(cv) + '/testDataset.csv', mode='w') as file:
        writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(list(data_headers) + [class_label, inst_label, group_label])
        for i in range(len(test_instance_labels)):
            writer.writerow(list(test_data_features[i]) + [test_data_phenotypes[i]] + [test_instance_labels[i]] + [
                test_group_labels[i]])
    file.close()

    # Get AT Scores for each instance
    AT_scores = model.get_attribute_tracking_scores(instance_labels=np.array(train_instance_labels))

    # Normalize AT Scores
    normalized_AT_scores = []
    for i in range(len(AT_scores)):
        normalized = AT_scores[i][1]
        max_score = max(normalized)
        for j in range(len(normalized)):
            if max_score != 0:
                normalized[j] /= max_score
            else:
                normalized[j] = 0
        normalized_AT_scores.append(list(normalized))

    # Save Normalized AT Scores
    with open(experiment_path + '/CV_' + str(cv) + '/normalizedATScores.csv', mode='w') as file:
        writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerow([inst_label] + list(data_headers))
        for i in range(len(train_instance_labels)):
            writer.writerow([train_instance_labels[i]] + normalized_AT_scores[i])
    file.close()

    # Save Runtime
    runtime_file = open(experiment_path + '/CV_' + str(cv) + '/runtime.txt', 'w')
    runtime_file.write(str(time.time() - job_start_time))
    runtime_file.close()

    # Print completion
    print('CV '+str(cv) + " phase 1 complete")

if __name__ == '__main__':
    job(sys.argv[1],int(sys.argv[2]))