from spta.classifier.dnn import ClassifierDNNParams


def classifier_experiments():

    # ClassifierDNNParams(<name>, <model_path>, <labels_path> <window_size>)

    experiment_id = {
        'dnn': ClassifierDNNParams('lstm_wSW_label_k-seed-ci', 'lstm/model_lstm_wSW_label_k-seed-ci.h5', 'lstm/labels_k-seed-ci.csv', 15),
        'dnnRV': ClassifierDNNParams('lstmRV_wSW_label_k-seed-ci', 'lstm/model_lstmRV_wSW_label_k-seed-ci.h5', 'lstm/labels_k-seed-ci.csv', 15),
    }

    return experiment_id
