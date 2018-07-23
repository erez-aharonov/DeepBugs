import time
from python.AnomalyDetector2 import AnomalyDetectorTrainee

print("AnomalyDetector2 started")
time_start = time.time()

# do_load_previous_data = False
do_load_previous_data = True

max_files_list_length = 1

anomaly_type = "SwappedArgs"
name_to_vector_file = "token_to_vector.json"
type_to_vector_file = "type_to_vector.json"
node_type_to_vector_file = "node_type_to_vector.json"

# training_data_path_pattern = "calls_train_1531142328694.json*"
# validation_data_path_pattern = "calls_eval_1531143063866.json*"

training_data_path_pattern = "calls_train*"
validation_data_path_pattern = "calls_eval*"

anomaly_detector_trainee = AnomalyDetectorTrainee()
anomaly_detector_trainee.set_max_files_list_length(max_files_list_length)
anomaly_detector_trainee.set_anomaly_type(anomaly_type)
anomaly_detector_trainee.set_embeddings_file_paths(name_to_vector_file, type_to_vector_file, node_type_to_vector_file)
anomaly_detector_trainee.set_training_validation_data_paths(training_data_path_pattern, validation_data_path_pattern)
anomaly_detector_trainee.set_do_load_previous_data(do_load_previous_data)

anomaly_detector_trainee.prepare_for_training_and_validation()

anomaly_detector_trainee.train_or_load_model()

time_learning_done = time.time()
print("Time for learning (seconds): " + str(round(time_learning_done - time_start)))

anomaly_detector_trainee.evaluate_model_on_validation_data()
anomaly_detector_trainee.compute_precision_and_recall_with_different_thresholds_for_reporting_anomalies()

time_prediction_done = time.time()
print("Time for prediction (seconds): " + str(round(time_prediction_done - time_learning_done)))