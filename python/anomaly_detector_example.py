import time
from python.AnomalyDetector2 import AnomalyDetectorLearner

print("AnomalyDetector2 started")
time_start = time.time()

anomaly_type = "SwappedArgs"
name_to_vector_file = "token_to_vector.json"
type_to_vector_file = "type_to_vector.json"
node_type_to_vector_file = "node_type_to_vector.json"

training_data_path_pattern = "calls_train_1531142328694.json*"
validation_data_path_pattern = "calls_eval_1531143063866.json*"

# training_data_path_pattern = "calls_train*"
# validation_data_path_pattern = "calls_train*"

anomaly_detector_learner = AnomalyDetectorLearner()
anomaly_detector_learner.set_anomaly_type(anomaly_type)
anomaly_detector_learner.set_embeddings_file_paths(name_to_vector_file, type_to_vector_file, node_type_to_vector_file)
anomaly_detector_learner.set_training_validation_data_paths(training_data_path_pattern, validation_data_path_pattern)

anomaly_detector_learner.read_embeddings_from_files()
anomaly_detector_learner.create_anomaly_detector_of_interest()
anomaly_detector_learner.print_statistics_on_training_data()
anomaly_detector_learner.prepare_xy_pairs_for_learning_and_validation()
anomaly_detector_learner.train_or_load_model()

time_learning_done = time.time()
print("Time for learning (seconds): " + str(round(time_learning_done - time_start)))

anomaly_detector_learner.evaluate_model_on_validation_data()
anomaly_detector_learner.compute_precision_and_recall_with_different_thresholds_for_reporting_anomalies()

time_prediction_done = time.time()
print("Time for prediction (seconds): " + str(round(time_prediction_done - time_learning_done)))