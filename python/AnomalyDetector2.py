"""
Created on Jun 23, 2017

@author: Michael Pradel
"""

import json
from os.path import join
from os import getcwd
from collections import Counter, namedtuple
import math
from keras.models import Sequential
# from keras.models import load_model
from keras.layers.core import Dense, Dropout
import time
import numpy as np
import glob
from python import Util
from python import LearningDataSwappedArgs
from python import LearningDataBinOperator
from python import LearningDataSwappedBinOperands
from python import LearningDataIncorrectBinaryOperand
from python import LearningDataIncorrectAssignment
from python import LearningDataMissingArg

name_embedding_size = 200
file_name_embedding_size = 50
type_embedding_size = 5

Anomaly = namedtuple("Anomaly", ["message", "score"])


class AnomalyDetectorLearner(object):
    def __init__(self):
        self._anomaly_type = None
        self._option = None
        self._name_to_vector_file = None
        self._type_to_vector_file = None
        self._node_type_to_vector_file = None
        self._training_data_paths = None
        self._validation_data_paths = None
        self._name_to_vector = None
        self._type_to_vector = None
        self._node_type_to_vector = None
        self._learning_data = None
        self._xs_training = None
        self._ys_training = None
        self._x_length = None
        self._model = None
        self._xs_validation = None
        self._code_pieces_validation = None

    def set_anomaly_type(self, anomaly_type):
        self._anomaly_type = anomaly_type

    def set_embeddings_file_paths(self, name_to_vector_file, type_to_vector_file, node_type_to_vector_file):
        self._name_to_vector_file = join(getcwd(), name_to_vector_file)
        self._type_to_vector_file = join(getcwd(), type_to_vector_file)
        self._node_type_to_vector_file = join(getcwd(), node_type_to_vector_file)

    def set_training_validation_data_paths(self, training_data_path_pattern, validation_data_path_pattern):
        self._training_data_paths = self._parse_data_path_pattern(training_data_path_pattern)
        self._validation_data_paths = self._parse_data_path_pattern(validation_data_path_pattern)

    def read_embeddings_from_files(self):
        with open(self._name_to_vector_file) as f:
            self._name_to_vector = json.load(f)
        with open(self._type_to_vector_file) as f:
            self._type_to_vector = json.load(f)
        with open(self._node_type_to_vector_file) as f:
            self._node_type_to_vector = json.load(f)

    def create_anomaly_detector_of_interest(self):
        if self._anomaly_type == "SwappedArgs":
            self._learning_data = LearningDataSwappedArgs.LearningData()
        elif self._anomaly_type == "BinOperator":
            self._learning_data = LearningDataBinOperator.LearningData()
        elif self._anomaly_type == "SwappedBinOperands":
            self._learning_data = LearningDataSwappedBinOperands.LearningData()
        elif self._anomaly_type == "IncorrectBinaryOperand":
            self._learning_data = LearningDataIncorrectBinaryOperand.LearningData()
        elif self._anomaly_type == "IncorrectAssignment":
            self._learning_data = LearningDataIncorrectAssignment.LearningData()
        elif self._anomaly_type == "MissingArg":
            self._learning_data = LearningDataMissingArg.LearningData()
        else:
            raise Exception("Incorrect argument for 'what'")

    def print_statistics_on_training_data(self):
        print("Statistics on training data:")
        self._learning_data.pre_scan(self._training_data_paths, self._validation_data_paths)

    def prepare_xy_pairs_for_learning_and_validation(self):
        # prepare x,y pairs for learning and validation
        print("Preparing xy pairs for training data:")
        self._xs_training, self._ys_training, _ = \
            self._prepare_xy_pairs(
                self._training_data_paths,
                self._learning_data,
                self._name_to_vector,
                self._type_to_vector,
                self._node_type_to_vector)
        self._x_length = len(self._xs_training[0])
        print("Training examples   : " + str(len(self._xs_training)))

    def train_or_load_model(self):
        self._build_and_train_model()

    def _build_and_train_model(self):
        model = self._build_simple_feed_forward_network()
        self._compile_and_train_model(model)
        self._save_model(model)
        self._model = model

    def _build_simple_feed_forward_network(self):
        model = Sequential()
        model.add(Dropout(0.2, input_shape=(self._x_length,)))
        model.add(Dense(200, input_dim=self._x_length, activation="relu", kernel_initializer='normal'))
        model.add(Dropout(0.2))
        # model.add(Dense(200, activation="relu"))
        model.add(Dense(1, activation="sigmoid", kernel_initializer='normal'))
        return model

    def _compile_and_train_model(self, model):
        # train
        model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
        model.fit(self._xs_training, self._ys_training, batch_size=100, epochs=10, verbose=1)

    @staticmethod
    def _save_model(model):
        time_stamp = math.floor(time.time() * 1000)
        model.save("anomaly_detection_model_" + str(time_stamp))

    def evaluate_model_on_validation_data(self):
        print("Preparing xy pairs for validation data:")
        self._xs_validation, ys_validation, self._code_pieces_validation = \
            self._prepare_xy_pairs(
                self._validation_data_paths,
                self._learning_data,
                self._name_to_vector,
                self._type_to_vector,
                self._node_type_to_vector)
        print("Validation examples : " + str(len(self._xs_validation)))

        # validate
        validation_loss = self._model.evaluate(self._xs_validation, ys_validation)
        print()
        print("Validation loss & accuracy: " + str(validation_loss))

    def compute_precision_and_recall_with_different_thresholds_for_reporting_anomalies(self):
        # assumption: correct and swapped arguments are alternating in list of x-y pairs
        threshold_to_correct = Counter()
        threshold_to_incorrect = Counter()
        threshold_to_found_seeded_bugs = Counter()
        threshold_to_warnings_in_orig_code = Counter()
        ys_prediction = self._model.predict(self._xs_validation)
        poss_anomalies = []
        for idx in range(0, len(self._xs_validation), 2):
            y_prediction_orig = ys_prediction[idx][0]  # probab(original code should be changed), expect 0
            y_prediction_changed = ys_prediction[idx + 1][0]  # probab(changed code should be changed), expect 1
            # higher means more likely to be anomaly in current code
            anomaly_score = \
                self._learning_data.anomaly_score(
                    y_prediction_orig,
                    y_prediction_changed)
            # higher means more likely to be correct in current code
            normal_score = \
                self._learning_data.normal_score(
                    y_prediction_orig,
                    y_prediction_changed)
            # is_anomaly = False
            for threshold_raw in range(1, 20, 1):
                threshold = threshold_raw / 20.0
                suggests_change_of_orig = anomaly_score >= threshold
                suggests_change_of_changed = normal_score >= threshold
                # counts for positive example
                if suggests_change_of_orig:
                    threshold_to_incorrect[threshold] += 1
                    threshold_to_warnings_in_orig_code[threshold] += 1
                else:
                    threshold_to_correct[threshold] += 1
                # counts for negative example
                if suggests_change_of_changed:
                    threshold_to_correct[threshold] += 1
                    threshold_to_found_seeded_bugs[threshold] += 1
                else:
                    threshold_to_incorrect[threshold] += 1

                # # check if we found an anomaly in the original code
                # if suggests_change_of_orig:
                #     is_anomaly = True

            # if is_anomaly:
            #     code_piece = self._code_pieces_validation[idx]
            #     message = "Score : " + str(anomaly_score) + " | " + code_piece.to_message()
            #     #             print("Possible anomaly: "+message)
            #     # Log the possible anomaly for future manual inspection
            #     poss_anomalies.append(Anomaly(message, anomaly_score))

        f_inspect = open('poss_anomalies.txt', 'w+')
        poss_anomalies = sorted(poss_anomalies, key=lambda a: -a.score)
        for anomaly in poss_anomalies:
            f_inspect.write(anomaly.message + "\n")
        print("Possible Anomalies written to file : poss_anomalies.txt")
        f_inspect.close()

        print()
        for threshold_raw in range(1, 20, 1):
            threshold = threshold_raw / 20.0
            recall = (threshold_to_found_seeded_bugs[threshold] * 1.0) / (len(self._xs_validation) / 2)
            precision = 1 - ((threshold_to_warnings_in_orig_code[threshold] * 1.0) / (len(self._xs_validation) / 2))
            if threshold_to_correct[threshold] + threshold_to_incorrect[threshold] > 0:
                accuracy = threshold_to_correct[threshold] * 1.0 / (
                            threshold_to_correct[threshold] + threshold_to_incorrect[threshold])
            else:
                accuracy = 0.0
            print("Threshold: " + str(threshold) + "   Accuracy: " + str(round(accuracy, 4)) + "   Recall: " + str(
                round(recall, 4)) + "   Precision: " + str(round(precision, 4)) + "  #Warnings: " + str(
                threshold_to_warnings_in_orig_code[threshold]))

    @staticmethod
    def _parse_data_path_pattern(data_path_pattern):
        data_paths_list = glob.glob(join(getcwd(), data_path_pattern))
        return data_paths_list

    @staticmethod
    def _prepare_xy_pairs(data_paths, learning_data, name_to_vector, type_to_vector, node_type_to_vector):
        xs = []
        ys = []
        code_pieces = []  # keep calls in addition to encoding as x,y pairs (to report detected anomalies)

        for code_piece in Util.DataReader(data_paths):
            learning_data.code_to_xy_pairs(
                code_piece,
                xs,
                ys,
                name_to_vector,
                type_to_vector,
                node_type_to_vector,
                code_pieces)
        x_length = len(xs[0])

        #     print("Stats: " + str(learning_data.stats))
        print("Number of x,y pairs: " + str(len(xs)))
        print("Length of x vectors: " + str(x_length))
        return [np.array(xs), np.array(ys), code_pieces]
