"""
Created on Jun 23, 2017

@author: Michael Pradel
"""

import json
from os.path import join
from os import getcwd
from collections import namedtuple
import math
from keras.models import Sequential
# from keras.models import load_model
from keras.layers.core import Dense, Dropout
import time
import numpy as np
import glob
import pickle
from python import Util
from python import LearningDataSwappedArgs
from python import LearningDataBinOperator
from python import LearningDataSwappedBinOperands
from python import LearningDataIncorrectBinaryOperand
from python import LearningDataIncorrectAssignment
from python import LearningDataMissingArg
import pandas as pd


pd.options.display.max_colwidth = 200


name_embedding_size = 200
file_name_embedding_size = 50
type_embedding_size = 5

Anomaly = namedtuple("Anomaly", ["message", "score"])


class AnomalyDetectorTrainee(object):
    def __init__(self):
        self._max_files_list_length = None
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
        self._ys_validation = None
        self._code_pieces_validation = None
        self._do_load_previous_data = False
        self._train_xy_code_pieces_pickle_file_path = "train_xy_code_pieces.pickle"
        self._validate_xy_code_pieces_pickle_file_path = "validate_xy_code_pieces.pickle"
        self._data_dump_path = "data_dump.pickle"

    def set_max_files_list_length(self, max_files_list_length):
        self._max_files_list_length = max_files_list_length

    def set_anomaly_type(self, anomaly_type):
        self._anomaly_type = anomaly_type

    def set_embeddings_file_paths(self, name_to_vector_file, type_to_vector_file, node_type_to_vector_file):
        self._name_to_vector_file = join(getcwd(), name_to_vector_file)
        self._type_to_vector_file = join(getcwd(), type_to_vector_file)
        self._node_type_to_vector_file = join(getcwd(), node_type_to_vector_file)

    def set_training_validation_data_paths(self, training_data_path_pattern, validation_data_path_pattern):
        self._training_data_paths = self._parse_data_path_pattern(training_data_path_pattern)
        self._validation_data_paths = self._parse_data_path_pattern(validation_data_path_pattern)

    def set_do_load_previous_data(self, do_load_previous_data):
        self._do_load_previous_data = do_load_previous_data

    def prepare_for_training_and_validation(self):
        if not self._do_load_previous_data:
            self._read_embeddings_from_files()
            self._create_anomaly_detector_of_interest()
            self._print_statistics_on_training_data()
            self._prepare_xy_pairs_for_training()
            self._prepare_xy_pairs_for_validation()
            self._dump_data()
        else:
            self._load_previous_data()

    def _dump_data(self):
        data_to_dump = [
            self._learning_data,
            self._xs_training,
            self._ys_training,
            self._xs_validation,
            self._ys_validation,
            self._code_pieces_validation]
        pickle.dump(data_to_dump, open(self._data_dump_path, "wb"))

    def _read_embeddings_from_files(self):
        with open(self._name_to_vector_file) as f:
            self._name_to_vector = json.load(f)
        with open(self._type_to_vector_file) as f:
            self._type_to_vector = json.load(f)
        with open(self._node_type_to_vector_file) as f:
            self._node_type_to_vector = json.load(f)

    def _create_anomaly_detector_of_interest(self):
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

    def _print_statistics_on_training_data(self):
        print("Statistics on training data:")
        self._learning_data.pre_scan(self._training_data_paths, self._validation_data_paths)

    def _prepare_xy_pairs_for_training(self):
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
        print("saving to pickle ...")
        pickle.dump([self._xs_training, self._ys_training], open(self._train_xy_code_pieces_pickle_file_path, "wb"))

    def _load_previous_data(self):
        self._learning_data, \
        self._xs_training, \
        self._ys_training, \
        self._xs_validation, \
        self._ys_validation, \
        self._code_pieces_validation = \
            pickle.load(open(self._data_dump_path, "rb"))
        self._x_length = len(self._xs_training[0])

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
        validation_loss = self._model.evaluate(self._xs_validation, self._ys_validation)
        print()
        print("Validation loss & accuracy: " + str(validation_loss))

    def _prepare_xy_pairs_for_validation(self):
        print("Preparing xy pairs for validation data:")
        self._xs_validation, self._ys_validation, self._code_pieces_validation = \
            self._prepare_xy_pairs(
                self._validation_data_paths,
                self._learning_data,
                self._name_to_vector,
                self._type_to_vector,
                self._node_type_to_vector)
        print("Validation examples : " + str(len(self._xs_validation)))
        data_to_dump = [self._xs_validation, self._ys_validation, self._code_pieces_validation]
        # data_to_dump = [self._xs_validation, self._ys_validation]
        print("saving to pickle ...")
        pickle.dump(data_to_dump, open(self._validate_xy_code_pieces_pickle_file_path, "wb"))

    def compute_precision_and_recall_with_different_thresholds_for_reporting_anomalies(self):

        # assumption: correct and swapped arguments are alternating in list of x-y pairs
        ys_prediction = self._model.predict(self._xs_validation)

        anomaly_df = self._create_anomaly_data_frame(ys_prediction)

        anomaly_df = self._set_anomaly_and_normal_scores_for_code_pieces(anomaly_df)

        self._print_scores_statistics(anomaly_df)
        self._create_html_for_inspection(anomaly_df)
        self._save_anomaly_df_to_pickle(anomaly_df)

    def _set_anomaly_and_normal_scores_for_code_pieces(self, anomaly_df):
        def get_scores(anomaly_series):
            anomaly_score = \
                self._learning_data.anomaly_score(
                    anomaly_series.probability_original,
                    anomaly_series.probability_swapped)
            normal_score = \
                self._learning_data.normal_score(
                    anomaly_series.probability_original,
                    anomaly_series.probability_swapped)
            return pd.Series([anomaly_score, normal_score], index=["anomaly_score", "normal_score"])

        scores_df = anomaly_df.apply(get_scores, axis=1)
        anomaly_df = pd.concat([anomaly_df, scores_df], axis=1)
        return anomaly_df

    @staticmethod
    def _print_scores_statistics(anomaly_df):
        score_threshold_array = np.array(range(1, 20, 1)) / 20.0
        anomaly_score_df = \
            anomaly_df.apply(
                lambda x: pd.Series(x.anomaly_score > score_threshold_array, index=score_threshold_array),
                axis=1)
        normal_score_df = \
            anomaly_df.apply(
                lambda x: pd.Series(x.normal_score > score_threshold_array, index=score_threshold_array),
                axis=1)
        threshold_to_correct_series = (~anomaly_score_df).sum() + normal_score_df.sum()
        threshold_to_incorrect_series = anomaly_score_df.sum() + (~normal_score_df).sum()
        threshold_to_warnings_in_orig_code_series = anomaly_score_df.sum()
        threshold_to_found_seeded_bugs_series = normal_score_df.sum()
        print("threshold_to_correct_series")
        print(threshold_to_correct_series)
        print("threshold_to_incorrect_series")
        print(threshold_to_incorrect_series)
        print("threshold_to_warnings_in_orig_code_series")
        print(threshold_to_warnings_in_orig_code_series)
        print("threshold_to_found_seeded_bugs_series")
        print(threshold_to_found_seeded_bugs_series)

    @staticmethod
    def _create_html_for_inspection(anomaly_df):
        anomaly_df["message"] = anomaly_df.code_piece.apply(lambda x: x.to_message())
        classified_anomaly_df = anomaly_df[anomaly_df.anomaly_score > 0]
        to_inspect_df = \
            classified_anomaly_df[["anomaly_score", "message"]].sort_values(
                by="anomaly_score",
                ascending=False)
        html_file_path = 'possible_anomalies.html'
        open(html_file_path, 'w+').write(to_inspect_df.to_html())
        print("created {}".format(html_file_path))

    def _create_anomaly_data_frame(self, ys_prediction):
        ys_validation = self._ys_validation.flatten()
        ys_prediction = ys_prediction.flatten()
        ys_validation_original = ys_validation[::2]
        ys_validation_swapped = ys_validation[1::2]
        ys_prediction_original = ys_prediction[::2]
        ys_prediction_swapped = ys_prediction[1::2]
        code_pieces_validation_unique = self._code_pieces_validation[::2]
        anomaly_df = pd.DataFrame(
            [ys_validation_original, ys_prediction_original, ys_validation_swapped, ys_prediction_swapped,
             code_pieces_validation_unique]).T
        anomaly_df.columns = \
            ["label_original",
             "probability_original",
             "label_swapped",
             "probability_swapped",
             "code_piece"]
        return anomaly_df

    def _parse_data_path_pattern(self, data_path_pattern):
        data_paths_list = glob.glob(join(getcwd(), data_path_pattern))[:self._max_files_list_length]
        return data_paths_list

    @staticmethod
    def _prepare_xy_pairs(data_paths, learning_data, name_to_vector, type_to_vector, node_type_to_vector):
        xs_list = []
        ys_list = []
        code_pieces_list = []  # keep calls in addition to encoding as x,y pairs (to report detected anomalies)

        for code_piece in Util.DataReader(data_paths):
            learning_data.code_to_xy_pairs(
                code_piece,
                xs_list,
                ys_list,
                name_to_vector,
                type_to_vector,
                node_type_to_vector,
                code_pieces_list)
        x_length = len(xs_list[0])

        print("Number of x,y pairs: " + str(len(xs_list)))
        print("Length of x vectors: " + str(x_length))
        return [np.array(xs_list), np.array(ys_list), code_pieces_list]

    @staticmethod
    def _save_anomaly_df_to_pickle(anomaly_df):
        anomaly_df_pickle_file_path = "anomaly_df.pickle"
        pickle.dump(anomaly_df, open(anomaly_df_pickle_file_path, "wb"))
        print("saved anomaly data frame into {}".format(anomaly_df_pickle_file_path))
