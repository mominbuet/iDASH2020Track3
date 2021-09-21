# Copyright 2019, The TensorFlow Federated Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Trains and evaluates an EMNIST classification model with DP-FedAvg."""

import collections

import numpy as np
import tensorflow as tf
import tensorflow_federated as tff
from absl import app
from absl import flags
from absl import logging
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Conv1D, Input, BatchNormalization

from Utils import getData
from research.utils import training_loop
from research.utils import training_utils
from research.utils import utils_impl

tf.compat.v1.disable_eager_execution()

with utils_impl.record_hparam_flags():
    # Experiment hyperparameters
    flags.DEFINE_enum(
        'model', 'cnn', ['cnn', '2nn'], 'Which model to use. This '
                                        'can be a convolutional model (cnn) or a two hidden-layer '
                                        'densely connected network (2nn).')
    flags.DEFINE_integer('client_batch_size', 10,
                         'Batch size used on the client.')
    flags.DEFINE_integer('clients_per_round', 2,
                         'How many clients to sample per round.')
    flags.DEFINE_integer(
        'client_epochs_per_round', 10,
        'Number of client (inner optimizer) epochs per federated round.')
    flags.DEFINE_boolean(
        'uniform_weighting', True,
        'Whether to weigh clients uniformly. If false, clients '
        'are weighted by the number of samples.')

    # Optimizer configuration (this defines one or more flags per optimizer).
    utils_impl.define_optimizer_flags('server', 'sgd', 1.0)
    utils_impl.define_optimizer_flags('client', 'sgd', 0.02)

    # Differential privacy flags
    flags.DEFINE_float('clip', 0.05, 'Initial clip.')
    flags.DEFINE_float('noise_multiplier', None,
                       'Noise multiplier. If None, no DP is used.')
    flags.DEFINE_float('adaptive_clip_learning_rate', 0,
                       'Adaptive clip learning rate.')
    flags.DEFINE_float('target_unclipped_quantile', 0.5,
                       'Target unclipped quantile.')
    flags.DEFINE_float(
        'clipped_count_budget_allocation', 0.1,
        'Fraction of privacy budget to allocate for clipped counts.')
    flags.DEFINE_boolean(
        'per_vector_clipping', False, 'Use per-vector clipping'
                                      'to indepednelty clip each weight tensor instead of the'
                                      'entire model.')

with utils_impl.record_new_flags() as training_loop_flags:
    flags.DEFINE_integer('total_rounds', 10000, 'Number of total training rounds.')
    flags.DEFINE_string(
        'experiment_name', 'exp_bctcga', 'The name of this experiment. Will be append to '
                                         '--root_output_dir to separate experiment results.')
    flags.DEFINE_string('root_output_dir', './differential_privacy_log/',
                        'Root directory for writing experiment output.')
    flags.DEFINE_boolean(
        'write_metrics_with_bz2', True, 'Whether to use bz2 '
                                        'compression when writing output metrics to a csv file.')
    flags.DEFINE_integer(
        'rounds_per_eval', 1,
        'How often to evaluate the global model on the validation dataset.')
    flags.DEFINE_integer('rounds_per_checkpoint', 50,
                         'How often to checkpoint the global model.')
    flags.DEFINE_integer(
        'rounds_per_profile', 0,
        '(Experimental) How often to run the experimental TF profiler, if >0.')

FLAGS = flags.FLAGS


# End of hyperparameter flags.

def create_keras_model(input_dimension, act='relu'):
    x1 = Input(shape=(input_dimension,))
    v1 = tf.keras.layers.Reshape((-1, 1))(x1)

    for i in (64, 128, 256, 512):
        v1 = Conv1D(i, kernel_size=9, padding='same', strides=2)(v1)
        v1 = BatchNormalization()(v1)
        v1 = tf.keras.layers.Activation(act)(v1)

    v1 = Conv1D(1, kernel_size=1, padding='same')(v1)
    v1 = tf.keras.layers.Activation('sigmoid')(v1)
    v1 = tf.keras.layers.GlobalAveragePooling1D()(v1)

    model = tf.keras.models.Model(inputs=x1, outputs=v1)

    return model


def main(argv):
    if len(argv) > 1:
        raise app.UsageError('Expected no command-line arguments, '
                             'got: {}'.format(argv))
    dataset = 'BC-TCGA'
    dataX, dataY = getData(dataset)
    # dataX = np.reshape (dataX, (len(dataX),len(dataX[0]),1))
    dataTrainX, dataTestX, dataTrainY, dataTestY = train_test_split(dataX, dataY, train_size=int(len(dataX) * 0.8))
    dim = len(dataX[0])
    dataTrainY = dataTrainY.astype(np.int32)
    dataTestY = dataTestY.astype(np.int32)

    def preprocess_train_dataset(dataset):
        def batch_format_fn(element):
            """Flatten a batch `gene` and return the features as an `OrderedDict`."""
            return collections.OrderedDict(
                x=tf.reshape(element['gene'], [-1, dim]),
                y=tf.reshape(element['label'], [-1, 1]))

        return (dataset
                # Repeat to do multiple local epochs
                .repeat(FLAGS.client_epochs_per_round)
                # Shuffle according to the largest client dataset
                .shuffle(buffer_size=len(dataTrainY))
                # Batch to a fixed client batch size
                .batch(FLAGS.client_batch_size, drop_remainder=False)
                # Take a maximum number of batches
                .take(-1).map(batch_format_fn).prefetch(10))

    def preprocess_test_dataset(dataset):
        """Preprocessing function for the EMNIST testing dataset."""
        return (dataset.batch(500, drop_remainder=False))

    split = 2
    data_per_set = int(np.floor(len(dataTrainX) / split))

    client_train_dataset = collections.OrderedDict()
    for i in range(1, split + 1):
        client_name = "client_" + str(i)
        start = data_per_set * (i - 1)
        end = data_per_set * i

        print(f"Adding data from {start} to {end} for client : {client_name}")
        data = collections.OrderedDict((('label', dataTrainY[start:end]), ('gene', dataTrainX[start:end])))
        client_train_dataset[client_name] = data

    train_dataset = tff.simulation.FromTensorSlicesClientData(client_train_dataset)
    client_train_dataset = train_dataset.create_tf_dataset_for_client(train_dataset.client_ids[0])

    preprocessed_sample_dataset = preprocess_train_dataset(client_train_dataset)

    # test_dataset = tff.simulation.FromTensorSlicesClientData(client_train_dataset)
    # test_dataset = preprocess_test_dataset(
    #     client_train_dataset.create_tf_dataset_from_all_clients())

    # sample_batch = tf.nest.map_structure(lambda x: x.numpy(), next(iter(preprocessed_sample_dataset)))

    # tensor_slices_dict = {'a': collections.OrderedDict((('label', dataTestY), ('gene', dataTestX))),
    #                       'b': collections.OrderedDict((('label', dataTestY), ('gene', dataTestX)))}
    # emnist_test = tff.simulation.FromTensorSlicesClientData(tensor_slices_dict)
    # emnist_test = emnist_test.create_tf_dataset_for_client(emnist_test.client_ids[0])
    # emnist_test = preprocess_train_dataset(emnist_test)

    # def make_federated_data(client_data, client_ids):
    #     return [preprocess_train_dataset(client_data.create_tf_dataset_for_client(x)) for x in client_ids]
    #
    # federated_train_data = make_federated_data(train_dataset , train_dataset .client_ids)
    # #
    # print('Number of client datasets: {l}'.format(l=len(federated_train_data)))
    # print('First dataset: {d}'.format(d=federated_train_data[0]))

    # emnist_train, emnist_test = emnist_dataset.get_emnist_datasets(
    #     FLAGS.client_batch_size, FLAGS.client_epochs_per_round, only_digits=False)

    # if FLAGS.model == 'cnn':
    #     model_builder = functools.partial(
    #         emnist_models.create_conv_dropout_model, only_digits=False)
    # elif FLAGS.model == '2nn':
    #     model_builder = functools.partial(
    #         emnist_models.create_two_hidden_layer_model, only_digits=False)
    # else:
    #     raise ValueError('Cannot handle model flag [{!s}].'.format(FLAGS.model))

    loss_builder = tf.keras.losses.SparseCategoricalCrossentropy
    metrics_builder = lambda: [tf.keras.metrics.SparseCategoricalAccuracy()]

    if FLAGS.uniform_weighting:

        def client_weight_fn(local_outputs):
            del local_outputs
            return 1.0

    else:
        client_weight_fn = None  # Defaults to the number of examples per client.

    keras_model = create_keras_model(input_dimension=dim)

    def model_fn():
        return tff.learning.from_keras_model(
            # model_builder(),
            keras_model,
            input_spec=preprocessed_sample_dataset.element_spec,
            loss=loss_builder(),
            metrics=metrics_builder())

    if FLAGS.noise_multiplier is not None:
        if not FLAGS.uniform_weighting:
            raise ValueError(
                'Differential privacy is only implemented for uniform weighting.')

        dp_query = tff.utils.build_dp_query(
            clip=FLAGS.clip,
            noise_multiplier=FLAGS.noise_multiplier,
            expected_total_weight=FLAGS.clients_per_round,
            adaptive_clip_learning_rate=FLAGS.adaptive_clip_learning_rate,
            target_unclipped_quantile=FLAGS.target_unclipped_quantile,
            clipped_count_budget_allocation=FLAGS.clipped_count_budget_allocation,
            expected_clients_per_round=FLAGS.clients_per_round,
            per_vector_clipping=FLAGS.per_vector_clipping,
            model=model_fn())

        weights_type = tff.learning.framework.weights_type_from_model(model_fn)
        aggregation_process = tff.utils.build_dp_aggregate_process(
            weights_type.trainable, dp_query)
    else:
        aggregation_process = None

    # server_optimizer_fn = optimizer_utils.create_optimizer_fn_from_flags('server')
    # client_optimizer_fn = optimizer_utils.create_optimizer_fn_from_flags('client')
    iterative_process = tff.learning.build_federated_averaging_process(
        model_fn=model_fn,
        # server_optimizer_fn=server_optimizer_fn,# client_optimizer_fn=client_optimizer_fn,
        # client_weight_fn=client_weight_fn,
        client_optimizer_fn=lambda: tf.optimizers.Adam(learning_rate=0.01),
        server_optimizer_fn=lambda: tf.optimizers.SGD(learning_rate=0.05),
        # aggregation_process=aggregation_process
    )

    client_datasets_fn = training_utils.build_client_datasets_fn(
        train_dataset, FLAGS.clients_per_round)

    evaluate_fn = training_utils.build_evaluate_fn(
        eval_dataset=tf.data.Dataset.from_tensor_slices((dataTestX, dataTestY)),
        model_builder=keras_model,
        loss_builder=loss_builder,
        metrics_builder=metrics_builder)
    # ev_result = eval_model.evaluate(dataTestX, dataTestY, verbose=0)
    logging.info('Training model:')
    logging.info(keras_model.summary())

    hparam_dict = utils_impl.lookup_flag_values(utils_impl.get_hparam_flags())
    training_loop_dict = utils_impl.lookup_flag_values(training_loop_flags)

    training_loop.run(
        iterative_process=iterative_process,
        client_datasets_fn=client_datasets_fn,
        validation_fn=evaluate_fn,
        hparam_dict=hparam_dict,
        **training_loop_dict)


if __name__ == '__main__':
    app.run(main)
