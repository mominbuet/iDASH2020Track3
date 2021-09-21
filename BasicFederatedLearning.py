import collections
from typing import Callable, List, Optional

import numpy as np
import tensorflow as tf
import tensorflow_federated as tff
from absl import flags, app
from sklearn.decomposition import PCA

from NaiveBayesClassifier import bucketizeData
from Utils import getSplitData
from research.utils import training_utils
from research.utils import utils_impl

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
NUMCLIENTS = 2
tf.get_logger().setLevel('INFO')


def getClientData(dataTestTrainX, dataTestTrainY):
    data_per_set = int(np.floor(len(dataTestTrainX) / NUMCLIENTS))
    client_train_dataset = collections.OrderedDict()
    for i in range(1, NUMCLIENTS + 1):
        client_name = "client_" + str(i)
        start = data_per_set * (i - 1)
        end = data_per_set * i

        print(f"Adding data from {start} to {end} for client : {client_name}")
        data = collections.OrderedDict((('label', dataTestTrainY[start:end]), ('gene', dataTestTrainX[start:end])))
        client_train_dataset[client_name] = data
    return client_train_dataset


NUM_EPOCHS = 100
BATCH_SIZE = 20
SHUFFLE_BUFFER = 100
PREFETCH_BUFFER = 10
# NUM_CLIENTS = 10

flags.DEFINE_integer('clients_per_round', 2,
                     'How many clients to sample per round.')
flags.DEFINE_boolean(
    'uniform_weighting', True,
    'Whether to weigh clients uniformly. If false, clients '
    'are weighted by the number of samples.')
# Differential privacy flags

flags.DEFINE_float('clip', 0.05, 'Initial clip.')
flags.DEFINE_float('noise_multiplier', None,
                   'Noise multiplier. If None, no DP is used.')
flags.DEFINE_float('adaptive_clip_learning_rate', .1,
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
flags.DEFINE_float('client_lr', 0.01, 'client learning rate')
flags.DEFINE_float('server_lr', 0.01, 'server learning rate')

with utils_impl.record_new_flags() as training_loop_flags:
    flags.DEFINE_integer('total_rounds', 100, 'Number of total training rounds.')
    flags.DEFINE_string(
        'experiment_name', 'test1', 'The name of this experiment. Will be append to '
                                    '--root_output_dir to separate experiment results.')
    flags.DEFINE_string('root_output_dir', 'log/differential_privacy/',
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


def make_federated_data(client_data, client_ids, dimension):
    return [
        preprocess(client_data.create_tf_dataset_for_client(x), dimension)
        for x in client_ids
    ]


def build_client_datasets_fn(
        train_dataset: tff.simulation.ClientData,
        train_clients_per_round: int,
        random_seed: Optional[int] = None
) -> Callable[[int], List[tf.data.Dataset]]:
    sample_clients_fn = training_utils.build_sample_fn(
        train_dataset.client_ids,
        size=train_clients_per_round,
        replace=False,
        random_seed=random_seed)

    def client_datasets(round_num):
        sampled_clients = sample_clients_fn(round_num)
        return [
            preprocess(train_dataset.create_tf_dataset_for_client(client))
            for client in sampled_clients
        ]

    return client_datasets


# def create_keras_model():
#     return tf.keras.models.Sequential([
#         tf.keras.layers.Input(shape=(DIMENSION,)),
#         # tf.keras.layers.Flatten(),
#         tf.keras.layers.Dense(256, activation='relu'),
#         tf.keras.layers.Softmax(),
#     ])


def create_keras_model(dimension, act='relu'):
    x1 = tf.keras.layers.Input(shape=(dimension,))
    v1 = tf.keras.layers.Reshape((-1, 1))(x1)
    # v1 = tf.keras.layers.Flatten()(x1)

    for i in (8, 16, 32):
        v1 = tf.keras.layers.Conv1D(i, kernel_size=9, padding='same', strides=2)(v1)
        v1 = tf.keras.layers.BatchNormalization()(v1)
        v1 = tf.keras.layers.Activation(act)(v1)
        v1 = tf.keras.layers.MaxPooling1D(pool_size=4,strides=2, padding='valid')(v1)
        # v1 = tf.keras.layers.Dropout(i / (8*100))(v1)

    v1 = tf.keras.layers.Dense(128, activation='relu')(v1)
    # v1 = tf.keras.layers.Conv1D(1, kernel_size=1, padding='same')(v1)
    v1 = tf.keras.layers.Dropout(0.5)(v1)
    v1 = tf.keras.layers.Dense(2, activation='sigmoid')(v1)
    v1 = tf.keras.layers.GlobalAveragePooling1D()(v1)

    model = tf.keras.models.Model(inputs=x1, outputs=v1)

    return model


def preprocess(dataset, dimension):
    def batch_format_fn(element):
        """Flatten a batch `pixels` and return the features as an `OrderedDict`."""
        return collections.OrderedDict(
            x=tf.reshape(element['gene'], [-1, dimension]),
            y=tf.reshape(element['label'], [-1, 1]))

    return dataset.repeat(NUM_EPOCHS).shuffle(SHUFFLE_BUFFER).batch(
        BATCH_SIZE).map(batch_format_fn).prefetch(PREFETCH_BUFFER)


def main(argv):
    DATASETNAME = 'GSE25066'
    dataTrainX, dataTrainY, dataTestX, dataTestY = getSplitData(DATASETNAME)
    dataTestY = dataTestY.astype(np.int)
    dataTrainY = dataTrainY.astype(np.int)


    ##reduce dimension
    pca = PCA()
    # normalize
    dataTrainX = dataTrainX / np.linalg.norm(dataTrainX)
    dataTestX = dataTestX / np.linalg.norm(dataTestX)
    # dataTrainX, dataTestX = dataX[:len(dataTrainY)], dataX[len(dataTrainY):]

    dataTrainX = pca.fit_transform(dataTrainX)
    dataTestX = pca.transform(dataTestX)
    # if CustomExponentialHistogram or ExponentialHistogram or CustomHistogram:
    dataTrainX = bucketizeData(dataTrainX, numbins=10, EPSILON=3)
    dataTestX = bucketizeData(dataTestX, numbins=10, EPSILON=0)

    dimension = len(dataTrainX[0])

    # dataX, dataY = getDPData(dataset, 3)
    # # dataX = np.reshape (dataX, (len(dataX),len(dataX[0]),1))
    # dataTrainX, dataTestX, dataTrainY, dataTestY = train_test_split(dataX, dataY, train_size=int(len(dataX) * 0.8))

    dataTrainY = dataTrainY.astype(np.int32)
    dataTestY = dataTestY.astype(np.int32)

    train_dataset = tff.simulation.FromTensorSlicesClientData(getClientData(dataTrainX, dataTrainY))
    # test_dataset = tf.data.Dataset.from_tensor_slices((dataTestX, dataTestY))

    sample_dataset = train_dataset.create_tf_dataset_for_client(train_dataset.client_ids[0])
    # sample_element = next(iter(sample_dataset))

    preprocessed_example_dataset = preprocess(sample_dataset, dimension)

    # sample_batch = tf.nest.map_structure(lambda x: x.numpy(), next(iter(preprocessed_example_dataset)))

    federated_train_data = make_federated_data(train_dataset, train_dataset.client_ids, dimension)

    print('Number of client datasets: {l}'.format(l=len(federated_train_data)))
    print('First dataset: {d}'.format(d=federated_train_data[0]))

    if FLAGS.uniform_weighting:

        def client_weight_fn(local_outputs):
            del local_outputs
            return 1.0

    else:
        client_weight_fn = None

    def model_fn():
        # We _must_ create a new model here, and _not_ capture it from an external
        # scope. TFF will call this within different graph contexts.
        keras_model = create_keras_model(dimension)
        # print(keras_model.summary())
        return tff.learning.from_keras_model(
            keras_model,
            input_spec=preprocessed_example_dataset.element_spec,
            loss=tf.keras.losses.BinaryCrossentropy(),
            metrics=[tf.keras.metrics.BinaryAccuracy()])

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
            model=model_fn)

        weights_type = tff.learning.framework.weights_type_from_model(model_fn)
        aggregation_process = tff.utils.build_dp_aggregate_process(
            weights_type.trainable, dp_query)
    else:
        aggregation_process = None

    iterative_process = tff.learning.build_federated_averaging_process(
        model_fn,
        client_optimizer_fn=lambda: tf.keras.optimizers.Adam(learning_rate=FLAGS.client_lr),
        server_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=FLAGS.server_lr),
        client_weight_fn=client_weight_fn,
        aggregation_process=aggregation_process)
    state = iterative_process.initialize()

    # client_datasets_fn = build_client_datasets_fn(train_dataset, FLAGS.clients_per_round)
    #
    # evaluate_fn = training_utils.build_evaluate_fn(
    #     eval_dataset=test_dataset,
    #     model_builder=create_keras_model,
    #     loss_builder=tf.keras.losses.BinaryCrossentropy(),
    #     metrics_builder=tf.keras.metrics.BinaryAccuracy())
    #
    # hparam_dict = utils_impl.lookup_flag_values(utils_impl.get_hparam_flags())
    # training_loop_dict = utils_impl.lookup_flag_values(training_loop_flags)
    #
    # training_loop.run(
    #     iterative_process=iterative_process,
    #     client_datasets_fn=client_datasets_fn,
    #     validation_fn=evaluate_fn,
    #     hparam_dict=hparam_dict,
    #     **training_loop_dict)

    eval_model = create_keras_model(dimension)
    print(eval_model.summary())
    eval_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=FLAGS.client_lr),
                       loss=tf.losses.BinaryCrossentropy(),
                       metrics=[tf.keras.metrics.BinaryAccuracy()])

    for round_num in range(1, NUM_EPOCHS):
        state, learning_metrics = iterative_process.next(state, federated_train_data)
        if round_num % 10 == 0:  # and round_num > 0:
            state.model.assign_weights_to(eval_model)

            loss, accuracy = eval_model.evaluate(dataTestX, dataTestY, verbose=0)
            print(
                'round {r:2d}\tEval: loss={l:.3f}, accuracy={a:.3f} \t Train: loss={lt:.3f}, accuracy={at:.3f} '.format(
                    l=loss, a=accuracy, r=round_num, lt=learning_metrics['train']['loss'],
                    at=learning_metrics['train']['binary_accuracy']))
        print('round {:2d}, metrics={}'.format(round_num, learning_metrics))


if __name__ == '__main__':
    app.run(main)
