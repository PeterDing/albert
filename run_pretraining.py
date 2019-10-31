import os

from absl import app
from absl import logging
from absl import flags

import tensorflow as tf
from tensorflow import keras
from tensorflow import distribute

from model.albert import ALBERT_CONFIG
from model.bert import Bert, PreTrainLMPredictor, PreTrainNextSentencePredictor
from model.loss import sparse_categorical_crossentropy
from model.util import create_attention_mask

from lamb import Lamb

FLAGS = flags.FLAGS
flags.DEFINE_string('strategy', 'mirror', 'Distribution strategy, `mirror` or `tpu`')
flags.DEFINE_string('tpu_addr', None, 'tpu address')
flags.DEFINE_list('train_files', None, 'Training tfrecord files')
flags.DEFINE_integer('train_batch_size', 2, 'Training data batch size')
flags.DEFINE_list('eval_files', None, 'Training tfrecord files')
flags.DEFINE_integer('eval_batch_size', 1, 'Evaluating data batch size')
flags.DEFINE_integer('eval_steps', 1, 'Steps per evaluation')
# flags.DEFINE_integer('max_seq_length', 512, 'Maximum sequence length of an example')
# flags.DEFINE_integer('max_mask_length', 20, 'Maximum sequence length of predictable mask token [MASK]')
flags.DEFINE_integer('shuffle_buffer_size', 4, 'Buffer size for shuffling')
flags.DEFINE_string('model_type', 'base', 'Albert model type')
flags.DEFINE_integer('epochs', 4, 'Epochs')
flags.DEFINE_integer('steps_per_epoch', 5, 'Steps per epoch')
flags.DEFINE_integer('steps_per_loop', 1, 'Steps per loop')
flags.DEFINE_float('learning_rate', 5e-5, 'The initial learning rate for Adam.')

flags.DEFINE_string(
    'model_dir', './model-out', 'The output directory where the model checkpoints will be written.'
)
flags.DEFINE_integer('max_checkpoint_to_keep', 5, 'How many checkpoints to keep')
flags.DEFINE_string('init_checkpoint', None, 'Initial checkpoint')


def _init(max_seq_length, max_mask_length):
    global NAME_TO_FEATURES
    NAME_TO_FEATURES = {
        'input_ids': tf.io.FixedLenFeature(shape=(max_seq_length,), dtype=tf.int64),
        'input_mask': tf.io.FixedLenFeature(shape=(max_seq_length,), dtype=tf.int64),
        'segment_ids': tf.io.FixedLenFeature(shape=(max_seq_length,), dtype=tf.int64),
        'mask_positions': tf.io.FixedLenFeature(shape=(max_mask_length,), dtype=tf.int64),
        'mask_label_ids': tf.io.FixedLenFeature(shape=(max_mask_length,), dtype=tf.int64),
        'mask_weights': tf.io.FixedLenFeature(shape=(max_mask_length,), dtype=tf.int64),
        'next_id': tf.io.FixedLenFeature(shape=(1,), dtype=tf.int64),
    }


def _float_metric_value(metric):
    return metric.result().numpy().astype(float)


def init_tpu(tpu_addr):
    """Initializes TPU for TF 2.0 training.
    Args:
      tpu_address: string, bns address of master TPU worker.
    Returns:
      A TPUClusterResolver.
    """
    cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu=tpu_addr)
    if tpu_addr not in ('', 'local'):
        tf.config.experimental_connect_to_cluster(cluster_resolver)
    tf.tpu.experimental.initialize_tpu_system(cluster_resolver)
    return cluster_resolver


def save_model(checkpoint, model_dir, path):
    checkpoint_path = os.path.join(model_dir, path)
    checkpoint.save(checkpoint_path)


def steps_to_run(current_step, steps_per_epoch, steps_per_loop):
    if steps_per_loop == 1:
        return steps_per_loop
    if current_step + steps_per_loop < steps_per_epoch:
        return steps_per_loop
    else:
        return steps_per_epoch - current_step


def dump_example(raw_example):
    example = tf.io.parse_single_example(raw_example, NAME_TO_FEATURES)
    return example


def gather_indexes(logits, mask_positions):
    # logits = [batch_size, max_seq_length, vocab_size]
    # mask_positions = [batch_size, max_mask_length]
    vocab_size = logits.shape[-1]
    masks_features = tf.gather(logits, mask_positions, batch_dims=1)
    # [batch_size * max_mask_length, vocab_size]
    return tf.reshape(masks_features, [-1, vocab_size])


def model_fn(config):
    max_seq_length = config['max_seq_length']
    input_ids = keras.Input(shape=(max_seq_length,), name='input_ids')
    attention_mask = keras.Input(shape=(max_seq_length,), name='attention_mask')
    segment_ids = keras.Input(shape=(max_seq_length,), name='segment_ids')

    albert = Bert(config)
    first_tokens, output = albert([input_ids, attention_mask, segment_ids])

    # albert_model = keras.Model(inputs=[input_ids, attention_mask, segment_ids], outputs=output)
    albert_model = keras.Model(
        inputs={
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'segment_ids': segment_ids
        },
        outputs=[first_tokens, output]
    )

    lm_predictor = PreTrainLMPredictor(
        config['input_hidden'],
        config['vocab_size'],
        max_seq_length,
    )
    lm_predict_outputs = lm_predictor([output, albert.embedding.embeddings, albert.projection])
    next_sentence_predictor = PreTrainNextSentencePredictor(2)
    next_sentence_predict_outputs = next_sentence_predictor(first_tokens)

    classifier_model = keras.Model(
        inputs={
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'segment_ids': segment_ids
        },
        outputs=[lm_predict_outputs, next_sentence_predict_outputs]
    )

    # Optimizer
    lamb_optimizer = Lamb()
    return classifier_model, albert_model, lamb_optimizer


def main(_):
    config = ALBERT_CONFIG[FLAGS.model_type]
    _init(config['max_seq_length'], config['max_mask_length'])

    # Make strategy
    assert FLAGS.strategy, 'Strategy can not be empty'
    if FLAGS.strategy == 'mirror':
        strategy = distribute.MirroredStrategy()
    elif FLAGS.strategy == 'tpu':
        cluster_resolver = init_tpu(FLAGS.tpu_addr)
        strategy = tf.distribute.experimental.TPUStrategy(cluster_resolver)
    else:
        raise ValueError('The distribution strategy type is not supported: %s' % FLAGS.strategy)

    # Prepare training dataset
    file_list = tf.data.Dataset.list_files(FLAGS.train_files)
    train_dataset = tf.data.TFRecordDataset(filenames=file_list)
    train_dataset = train_dataset.shuffle(buffer_size=FLAGS.shuffle_buffer_size)
    train_dataset = train_dataset.map(
        dump_example, num_parallel_calls=tf.data.experimental.AUTOTUNE
    ).cache()
    train_dataset = train_dataset.shuffle(100)
    train_dataset = train_dataset.repeat()  # loop
    train_dataset = train_dataset.batch(FLAGS.train_batch_size)
    train_dataset = train_dataset.prefetch(FLAGS.train_batch_size)

    # Prepare evaluation dataset
    file_list = tf.data.Dataset.list_files(FLAGS.eval_files)
    eval_dataset = tf.data.TFRecordDataset(filenames=file_list)
    eval_dataset = eval_dataset.map(
        dump_example, num_parallel_calls=tf.data.experimental.AUTOTUNE
    ).cache()
    eval_dataset = eval_dataset.batch(FLAGS.eval_batch_size)
    eval_dataset = eval_dataset.prefetch(FLAGS.eval_batch_size)

    def make_iter_dataset(dataset):
        iter_dataset = iter(strategy.experimental_distribute_dataset(train_dataset))
        return iter_dataset

    # Training
    with strategy.scope():
        # Build Albert model
        logging.info('Build albert: config: %s', config)
        classifier_model, albert_model, optimizer = model_fn(config)

        if FLAGS.init_checkpoint:
            logging.info('Restore albert_model from initial checkpoint: %s', FLAGS.init_checkpoint)
            checkpoint = tf.train.Checkpoint(albert_model=albert_model)
            checkpoint.restore(FLAGS.init_checkpoint)

        # Make metric functions
        train_loss_metric = keras.metrics.Mean('training_loss', dtype=tf.float32)
        eval_metrics = [
            keras.metrics.SparseCategoricalAccuracy('test_accuracy', dtype=tf.float32),
        ]
        train_metrics = [
            keras.metrics.SparseCategoricalAccuracy('test_accuracy', dtype=tf.float32),
        ]

        # Make summary writers
        summary_dir = os.path.join(FLAGS.model_dir, 'summaries')
        eval_summary_writer = tf.summary.create_file_writer(os.path.join(summary_dir, 'eval'))
        train_summary_writer = tf.summary.create_file_writer(os.path.join(summary_dir, 'train'))

        def step_fn(batch):
            input_ids = batch['input_ids']
            input_mask = batch['input_mask']
            segment_ids = batch['segment_ids']
            mask_positions = batch['mask_positions']
            mask_label_ids = batch['mask_label_ids']
            mask_weights = batch['mask_weights']
            next_id = batch['next_id']
            attention_mask = create_attention_mask(input_mask)

            train_inputs = {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'segment_ids': segment_ids,
            }

            with tf.GradientTape() as tape:
                train_outputs = classifier_model(train_inputs)
                lm_logits = gather_indexes(train_outputs[0], mask_positions)
                loss_lm = sparse_categorical_crossentropy(
                    mask_label_ids,
                    lm_logits,
                    weights=mask_weights,
                    from_logits=True,
                )
                next_logits = train_outputs[1]
                loss_next = sparse_categorical_crossentropy(
                    next_id,
                    next_logits,
                    from_logits=True,
                )
                loss = loss_lm + loss_next

            grads = tape.gradient(loss, classifier_model.trainable_weights)
            optimizer.apply_gradients(list(zip(grads, classifier_model.trainable_weights)))

            # metric
            train_loss_metric.update_state(loss)
            for metric in train_metrics:
                # mask_label_ids = [batch_size * max_mask_length, 1]
                mask_label_ids = tf.reshape(mask_label_ids, [-1, 1])

                metric.update_state(mask_label_ids, lm_logits)
                metric.update_state(next_id, next_logits)

            return loss

        @tf.function
        def train_steps(iterator, steps):
            for step in tf.range(steps):
                strategy.experimental_run_v2(step_fn, args=(next(iterator),))

        @tf.function
        def test_step(iterator):

            def test_step_fn(batch):
                input_ids = batch['input_ids']
                input_mask = batch['input_mask']
                segment_ids = batch['segment_ids']
                mask_positions = batch['mask_positions']
                mask_label_ids = batch['mask_label_ids']
                next_id = batch['next_id']
                attention_mask = create_attention_mask(input_mask)

                eval_inputs = {
                    'input_ids': input_ids,
                    'attention_mask': attention_mask,
                    'segment_ids': segment_ids,
                }
                eval_outputs = classifier_model(eval_inputs)
                lm_logits = gather_indexes(eval_outputs[0], mask_positions)
                next_logits = eval_outputs[1]

                for metric in eval_metrics:
                    # mask_label_ids = [batch_size * max_mask_length, 1]
                    mask_label_ids = tf.reshape(mask_label_ids, [-1, 1])

                    metric.update_state(mask_label_ids, lm_logits)
                    metric.update_state(next_id, next_logits)

            strategy.experimental_run_v2(test_step_fn, args=(next(iterator),))

        def _run_evaluation(current_step, iterator):
            for _ in range(FLAGS.eval_steps):
                test_step(iterator)

            log = f'eval step: {current_step}, '
            with eval_summary_writer.as_default():
                for metric in eval_metrics:
                    metric_value = _float_metric_value(metric)
                    tf.summary.scalar(metric.name, metric_value, step=current_step)
                    log += f'metric: {metric.name} = {metric_value}, '
                eval_summary_writer.flush()
            logging.info(log)

        # Restore classifier_model
        checkpoint = tf.train.Checkpoint(model=classifier_model, optimizer=optimizer)
        latest_checkpoint_file = tf.train.latest_checkpoint(FLAGS.model_dir)
        if latest_checkpoint_file:
            logging.info(
                'Restore classifier_model from the latest checkpoint file: %s',
                latest_checkpoint_file
            )
            checkpoint.restore(latest_checkpoint_file)

        train_iter_dataset = make_iter_dataset(train_dataset)
        total_training_steps = FLAGS.epochs * FLAGS.steps_per_epoch
        current_step = optimizer.iterations.numpy()
        while current_step < total_training_steps:
            steps = steps_to_run(current_step, FLAGS.steps_per_epoch, FLAGS.steps_per_loop)
            # Converts steps to a Tensor to avoid tf.function retracing.
            train_steps(train_iter_dataset, tf.convert_to_tensor(steps, dtype=tf.int32))
            current_step += steps
            logging.info('Step: %s', current_step)

            log = f'training step: {current_step}, '
            with train_summary_writer.as_default():
                for metric in train_metrics + [train_loss_metric]:
                    metric_value = _float_metric_value(metric)
                    tf.summary.scalar(metric.name, metric_value, step=current_step)
                    log += f'metric: {metric.name} = {metric_value}, '
                train_summary_writer.flush()

            if current_step % FLAGS.steps_per_epoch:
                checkpoint_path = os.path.join(FLAGS.model_dir, f'model_step_{current_step}.ckpt')
                checkpoint.save(checkpoint_path)
                logging.info('Save model to {checkpoint_path}')

                eval_iter_dataset = make_iter_dataset(eval_dataset)
                _run_evaluation(current_step, eval_iter_dataset)


if __name__ == '__main__':
    app.run(main)
