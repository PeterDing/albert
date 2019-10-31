from absl import app
from absl import flags

from utils.data import make_tfrecord
from utils.tokenization import FullTokenizer

FLAGS = flags.FLAGS
flags.DEFINE_list('file_patterns', None, "Document's files patterns.")
flags.DEFINE_integer('max_seq_length', 512, 'Maximum sequential length.')
flags.DEFINE_float('mask_prob', 0.1, 'The probability of mask LM number per sequence.')
flags.DEFINE_integer(
    'max_mask_length', 20, 'Maximum number of masked LM predictions per sequence.'
)
flags.DEFINE_bool(
    'random_seq', False,
    'Original Bert use random next sequence to predict. For Albert, we set this as False'
)
flags.DEFINE_bool(
    'whole_word_mask', False, 'Maximum number of masked LM predictions per sequence.'
)
flags.DEFINE_string(
    'tfrecord_file_prefix', 'data-training/albert-training-data',
    'Prefix used as saved tfrecord files.'
)
flags.DEFINE_integer('tfrecord_file_num', 10, 'How many tfrecord files to store.')
flags.DEFINE_bool('do_lower_case', True, '')
flags.DEFINE_string('vocab_file', None, '')


def main(_):
    tokenizer = FullTokenizer(
        vocab_file=FLAGS.vocab_file,
        do_lower_case=FLAGS.do_lower_case,
    )

    file_patterns = FLAGS.file_patterns
    max_seq_length = FLAGS.max_seq_length
    mask_prob = FLAGS.mask_prob
    max_mask_length = FLAGS.max_mask_length
    random_seq = FLAGS.random_seq
    whole_word_mask = FLAGS.whole_word_mask
    tfrecord_file_prefix = FLAGS.tfrecord_file_prefix
    tfrecord_file_num = FLAGS.tfrecord_file_num

    make_tfrecord(
        tokenizer,
        file_patterns,
        max_seq_length,
        mask_prob,
        max_mask_length,
        tfrecord_file_prefix=tfrecord_file_prefix,
        tfrecord_file_num=tfrecord_file_num,
        random_seq=random_seq,
        whole_word_mask=whole_word_mask,
    )


if __name__ == '__main__':
    app.run(main)
