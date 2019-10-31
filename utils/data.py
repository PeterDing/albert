import tensorflow as tf
import random
import time

random = random.SystemRandom(time.time())


def int_feature(values):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))


def float_feature(values):
    return tf.train.Feature(float_list=tf.train.FloatList(value=list(values)))


def make_masks(tokens, mask_prob, max_mask_length, vocab_tokens, whole_word_mask=False):
    output_tokens = list(tokens)

    token_indexes = [i for i, token in enumerate(tokens) if token not in ['[CLS]', '[SEP]']]
    mask_num = min(max_mask_length, max(1, round(len(token_indexes) * mask_prob)))

    cand_indexes_chunks = []
    for index in token_indexes:
        if tokens[index].startswith('#'):
            if whole_word_mask:
                cand_indexes_chunks[-1].append(index)
            else:
                cand_indexes_chunks.append([index])
        else:
            cand_indexes_chunks.append([index])

    # Randomly shuffle cand_indexes_chunks, let all tokens have same probability to be chosed
    random.shuffle(cand_indexes_chunks)

    mask_indexes = []
    for indexes in cand_indexes_chunks:
        if len(mask_indexes) >= mask_num:
            break

        # Ignore too large tokens of one word
        if len(mask_indexes) + len(indexes) > mask_num:
            continue

        for index in indexes:
            # Use mask [MASK] at probability 80%
            if random.random() < 0.8:
                masked_token = '[MASK]'
            else:
                # 10% to keep original tokens
                if random.random() < 0.5:
                    masked_token = tokens[index]
                # 10% to choose a random token
                else:
                    masked_token = random.choice(vocab_tokens)

            output_tokens[index] = masked_token
            mask_indexes.append(index)

    mask_indexes.sort()
    mask_labels = [tokens[i] for i in mask_indexes]

    return output_tokens, mask_indexes, mask_labels


def split_segments(segments):
    pin = random.randint(1, len(segments) - 1)
    a_tokens = [t for tokens in segments[:pin] for t in tokens]
    b_tokens = [t for tokens in segments[pin:] for t in tokens]
    return a_tokens, b_tokens


def truncate_tokens(a_tokens, b_tokens, target_seq_num):
    while True:
        if len(a_tokens) + len(b_tokens) <= target_seq_num:
            return

        if len(a_tokens) < len(b_tokens):
            b_tokens.pop()
        else:
            del a_tokens[0]


def make_example(
    tokenizer, tokens, segment_ids, is_next, mask_positions, mask_labels, max_seq_length,
    max_mask_length
):
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1] * len(tokens)
    segment_ids = list(segment_ids)

    while len(tokens) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)

    assert len(tokens) == len(input_mask) == len(segment_ids)

    mask_positions = list(mask_positions)
    mask_label_ids = tokenizer.convert_tokens_to_ids(mask_labels)
    mask_weights = [1] * len(mask_positions)
    while len(mask_positions) < max_mask_length:
        mask_positions.append(0)
        mask_label_ids.append(0)
        mask_weights.append(0)

    assert len(mask_positions) == len(mask_label_ids) == len(mask_weights)

    next_id = 0 if is_next else 1

    features = {
        'input_ids': int_feature(input_ids),
        'input_mask': int_feature(input_mask),
        'segment_ids': int_feature(segment_ids),
        'mask_positions': int_feature(mask_positions),
        'mask_label_ids': int_feature(mask_label_ids),
        'mask_weights': int_feature(mask_weights),
        'next_id': int_feature([next_id]),
    }

    print(''.join(tokenizer.convert_ids_to_tokens(input_ids)))
    print(mask_positions, len(mask_positions))
    print(mask_weights, len(mask_weights))
    print()

    return tf.train.Example(features=tf.train.Features(feature=features))


def make_albert_examples(
    tokenizer, document, max_seq_length, mask_prob, max_mask_length, whole_word_mask=False
):
    vocab_tokens = list(tokenizer.vocab)
    target_seq_num = max_seq_length - 3  # [CLS] [SEP] [SEP]
    segments = []  # [[*paragraph1's tokens], [*paragraph1's tokens], ...]
    for i, sentence in enumerate(document):
        tokens = tokenizer.tokenize(sentence)
        if not tokens:
            continue

        if i == len(document) or sum(len(tks) for tks in segments) + len(tokens) >= target_seq_num:
            # Ignore the sentence which has more then target_seq_num tokens
            if len(tokens) < target_seq_num:
                segments.append(tokens)

            # There must have A, B tokens
            if len(segments) < 2:
                continue

            a_tokens, b_tokens = split_segments(segments)

            # There are 0.5 probability to swap a_tokens and b_tokens
            is_next = True
            if random.random() < 0.5:
                a_tokens, b_tokens = b_tokens, a_tokens
                is_next = False

            # If the total length of a_tokens and b_tokens is large,
            # we truncate them
            truncate_tokens(a_tokens, b_tokens, target_seq_num)

            # [CLS] [a1] [a2] ... [SEP] [b1] [b2] ... [SEP]
            #   0    0    0    0    0    1    1    1    1
            segment_ids = [0, *[0] * len(a_tokens), 0, *[1] * len(b_tokens), 1]
            tokens = ['[CLS]', *a_tokens, '[SEP]', *b_tokens, '[SEP]']

            # Mask LM task
            tokens, mask_positions, mask_labels = make_masks(
                tokens, mask_prob, max_mask_length, vocab_tokens, whole_word_mask=whole_word_mask
            )
            yield make_example(
                tokenizer, tokens, segment_ids, is_next, mask_positions, mask_labels,
                max_seq_length, max_mask_length
            )

            segments = []
        else:
            segments.append(tokens)


def make_examples(
    tokenizer,
    file_paths,
    max_seq_length,
    mask_prob,
    max_mask_length,
    random_seq=False,
    whole_word_mask=False
):
    """file_paths are files where documents are recorded as following:

    ```
    [doc1's real sematic sentence 1]\n
    [doc1's real sematic sentence 2]\n
    [doc1's real sematic sentence ...]\n
    \n
    [doc2's real sematic sentence 1]\n
    [doc2's real sematic sentence 2]\n
    [doc2's real sematic sentence ...]\n
    \n
    ...
    ```

    A document is a natural paragraph.
    """

    for file_path in file_paths:
        documents = []
        document = []

        if random_seq:
            documents.append(document)

        for line in open(file_path):
            if line.strip():
                document.append(line)
            else:
                if not document:
                    continue

                # if random_seq:
                #     for example in make_bert_examples(documents, max_seq_length):
                #         yield example
                #     document = []
                #     documents.append(document)
                # else:
                for example in make_albert_examples(
                    tokenizer,
                    document,
                    max_seq_length,
                    mask_prob,
                    max_mask_length,
                    whole_word_mask=whole_word_mask
                ):
                    yield example


def make_tfrecord(
    tokenizer,
    file_patterns,
    max_seq_length,
    mask_prob,
    max_mask_length,
    tfrecord_file_prefix='bert-data',
    tfrecord_file_num=6,
    random_seq=False,
    whole_word_mask=False
):
    file_paths = [path for pat in file_patterns for path in tf.io.gfile.glob(pat)]

    zpad = len(str(tfrecord_file_num))
    tfrecords = [
        tf.io.TFRecordWriter(tfrecord_file_prefix + '.{i}.tfrecord'.format(i=str(i).zfill(zpad)))
        for i in range(1, tfrecord_file_num + 1)
    ]

    for i, example in enumerate(
        make_examples(
            tokenizer,
            file_paths,
            max_seq_length,
            mask_prob,
            max_mask_length,
            random_seq=random_seq,
            whole_word_mask=False
        )
    ):
        tfrecord = tfrecords[i % tfrecord_file_num]
        tfrecord.write(example.SerializeToString())

    for tfrecord in tfrecords:
        tfrecord.close()
