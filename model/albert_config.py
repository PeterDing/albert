ALBERT_CONFIG = {
    'base':
        {
            'vocab_size': 30_000,
            'embedding_size': 128,
            'factorize_embedding': True,
            'input_hidden': 768,
            'segment_size': 2,
            'max_seq_length': 512,
            'max_mask_length': 20,
            'dropout': 0.0,
            'layer_num': 12,
            'layer_norm_output': False,
            'init_stddev': 0.02,
            'share_all': True,
            'attention_config':
                {
                    'head_size': 12,
                    'per_hidden_size': 64,
                    'dropout': 0.0,
                    'init_stddev': 0.02,
                    'return_2d': True,
                    'shared': True,
                    'shared_output': True,
                },
            'feedforward_config':
                {
                    'hidden_size': 768,
                    'intermediate_hidden_size': 3072,
                    'dropout': 0.0,
                    'init_stddev': 0.02,
                    'shared_intermediate': True,
                    'shared_output': True,
                },
        },
    'large':
        {
            'vocab_size': 30_000,
            'embedding_size': 128,
            'factorize_embedding': True,
            'input_hidden': 1024,
            'segment_size': 2,
            'max_seq_length': 512,
            'max_mask_length': 20,
            'dropout': 0.0,
            'layer_num': 24,
            'layer_norm_output': False,
            'init_stddev': 0.02,
            'share_all': True,
            'attention_config':
                {
                    'head_size': 16,
                    'per_hidden_size': 64,
                    'dropout': 0.0,
                    'init_stddev': 0.02,
                    'return_2d': True,
                    'shared': True,
                    'shared_output': True,
                },
            'feedforward_config':
                {
                    'hidden_size': 1024,
                    'intermediate_hidden_size': 4096,
                    'dropout': 0.0,
                    'init_stddev': 0.02,
                    'shared_intermediate': True,
                    'shared_output': True,
                },
        },
    'xlarge':
        {
            'vocab_size': 30_000,
            'embedding_size': 128,
            'factorize_embedding': True,
            'input_hidden': 2048,
            'segment_size': 2,
            'max_seq_length': 512,
            'max_mask_length': 20,
            'dropout': 0.0,
            'layer_num': 24,
            'layer_norm_output': False,
            'init_stddev': 0.02,
            'share_all': True,
            'attention_config':
                {
                    'head_size': 32,
                    'per_hidden_size': 64,
                    'dropout': 0.0,
                    'init_stddev': 0.02,
                    'return_2d': True,
                    'shared': True,
                    'shared_output': True,
                },
            'feedforward_config':
                {
                    'hidden_size': 2048,
                    'intermediate_hidden_size': 8192,
                    'dropout': 0.0,
                    'init_stddev': 0.02,
                    'shared_intermediate': True,
                    'shared_output': True,
                },
        },
    'xxlarge':
        {
            'vocab_size': 30_000,
            'embedding_size': 128,
            'factorize_embedding': True,
            'input_hidden': 4092,
            'segment_size': 2,
            'max_seq_length': 512,
            'max_mask_length': 20,
            'dropout': 0.0,
            'layer_num': 12,
            'layer_norm_output': False,
            'init_stddev': 0.02,
            'share_all': True,
            'attention_config':
                {
                    'head_size': 64,
                    'per_hidden_size': 64,
                    'dropout': 0.0,
                    'init_stddev': 0.02,
                    'return_2d': True,
                    'shared': True,
                    'shared_output': True,
                },
            'feedforward_config':
                {
                    'hidden_size': 4092,
                    'intermediate_hidden_size': 16384,
                    'dropout': 0.0,
                    'init_stddev': 0.02,
                    'shared_intermediate': True,
                    'shared_output': True,
                },
        },
}