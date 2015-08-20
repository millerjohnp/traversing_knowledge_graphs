__author__ = 'Kelvin Gu'

freebase_experiment = {
    'description': 'Best performing FreeBase model',
    # negative sampling
    'type_matching_negs': True,
    'max_negative_samples_train': 10,
    'max_negative_samples_eval': 200,
    'positive_branch_factor': 75,

    # model related
    'path_model': 'bilinear_diag',  # bilinear, bilinear_diag
    'wvec_dim': 100,
    'l2_reg': 0.0001,
    'init_scale': 0.1,  # variance of Gaussians used for initialization

    # SGD
    'step_size_single': 0.05,
    'step_size_path': 0.05,
    'batch_size': 300,
    'hidden_dim': 20,  # only used in Neural Tensor Network
    'report_wait': 50,

    # max steps
    'max_steps_single': 60000,
    'max_steps_path': 70000,
}


wordnet_experiment = {
    'description': 'Best performing WordNet model.',
    # negative sampling
    'type_matching_negs': True,
    'max_negative_samples_train': 10,
    'max_negative_samples_eval': 200,
    'positive_branch_factor': float('inf'),

    # model related
    'path_model': 'bilinear',  # transE, bilinear_diag
    'wvec_dim': 100,
    'l2_reg': 0.0001,
    'init_scale': 0.1,  # variance of Gaussians used for initialization

    # SGD
    'step_size_single': 0.1,
    'step_size_path': 0.01,
    'batch_size': 300,
    'hidden_dim': 20,  # only used in Neural Tensor Network
    'report_wait': 50,

    # max steps
    'max_steps_single': 40000,
    'max_steps_path': 40000,
}