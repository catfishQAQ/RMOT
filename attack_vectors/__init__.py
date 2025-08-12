
from .acoustic_attack import stn_blur_general, init_blur_params
from .emi_attack import apply_emi_attack, initialize_stripe_params, init_emi_params

def build_attack_vector(attack_vector):
    vectors_catalog = {
        'acoustic': stn_blur_general,
        'emi': apply_emi_attack,
    }

    params_init = {
        'acoustic': init_blur_params,
        'emi': initialize_stripe_params,
    }

    misc = {
        'acoustic': init_blur_params,
        'emi': init_emi_params,
    }

    assert attack_vector in vectors_catalog, 'invalid attack vector: {}'.format(attack_vector)
    assert attack_vector in params_init, 'invalid attack params initializer: {}'.format(attack_vector)
    return params_init[attack_vector], vectors_catalog[attack_vector], misc[attack_vector]