
from .physical.pgd_shrink import PhyPGDAttackerShrink
from .physical.pgd_ff import PhyPGDAttackerFF
from .physical.pgd_daedalus import PhyPGDAttackerDaedalus   
from .physical.pgd_ours import PhyPGDAttackerOurs

from .digital.pgd_shrink import PGDAttackerShrink
from .digital.pgd_ff import PGDAttackerFF
from .digital.pgd_daedalus import PGDAttackerDaedalus
from .digital.pgd_ours import PGDAttackerOurs

from .attack_utils import Attack_Scheduler

def build_attacker(attack_type):
    attacker_catalog = {
        'phy_ours_1': PhyPGDAttackerOurs,
        'phy_ours_2': PhyPGDAttackerOurs,
        'phy_ours_3': PhyPGDAttackerOurs,
        'phy_ff': PhyPGDAttackerFF,
        'phy_daedalus': PhyPGDAttackerDaedalus,
        'phy_shrink': PhyPGDAttackerShrink,
        'ours_1': PGDAttackerOurs,
        'ours_2': PGDAttackerOurs,
        'ours_3': PGDAttackerOurs,
        'ff': PGDAttackerFF,
        'daedalus': PGDAttackerDaedalus,
        'shrink': PGDAttackerShrink,
    }
    assert attack_type in attacker_catalog, 'invalid attack: {}'.format(attack_type)
    return attacker_catalog[attack_type]


