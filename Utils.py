import numpy as np


__all__ = ['bitmask_encode', 'bitmask_decode']


def bitmask_encode(bits):
    if isinstance(bits, int):
        return 2**bits
    elif isinstance(bits, np.ndarray):
        return (2**bits).sum()
    elif isinstance(bits, list):
        return sum([2**ii for ii in bits])
    else:
        raise ValueError('Input bits must be of type: \
                         int, list, or np.ndarray')


def bitmask_decode(target_bit, maxbit=64):
    bits_found = []
    for bit in range(maxbit):
        if (target_bit & np.int64(2)**bit) != 0:
            bits_found.append(bit)

    return bits_found

def get_ebosstarget_class(target_bit0, target_bit1, target_bit2):
    # use eboss_target2 flag to get subclasses
    target_classes = []
    if (target_bit0 & 2**10) != 0: target_classes.append('SEQUELS_QSO_EBOSS_CORE')
    if (target_bit0 & 2**11) != 0: target_classes.append('SEQUELS_QSO_PTF')
    if (target_bit0 & 2**12) != 0: target_classes.append('SEQUELS_QSO_REOBS')
    if (target_bit0 & 2**13) != 0: target_classes.append('SEQUELS_QSO_EBOSS_KDE')
    if (target_bit0 & 2**14) != 0: target_classes.append('SEQUELS_QSO_EBOSS_FIRST')
    if (target_bit0 & 2**15) != 0: target_classes.append('SEQUELS_QSO_BAD_BOSS')
    if (target_bit0 & 2**16) != 0: target_classes.append('SEQUELS_QSO_QSO_BOSS_TARGET')
    if (target_bit0 & 2**17) != 0: target_classes.append('SEQUELS_QSO_SDSS_TARGET')
    if (target_bit0 & 2**18) != 0: target_classes.append('SEQUELS_QSO_KNOWN')
    if (target_bit0 & 2**19) != 0: target_classes.append('SEQUELS_DR9_CALIB_TARGET')
    if (target_bit0 & 2**20) != 0: target_classes.append('SEQUELS_SPIDERS_RASS_AGN')
    if (target_bit0 & 2**21) != 0: target_classes.append('SEQUELS_SPIDERS_RASS_CLUS')
    if (target_bit0 & 2**22) != 0: target_classes.append('SEQUELS_SPIDERS_ERASS_AGN')
    if (target_bit0 & 2**23) != 0: target_classes.append('SEQUELS_SPIDERS_ERASS_CLUS')
    if (target_bit0 & 2**30) != 0: target_classes.append('SEQUELS_TDSS_A')
    if (target_bit0 & 2**31) != 0: target_classes.append('SEQUELS_TDSS_FES_DE')
    if (target_bit0 & 2**32) != 0: target_classes.append('SEQUELS_TDSS_FES_DWARFC')
    if (target_bit0 & 2**33) != 0: target_classes.append('SEQUELS_TDSS_FES_NQHISN')
    if (target_bit0 & 2**34) != 0: target_classes.append('SEQUELS_TDSS_FES_MGII')
    if (target_bit0 & 2**35) != 0: target_classes.append('SEQUELS_TDSS_FES_VARBAL')
    if (target_bit0 & 2**40) != 0: target_classes.append('SEQUELS_PTF_VARIABLE')

    if (target_bit1 & 2**9) != 0: target_classes.append('eBOSS_QSO1_VAR_S82')
    if (target_bit1 & 2**10) != 0: target_classes.append('eBOSS_QSO1_EBOSS_CORE')
    if (target_bit1 & 2**11) != 0: target_classes.append('eBOSS_QSO1_PTF')
    if (target_bit1 & 2**12) != 0: target_classes.append('eBOSS_QSO1_REOBS')
    if (target_bit1 & 2**13) != 0: target_classes.append('eBOSS_QSO1_EBOSS_KDE')
    if (target_bit1 & 2**14) != 0: target_classes.append('eBOSS_QSO1_EBOSS_FIRST')
    if (target_bit1 & 2**15) != 0: target_classes.append('eBOSS_QSO1_BAD_BOSS')
    if (target_bit1 & 2**16) != 0: target_classes.append('eBOSS_QSO_BOSS_TARGET')
    if (target_bit1 & 2**17) != 0: target_classes.append('eBOSS_QSO_SDSS_TARGET')
    if (target_bit1 & 2**18) != 0: target_classes.append('eBOSS_QSO_KNOWN')
    if (target_bit1 & 2**30) != 0: target_classes.append('TDSS_TARGET')
    if (target_bit1 & 2**31) != 0: target_classes.append('SPIDERS_TARGET')

    if (target_bit2 & 2**0) != 0: target_classes.append('SPIDERS_RASS_AGN')
    if (target_bit2 & 2**1) != 0: target_classes.append('SPIDERS_RASS_CLUS')
    if (target_bit2 & 2**2) != 0: target_classes.append('SPIDERS_ERASS_AGN')
    if (target_bit2 & 2**3) != 0: target_classes.append('SPIDERS_ERASS_CLUS')
    if (target_bit2 & 2**4) != 0: target_classes.append('SPIDERS_XMMSL_AGN')
    if (target_bit2 & 2**5) != 0: target_classes.append('SPIDERS_XCLASS_CLUS')
    if (target_bit2 & 2**5) != 0: target_classes.append('SPIDERS_XCLASS_CLUS')
    if (target_bit2 & 2**20) != 0: target_classes.append('TDSS_A')
    if (target_bit2 & 2**21) != 0: target_classes.append('TDSS_FES_DE')
    if (target_bit2 & 2**22) != 0: target_classes.append('TDSS_FES_DWARFC')
    if (target_bit2 & 2**23) != 0: target_classes.append('TDSS_FES_NQHISN')
    if (target_bit2 & 2**24) != 0: target_classes.append('TDSS_FES_MGII')
    if (target_bit2 & 2**25) != 0: target_classes.append('TDSS_FES_VARBAL')
    if (target_bit2 & 2**26) != 0: target_classes.append('TDSS_B')
    if (target_bit2 & 2**27) != 0: target_classes.append('TDSS_FES_HYPQSO')
    if (target_bit2 & 2**28) != 0: target_classes.append('TDSS_FES_HYPSTAR')
    if (target_bit2 & 2**29) != 0: target_classes.append('TDSS_FES_WDDM')
    if (target_bit2 & 2**30) != 0: target_classes.append('TDSS_FES_ACTSTAR')
    if (target_bit2 & 2**31) != 0: target_classes.append('TDSS_COREPTF')

    return target_classes


if (__name__ == '__main__'):
    print('Utils.py is not meant to be run. Please import only.')
