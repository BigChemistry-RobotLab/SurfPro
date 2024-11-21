import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem.Descriptors import ExactMolWt
import rdkit.Chem.SaltRemover as SaltRemover


def calc_Gamma_max():
    """from 10.1016/j.jcis.2014.02.017
    $ aw_st = -1/(2.303*n*R*T) * (dγ/d log C_surf,t)_T $
    units: mol/m^2
    surface pressure of the solution is dγ/dlogCsurf,t can be obtained from the isotherm
    also known as Γ_max (Gamma_max)
    """
    pass


def calc_pi_cmc(AW_ST, T):
    """gamma_0 = surface tension of pure water (γ0) equals ± 72 mN/m
    AW_ST = gamma_cmc = surface tension at cmc"""
    # gamma_0 from https://en.wikipedia.org/wiki/Water_%28data_page%29
    if round(T) == 20:
        gamma_0 = 72.75
    elif round(T) == 25:
        gamma_0 = 71.97
    elif round(T) == 30:
        gamma_0 = 71.18
    else:
        return np.nan
    pi_cmc = gamma_0 - float(AW_ST)
    assert pi_cmc > 0 and pi_cmc < 100
    return pi_cmc


def calc_aw_st(Pi_CMC, T):
    """gamma_0 = surface tension of pure water (γ0) equals ± 72 mN/m
    AW_ST = gamma_cmc = surface tension at cmc"""
    # gamma_0 from https://en.wikipedia.org/wiki/Water_%28data_page%29
    if round(T) == 20:
        gamma_0 = 72.75
    elif round(T) == 25:
        gamma_0 = 71.97
    elif round(T) == 30:
        gamma_0 = 71.18
    else:
        return np.nan
    aw_st = gamma_0 - float(Pi_CMC)
    return aw_st


def calc_area_cmc(Gamma_max):
    """from 10.1016/j.jcis.2014.02.017
    units: nm^2"""
    const = 1 * 10**18
    avogadro_const = 6.02214076 * 10**23
    area = const / (avogadro_const * float(Gamma_max))
    assert area > 0.001 and area < 10
    return area


def calc_pc20(AW_ST, Gamma_max, pCMC, T, charge):
    """$ pc20 = \frac{γ_0 - 20 - γ_cmc}{2.303*n*R*T * Gamma_max} - logCMC $
    R = gas constant
    T = temp in Celsius, T_K = temp in Kelvin
    n = number of species comprising the surfactant and adsorbed counterion
    n=2 for traditional surfactants, n=3 for dimeric, calculated from charge
    AW_ST = gamma_cmc = surface tension at cmc (! in mN/m, /1000 -> N/m)
    Gamma_max ="""
    T_K = T + 273.15

    n = np.abs(charge) + 1

    gamma_0 = 72  # γ_0
    R = 8.31446261815324

    nom = (gamma_0 - 20 - float(AW_ST)) / 1000  # convert to N/m (SI unit)
    denom = (2.303 * n * R * T_K) * float(Gamma_max)
    pc20 = nom / denom + float(pCMC)

    assert pc20 > 0 and pc20 < 10
    return pc20


def calc_langmuir(AW_ST, Gamma_max, CMC, T):
    """K_L = exp( \frac(γ_0 - γ_cmc)(RT * Gamma_max) - 1) / CMC"""
    T_K = T + 273.15
    gamma_0 = 72  # γ_0
    R = 8.31446261815324

    frac = ((gamma_0 - AW_ST) / 1000) / (R * T_K * Gamma_max)
    return (np.exp(frac) - 1) / CMC


def raw_to_neglog(val):
    return -np.log10(float(val))


def raw_to_log(val):
    return np.log10(float(val))


def neglog_to_raw(val):
    return np.exp(-float(val))


def calc_charge_from_smiles(smi):
    # TODO differentiate zwitter an non-ionic
    # str comparision against +/- or using partial charge
    mol = Chem.MolFromSmiles(smi)
    remover = SaltRemover.SaltRemover(defnData="[Cl-,Br-,Na+,I-,Li+,K+]")
    mol = remover.StripMol(mol)

    for salt in [
        "CS(=O)(=O)[O-]",
        "C[N+](C)(C)C",
        "O=[PH](=O)([O-])O",
        "F[B-](F)(F)F",
        "CCC[N+](CCC)(CCC)CCC",
        "[NH4+]",
        "Cc1ccc(S(=O)(=O)[O-])cc1",
    ]:
        remover = SaltRemover.SaltRemover(defnData=salt, defnFormat="smiles")
        mol = remover.StripMol(mol)

    formal_charge = Chem.GetFormalCharge(mol)
    return formal_charge


def map_smiles_to_class(smi):
    map = {
        "-2": "gemini anionic",
        "-1": "anionic",
        # "0": "non-ionic or zwitterionic",
        "1": "cationic",
        "2": "gemini cationic",
        "3": "gemini cationic",
        "4": "gemini cationic",
    }

    formal_charge = calc_charge_from_smiles(smi)
    if formal_charge < -1 or formal_charge > 2:
        print(smi)

    # charge 0: either "non-ionic or zwitterionic",
    if formal_charge == 0:
        mol = Chem.MolFromSmiles(smi)
        abs_charge = np.sum(np.abs(atom.GetFormalCharge()) for atom in mol.GetAtoms())
        if abs_charge == 0:
            return "non-ionic"
        else:
            return "zwitterionic"
    else:
        return map.get(str(formal_charge))
