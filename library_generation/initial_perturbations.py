from rdkit.Chem import AllChem, MolFromSmiles, MolToSmiles, Draw, Crippen
from rdkit.Chem.rdmolops import FastFindRings
from rdkit import Chem
from tqdm import tqdm
import pandas as pd
import numpy as np
import logging
import time 

logging.basicConfig(level=logging.INFO)

def canonicalize(mol_list, showprogress=False):
    """
    Function that returns the set of unique RDKit molecules from a list of input RDKit molecules
    by turning them into canonical SMILES and checking the strings for uniqueness. 
    """
    smi_list = []
    if showprogress:
        print('Canonicalising mols')
        for mol in tqdm(mol_list):
            if mol is not None:
                smi_list.append(MolToSmiles(mol))
    else:
        for mol in mol_list:
            if mol is not None:
                smi_list.append(MolToSmiles(mol))
    mol_list = list(set(smi_list))
    if showprogress:
        mol_list = [MolFromSmiles(smi) for smi in tqdm(mol_list)]
    else:
        mol_list = [MolFromSmiles(smi) for smi in mol_list]
    return mol_list


def simple_rxn(mol_list, rxn, debug=False):
    """
    Function that applies a simple one-reactant one-product reaction SMILES to a list of input RDKit molecules,
    returning the products as a list of RDKit molecules.
    """ 
    prod_list = []
    for mol in mol_list:
        if debug:
            logging.info('Input: '+ MolToSmiles(mol))
        products = rxn.RunReactants((Chem.AddHs(mol),))
        if debug:
            logging.info('Products: {}'.format(products))
        if products != ():
            for prod in products:
                if debug:
                    logging.info(prod)
                    logging.info(MolToSmiles(prod[0]))
                prod_list.append(prod[0])
    return prod_list

def pair_prods(mol_list, rxn, debug=False):
    """
    Function that applies a one-reactant two-product reaction SMILES to a list of input RDKit molecules,
    returning the products as two separate lists of RDKit molecules.
    """ 
    prod1_list = []
    prod2_list = []
    for mol in mol_list:
        if debug:
            logging.info(MolToSmiles(mol))
        try:
            mol.UpdatePropertyCache()
            FastFindRings(mol)
        except:
            logging.info('This mol fails! ' + MolToSmiles(mol))
            continue
        products = rxn.RunReactants((Chem.AddHs(mol),))
        if products != ():
            for prod in products:
                prod1_list.append(prod[0])
                prod2_list.append(prod[1])
    return prod1_list, prod2_list

def pair_rxnts(mol1_list, mol2_list, rxn, debug=False):
    """
    Function that applies a two-reactant one-product reaction SMILES to two lists of input RDKit molecules,
    returning the products as a list of RDKit molecules.
    """ 
    prod_list = []
    for mol1 in mol1_list:
        for mol2 in mol2_list:

            products = rxn.RunReactants((Chem.AddHs(mol1),Chem.AddHs(mol2)))
            if debug:
                logging.info(products)
            if products != ():
                for prod in products:
                    if debug:
                        logging.info(MolToSmiles(prod[0]))
                    prod_list.append(prod[0])
    return prod_list

t1 = time.time()

# Read in the list of active noncovalent molecules
df = pd.read_csv('../data/noncovalent_activity.smi')
df = df[df['activity']==1]

logging.info('Number of non-covalent actives: {}'.format(len(df)))
smiles_list = df['SMILES'].values

mols = [MolFromSmiles(smi) for smi in smiles_list]

# List of reaction SMARTS for breaking along synthetically accessible bonds, performing chemical perturbations, and reconnecting along synthons
amide_to_urea = AllChem.ReactionFromSmarts('[C:1]([#1])[C:2](=[O:3])[N:4] >> [N:1][C:2](=[O:3])[N:4]')
amide_swap = AllChem.ReactionFromSmarts('[C:1]([#1])[C:2](=[O:3])[N:4] >> [N:1][C:2](=[O:3])[C:4][#1]')
urea_to_amide = AllChem.ReactionFromSmarts('[N:1][C:2](=[O:3])[N:4]>>[N:1][C:2](=[O:3])[C:4]')
amide_slice = AllChem.ReactionFromSmarts('[C:1](=[O:2])[N:3]>>[C:1](=[O:2])[O][#1].[N:3][#1]')
amine_decomp = AllChem.ReactionFromSmarts('[NR0H1:1]([#6:2])[#6:3]>>[N:1]([#6:2])[#1].[#6:3][Br]')
amine_comb = AllChem.ReactionFromSmarts('[N:1]([#6:2])([#1])[#1].[#6:3][Br]>>[N:1]([#6:2])[#6:3]')
amide_lib = AllChem.ReactionFromSmarts('[C:1](=[O:2])[O][#1].[N:3][#1]>>[C:1](=[O:2])[N:3]')
urea_lib = AllChem.ReactionFromSmarts('[N:1][#1].[N:2][#1]>>[N:1][C](=[O])[N:2]')
extra = AllChem.ReactionFromSmarts('[N:1][n,c:2].[N,O,C;!$(NC=O):3][c:4]>>[*:3][*:2]')

# Apply reaction SMARTS to our set of molecules
urea_list = simple_rxn(mols, amide_to_urea)
logging.info('ureas before canon: {}'.format(len(urea_list)))

swap_list = simple_rxn(mols, amide_swap)
logging.info('swapped amides before canon: {}'.format(len(swap_list)))

amide_list = simple_rxn(mols, urea_to_amide)
logging.info('amides before canon: {}'.format(len(amide_list)))

products_so_far = canonicalize(mols + urea_list + swap_list + amide_list)
logging.info('number of canonicalized products: {}'.format(len(products_so_far)))

acid_list, amine_list = pair_prods(products_so_far, amide_slice)
logging.info('Number of carboxylate acids: {}'.format(len(acid_list)))

acid_list = canonicalize(acid_list)
logging.info('Number of canonised carboxylate acids: {}'.format(len(acid_list)))

amine2_list, subs_list = pair_prods(amine_list, amine_decomp)
subs_list = canonicalize(subs_list)
amine_list = canonicalize(amine_list + amine2_list)

# Save the list of all possible primary amines for use later
with open('../data/amine_list.txt', 'w') as filehandle:
     filehandle.writelines("%s\n" % MolToSmiles(mol) for mol in amine_list)

logging.info('Number of primary amines: {}'.format(len(amine_list)))
logging.info('Number of substituents: {}'.format(len(subs_list)))

second_amine_lib = canonicalize(pair_rxnts(amine_list, subs_list, amine_comb))
logging.info('Number of canonised secondary amines: {}'.format(len(second_amine_lib)))

amide_lib_list = canonicalize(pair_rxnts(acid_list, amine_list, amide_lib))
logging.info('Number of canonised primary amides: {}'.format(len(amide_lib_list)))

second_amide_lib_list = canonicalize(pair_rxnts(acid_list, second_amine_lib, amide_lib))
logging.info('Number of secondary amides: {}'.format(len(second_amide_lib_list)))

amide_lib_list = canonicalize(amide_lib_list + second_amide_lib_list)
urea_lib_list = canonicalize(pair_rxnts(amine_list, second_amine_lib, urea_lib))

logging.info('Number of ureas: {}'.format(len(urea_lib_list)))
penultimate_lib = amide_lib_list + urea_lib_list + mols
penultimate_lib = canonicalize(penultimate_lib)
penultimate_lib = [MolToSmiles(mol) for mol in penultimate_lib]

# Savet the list of all amides, ureas, and the active molecules for use later 
with open('../data/penul_lib.txt', 'w') as filehandle:
     filehandle.writelines("%s\n" % mol for mol in penultimate_lib)

t2 = time.time()
print('Time taken: {:.1f}mins'.format((t2-t1)/60))
