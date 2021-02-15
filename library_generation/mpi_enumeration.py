from rdkit.Chem import AllChem, MolFromSmiles, MolToSmiles, Draw, Crippen
from rdkit.Chem.rdmolops import FastFindRings
from rdkit import Chem
from tqdm import tqdm
import pandas as pd
import numpy as np
from mpi4py import MPI

import logging
import sys

mpi_comm = MPI.COMM_WORLD
mpi_rank = mpi_comm.Get_rank()
mpi_size = mpi_comm.Get_size()

if mpi_rank==0:
    print("\nSlicing amide library on " + str(mpi_size) + " MPI processes.\n")


logging.basicConfig(level=logging.INFO)
def canonicalize_and_filter(smi_list, showprogress=False):
    """
    Function that returns the set of unique RDKit molecules from a list of input RDKit molecules
    by turning them into canonical SMILES and checking the strings for uniqueness.

    Also performs rudimentary Lipinski rule-of-5 filtering by dropping molecules with logP >5 and 
    more than 17 heavy atoms.
    """
    mol_list = []
    if showprogress:
        print('Canonicalising mols')
        for smi in tqdm(smi_list):
            mol = MolFromSmiles(smi)
            if mol is not None:
                mol_list.append(MolToSmiles(mol))
    else:
        for smi in smi_list:
            mol = MolFromSmiles(smi)
            if mol is not None:
                mol_list.append(mol)
    mol_list = list(set(mol_list))
    final_list = []
    if showprogress:
        print('Size of unfiltered final library: {}'.format(len(mol_list)))
        print('Filtering by n_heavy and logP:')
        for smi in tqdm(mol_list):
            mol = MolFromSmiles(smi)
            n_heavy = mol.GetNumHeavyAtoms()
            if n_heavy > 17:
                logP = Crippen.MolLogP(mol)
                if logP <= 5:
                    final_list.append(smi)
    else:
        for smi in mol_list:
            mol = MolFromSmiles(smi)
            n_heavy = mol.GetNumHeavyAtoms()
            if n_heavy > 17:
                logP = Crippen.MolLogP(mol)
                if logP <= 5:
                    final_list.append(smi)
    return final_list

def pair_rxnts(mol1_list, mol2_list, rxn, debug=False):
    """
    Function that applies a two-reactant one-product reaction SMILES to a list of input RDKit molecules,
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
                    prod_list.append(MolToSmiles(prod[0]))
    return prod_list

def return_borders(n, dat_len, n_size):
    """
    Function that returns the indices allocated to the n-th MPI process when splitting a list of size
    dat_len between n_size MPI processes.
    """
    borders = np.linspace(0, dat_len, n_size + 1).astype('int')

    border_low = borders[n]
    border_high = borders[n+1]
    return border_low, border_high

# swapping and recombination of substituents for enumeration of chemical library
extra = AllChem.ReactionFromSmarts('[N:1][n,c:2].[N,O,C;!$(NC=O):3][c:4]>>[*:3][*:2]')

# Read RDKit molecules from previously generated files
file = open('../data/amine_list.txt', 'r')
amine_list = file.read().splitlines()
amine_list = [MolFromSmiles(smi) for smi in amine_list]
if mpi_rank==0:
    print('Finised reading amines')

file2 = open('../data/penul_lib.txt', 'r')
penultimate_lib = file2.read().splitlines()
len_lib = len(penultimate_lib)
if mpi_rank==0:
    print('Finised reading penultimates')

# read job index for library enumeration, read in respective subarray from penul_lib for enumeration
index = int(sys.argv[1])
job_size = int(sys.argv[2])
border_low, border_high = return_borders(index, len_lib, size=job_size)
penultimate_lib = penultimate_lib[border_low:border_high]
my_low, my_high = return_borders(mpi_rank, len(penultimate_lib), size=mpi_size)
penultimate_lib = penultimate_lib[my_low:my_high]

penultimate_lib_mols = [MolFromSmiles(smi) for smi in penultimate_lib]
print('Finished converting to RDKit Mols')

# Run enumeration, canonicalization, and filtering by rule-of-5 before saving to file
extra_mols = pair_rxnts(amine_list, penultimate_lib_mols, extra)
if mpi_rank==0:
    print('Finished running reactions')

final_lib = canonicalize_and_filter(penultimate_lib + extra_mols, showprogress=True)
if mpi_rank==0:
    print('Finished canonicalisation and rule-of-5 filtering')

file = open('../data/noncovalent_lib_final_batch_'+str(index)+'_rank_'+str(mpi_rank)+'.txt', 'w')
file.write('SMILES\n')
file.writelines("%s\n" % mol for mol in final_lib)
