import pandas as pd
import numpy as np
from rdkit.Chem import Draw, Lipinski, Descriptors, Crippen, MolFromSmiles, AllChem
from rdkit import Chem

# Read activity data
df = pd.read_csv('activity_data_13Jul2020.csv')

df['mol'] = [MolFromSmiles(x) for x in df['SMILES']]
acry = MolFromSmiles('O=C(C=C)N')
chloroace = MolFromSmiles('ClCC(=O)N')

df['acry'] = [x.HasSubstructMatch(acry) for x in df['mol']]
df['chloroace'] = [x.HasSubstructMatch(chloroace) for x in df['mol']]

df_active = df[df['f_avg_pIC50'].notnull()]
df_inactive = df[df['f_avg_pIC50'].isnull()]

with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
    #print(df['chloroace'])
    print('Number of actives: {}'.format(len(df_active)))
    print('Number of inactives: {}'.format(len(df_inactive)))

    print('Number of acry: {}'.format(len(df[df['acry']])))
    print('Number of chloroace: {}'.format(len(df[df['chloroace']])))
    print('Number of both: {}'.format(len(df[df['acry'] & df['chloroace']])))

df_acry_actives = df_active[df_active['acry']]
df_acry_inactives = df_inactive[df_inactive['acry']]
df_chloro_actives = df_active[df_active['chloroace']]
df_chloro_inactives = df_inactive[df_inactive['chloroace']]
df_noncovalent_actives = df_active[~df_active['acry'] & ~df_active['chloroace']]
df_noncovalent_inactives = df_inactive[~df_inactive['acry'] & ~df_inactive['chloroace']]

print('Old Numbers:')
print('Number of acry actives: {}'.format(len(df_acry_actives)))
print('Number of acry inactives: {}'.format(len(df_acry_inactives)))
print('Number of chloro actives: {}'.format(len(df_chloro_actives)))
print('Number of chloro inactives: {}'.format(len(df_chloro_inactives)))
print('Number of noncovalent actives: {}'.format(len(df_noncovalent_actives)))
print('Number of noncovalent inactives: {}'.format(len(df_noncovalent_inactives)))
print('Length of original dataset: {}, sum of split datasets: {}'.format(len(df), len(df_acry_actives) +
                                                                         len(df_acry_inactives) + len(df_chloro_actives) +
                                                                         len(df_chloro_inactives) + len(df_noncovalent_actives) +
                                                                         len(df_noncovalent_inactives)))
df['activity'] = df['f_avg_pIC50'].where(df['f_avg_pIC50'].isnull(),1)
df['activity'] = df['activity'].where(df['activity'].notnull(),0).astype(int)
df = df[~((df['activity']==0) & (df['f_inhibition_at_50_uM']>50))]

df_active = df[df['activity']==1]
df_inactive = df[df['activity']==0]
df_acry_actives = df_active[df_active['acry']]
df_acry_inactives = df_inactive[df_inactive['acry']]
df_chloro_actives = df_active[df_active['chloroace']]
df_chloro_inactives = df_inactive[df_inactive['chloroace']]
df_noncovalent_actives = df_active[~df_active['acry'] & ~df_active['chloroace']]
df_noncovalent_inactives = df_inactive[~df_inactive['acry'] & ~df_inactive['chloroace']]

print('New Numbers:')
print('Number of acry actives: {}'.format(len(df_acry_actives)))
print('Number of acry inactives: {}'.format(len(df_acry_inactives)))
print('Acry activity: {:.2f}%'.format(100*len(df_acry_actives)/(len(df_acry_inactives)+len(df_acry_actives))))
print('Number of chloro actives: {}'.format(len(df_chloro_actives)))
print('Number of chloro inactives: {}'.format(len(df_chloro_inactives)))
print('chloro activity: {:.2f}%'.format(100*len(df_chloro_actives)/(len(df_chloro_inactives)+len(df_chloro_actives))))
print('Number of noncovalent actives: {}'.format(len(df_noncovalent_actives)))
print('Number of noncovalent inactives: {}'.format(len(df_noncovalent_inactives)))
print('noncovalent activity: {:.2f}%'.format(100*len(df_noncovalent_actives)/(len(df_noncovalent_inactives)+len(df_noncovalent_actives))))

print('Length of original dataset: {}, sum of split datasets: {}'.format(len(df), len(df_acry_actives) +
                                                                         len(df_acry_inactives) + len(df_chloro_actives) +
                                                                         len(df_chloro_inactives) + len(df_noncovalent_actives) +
                                                                         len(df_noncovalent_inactives)))

df['activity'] = df['activity'].astype('Int64')

df_acry = df[df['acry']]
df_acry = df_acry[['SMILES','CID','activity','f_avg_IC50']]
df_acry.to_csv('acry_activity.smi', index=False)

df_chloro = df[df['chloroace']]
df_chloro = df_chloro[['SMILES','CID','activity','f_avg_IC50']]
df_chloro.to_csv('chloroace_activity.smi', index=False)

df_noncovalent = df[~df['acry'] & ~df['chloroace']]
df_noncovalent = df_noncovalent[['SMILES','CID','activity','f_avg_IC50']]
df_noncovalent.to_csv('noncovalent_activity.smi', index=False)

