!/usr/bin/env python
## Intermetallic datasets retrieval from the Materials Project via Pymatgen and Matminer library 


import csv
import itertools
from pymatgen import Element, MPRester
from matminer.data_retrieval.retrieve_MP import MPDataRetrieval
mpr = MPRester("xxxxxxxxxxx")
mpdr = MPDataRetrieval(api_key='xxxxxxxxxx')

def binary_alloy_systems():
    """
    Return a sorted list of chemical systems
        of the form [...,"Na-Si",...,"Na-Tl",...]
    """
    first_el = {el.symbol for el in Element
                if el.is_alkali or el.is_alkaline or el.is_transition_metal or el.is_post_transition_metal}
    second_el = {el.symbol for el in Element
                 if el.is_alkali or el.is_alkaline or el.is_transition_metal or el.is_post_transition_metal}
    return sorted(["{}-{}".format(*sorted(pair))
                   for pair in itertools.product(first_el, second_el)])

def ternary_alloy_systems():

    first_el = {el.symbol for el in Element
                if el.is_alkali or el.is_alkaline or el.is_transition_metal or el.is_post_transition_metal}
    second_el = {el.symbol for el in Element
                 if el.is_alkali or el.is_alkaline or el.is_transition_metal or el.is_post_transition_metal}
    third_el = {el.symbol for el in Element
                 if el.is_alkali or el.is_alkaline or el.is_transition_metal or el.is_post_transition_metal}

    return sorted(["{}-{}-{}".format(*sorted(triple))
                   for triple in itertools.product(first_el, second_el,third_el)])

## Add any other properties you are interested in
df_binary_all = mpdr.get_dataframe({'chemsys': {'$in': binary_alloy_systems()}}, ['material_id','pretty_formula', 'e_above_hull','formation_energy_per_atom',
                                                                              'is_ordered'])
## Filter out ordered intermetallic compounds that are on convex hull
df_binary = df_binary_all.loc[(df_binary_all['e_above_hull']==0)&df_binary_all['is_ordered']==True]
## Save to csv file
df_binary.to_csv('../Datasets/binary_intermetallics.csv')
