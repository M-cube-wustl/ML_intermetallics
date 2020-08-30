## Add numerical descriptors via Matminer
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matminer.utils.conversions import str_to_composition
from matminer.featurizers.composition import ElementProperty

##check for missing values
def missing_values_table(df):
        # Total missing values
        mis_val = df.isnull().sum()
        
        # Percentage of missing values
        mis_val_percent = 100 * df.isnull().sum() / len(df)
        
        # Make a table with the results
        mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
        
        # Rename the columns
        mis_val_table_ren_columns = mis_val_table.rename(
        columns = {0 : 'Missing Values', 1 : '% of Total Values'})
        
        # Sort the table by percentage of missing descending
        mis_val_table_ren_columns = mis_val_table_ren_columns[
            mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(
        '% of Total Values', ascending=False).round(1)
        
        # Print some summary information
        print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"      
            "There are " + str(mis_val_table_ren_columns.shape[0]) +
              " columns that have missing values.")
        
        # Return the dataframe with missing information
        return mis_val_table_ren_columns

##read data
df_binary = pd.read_csv('../Datasets/Binary_intermetallics.csv')
df_el = pd.read_csv('../Datasets/metallic_elements.csv')
unwanted_columns = ['e_above_hull','energy_per_atom']
df_binary = df_binary.drop(unwanted_columns, axis=1)
df_el = df_el.drop(unwanted_columns, axis=1)

##transfer fomula to chemical compositions
from matminer.utils.conversions import str_to_composition
df_binary['composition'] = df_binary['pretty_formula'].transform(str_to_composition)
df_el['composition'] = df_el['pretty_formula'].transform(str_to_composition)

##featurization with elemental properties
from matminer.featurizers.composition import ElementProperty
## From magpie import CovalentRadius, BulkModulus, AtomicVolume, Cohesive_Energy(custom created) and MendeleevNumber...
features_1= ["NValence","Cohesive_Energy",'FirstIonizationEnergy','Electronegativity']
ep_feat_1 = ElementProperty(data_source="magpie", features=features_1 ,stats=["maximum","minimum","mean","mean_dev"])
df_binary = ep_feat_1.featurize_dataframe(df_binary, col_id="composition")  # input the "composition" column to the featurizer
df_el = ep_feat_1.featurize_dataframe(df_el, col_id="composition")  # input the "composition" column to the featurizer

##check for missing value
missing_values_table(df_binary)
missing_values_table(df_el)
