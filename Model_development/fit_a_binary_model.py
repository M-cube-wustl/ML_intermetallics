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

##Allign data and normalization
excluded = ['pretty_formula','formation_energy_per_atom','composition']
X = df_binary.drop(excluded, axis=1)
X_el = df_el.drop(excluded, axis=1)
Y = df_binary.filter(items=['formation_energy_per_atom'],axis=1)
Y_el= df_el.filter(items=['formation_energy_per_atom'],axis=1)

from mlxtend.preprocessing import minmax_scaling
X_n = minmax_scaling(X, columns=X.columns)
X_el_n = minmax_scaling(X_el, columns=X_el.columns)

## Plot Pearson correlation
import seaborn as sns
df_corr = pd.concat([df_binary['formation_energy_per_atom'], X], axis = 1)
df_corr = df_corr.corr()
plt.figure(figsize = (6, 6) )
sns.heatmap(df_corr, cmap='RdYlBu', vmin = -0.8, vmax = 0.8)
plt.title('Heatmap', fontsize = 20)

## Train_test_split
from sklearn.model_selection import train_test_split
##use 20% of binary as testing set
X_1, X_2, Y_1, Y_2 = train_test_split(X_n, Y, test_size=0.2, random_state=2)
X_t=pd.concat([X_1, X_el_n])
Y_t=pd.concat([Y_1, Y_el]) ##add metallic elements as training data

##10-fold CV with GPR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, RationalQuadratic, ExpSineSquared, Matern
from sklearn.model_selection import cross_val_predict, KFold
kernel = 1.0*WhiteKernel()+1.0*RBF() 
gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5,
                              optimizer= 'fmin_l_bfgs_b', copy_X_train=True, normalize_y=True)
kf = KFold(n_splits=10,shuffle=True, random_state=12345)
for train_index, test_index in kf.split(X_t):
    X_train, X_test = X_t.iloc[train_index], X_t.iloc[test_index]
    Y_train, Y_test = Y_t.iloc[train_index], Y_t.iloc[test_index]
    gpr.fit(X_train,Y_train)
    gpr.score(X_train,Y_train) 
    gpr.score(X_test,Y_test)

def rmse (predictions, targets):
    return np.sqrt(((predictions-targets)**2).mean())
def MAE (predictions, targets):
    return np.mean(abs(predictions-targets))


##training set accuracy
predicted_1=gp_try.predict(X_1)
pred = np.array(predicted_1)
Y_1_array=np.array(Y_1)
print("root mean square error: %.3f" %(rmse(pred, Y_1_array)))
print("mean abosolute error: %.3f" %(MAE(pred, Y_1_array)))

##testing set accuracy
predicted_2=gp_try.predict(X_2)
pred_2 = np.array(predicted_2)
Y_2_array=np.array(Y_2)
print("root mean square error: %.3f" %(rmse(pred_2, Y_2_array)))
print("mean abosolute error: %.3f" %(MAE(pred_2, Y_2_array)))
