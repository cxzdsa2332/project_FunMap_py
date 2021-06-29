#import sys
#sys.modules[__name__].__dict__.clear()
import math
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
df_geno = pd.read_csv('geno_df.csv')
df_ck = pd.read_csv('ck_df.csv',index_col=0)
df_salt = pd.read_csv('salt_df.csv',index_col=0)

#select sum_root
df_ck = df_ck.loc[df_ck['roottype']=='sum_root'].iloc[:,1:15]
df_salt = df_salt[df_salt['roottype']=='sum_root'].iloc[:,1:15]

#get_mean for duplication
df_ck = df_ck.groupby(df_ck.index).mean()
df_salt = df_salt.groupby(df_salt.index).mean()

#clean data
df_pheno = pd.merge(df_ck,df_salt,left_index=True,right_index=True)
df_geno = df_geno.loc[:,map(lambda x:str(x),df_pheno.index.to_list())]

#biFunMap
i=2
df_pheno0 = df_pheno.loc[map(lambda x:int(x),df_geno.iloc[i,:][df_geno.iloc[i,:]==0].index)]
df_pheno1 = df_pheno.loc[map(lambda x:int(x),df_geno.iloc[i,:][df_geno.iloc[i,:]==1].index)]
df_pheno2 = df_pheno.loc[map(lambda x:int(x),df_geno.iloc[i,:][df_geno.iloc[i,:]==2].index)]
y_all = pd.concat([df_pheno0,df_pheno1,df_pheno2])

def get_miu(par):
    """mean_vector of growth curve"""
    miu = par[1]/(1+par[2]*math.exp(-par[3])
