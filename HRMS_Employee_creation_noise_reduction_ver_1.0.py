# %% [markdown]
# # Task Mining 

# %%
import numpy as np
import pandas as pd
import pandas_profiling as pp
import streamlit as st
import altair as alt
import matplotlib.pyplot as plt
import seaborn as sns
import quickda as qa
from quickda.explore_data import *
from quickda.clean_data import *
from quickda.explore_numeric import *
from quickda.explore_categoric import *
from quickda.explore_numeric_categoric import *
from quickda.explore_time_series import * 
#pd.options.display.max_colwidth=1000
#pd.options.display.max_rows=None
#pd.options.display.max_columns=None
pd.options.mode.chained_assignment = None
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import recall_score,confusion_matrix,accuracy_score,precision_score
import pickle

# %%
#df=pd.read_csv('HRMS Employee Creation Sample - 1000Records_Updated_withrandomplacement.csv')
#df=pd.read_csv('HRMS Employee Creation Sample - 10000Records.csv')
df=pd.read_csv('HRMS Employee Creation oversampled data.csv')

# %%
df.shape

# %%
#Taking 'ActiveWindow' as y in extracting nearest neighbors, 'ActiveWindow' is an imbalanced Target
df['ActiveWindow'].value_counts()

# %%
X=df.drop(['ActiveWindow'],axis=1)
y=df['ActiveWindow']

# %%
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
ohe=OneHotEncoder()
le=LabelEncoder()
encoded_x=ohe.fit_transform(X)

# %%
from sklearn.neighbors import KNeighborsClassifier,NearestNeighbors
knn=KNeighborsClassifier(n_neighbors=200,n_jobs=-1) 
nn=NearestNeighbors(n_neighbors=200,n_jobs=-1)

# %%
#model=knn.fit(encoded_xover,yover)
model=knn.fit(encoded_x,y)

# %%
# save the model to disk
filename = 'noise_reduction_model.sav'
pickle.dump(model, open(filename, 'wb'))

# %%
# load the model from disk
loaded_model = pickle.load(open(filename, 'rb'))
score = loaded_model.score(encoded_x,y)
print(score)

# %%
result=loaded_model.predict(encoded_x)

# %%
compare_df=pd.DataFrame({'Actual':y,'Predicted':result})

# %%
#compare_df[compare_df['Actual']!=compare_df['Predicted']]

# %%
recall_score(result,y,average='weighted',labels=np.unique(result))

# %%
result_df=pd.DataFrame()
active_window=[]
for i in range(len(y)):
     if result[i]==y[i]:
        active_window.append(result[i])
        result_df=result_df.append(X.iloc[[i]])
#result_df['ActiveWindow']=active_window     
result_df.insert(9,'ActiveWindow',active_window)

# %%
result_df.to_csv('HRMS Employee Creation data without noise.csv',index=False)

# %% [markdown]
# ## Verification

# %% [markdown]
# ### Noise is uploaded to Noise.csv

# %%
noise_df=pd.read_csv('Noise.csv')

# %%
noise_df.rename(columns={'Unnamed: 0':'Orig_row_index'},inplace=True)

# %% [markdown]
# ### Enter a session Id to verify

# %%
print('Total records in the dataset including noise: ',len(df))
print('Total records in the dataset after removing noise: ',len(result_df))
print('Total Noisy records in the dataset: ',len(noise_df))
print('Total noisy records removed: ', len(df) - len(result_df))
#print('Remaining Noisy records: ', len(noise_df) - (len(df) - len(result_df)))

# %% [markdown]
# ### Session Ids for which noise is added

# %%
noise_sess_id=noise_df['SessionId']
noise_sess_id

# %%
session_id=int(input())
diff_df=pd.DataFrame()
if (len(df[df['SessionId']==session_id])==len(result_df[result_df['SessionId']==session_id])):
    print('Noise record is not removed for the SessionId ',session_id)
else:
    diff_df=pd.concat([df[df['SessionId']==session_id],result_df[result_df['SessionId']==session_id]]).drop_duplicates(keep=False)
diff_df
    

# %%



