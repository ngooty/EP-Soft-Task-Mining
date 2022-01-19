# %%
import getopt, sys, pickle,pandas,os
# To be used for creating pipelines and personalizing them
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

curr_dir=os.getcwd()
dataset=curr_dir+'/Datasets/HRMS Employee Creation oversampled data.csv'
model_name=curr_dir+'/Models/noise_reduction_model.sav'
output_file=curr_dir+'/Output/HRMS Employee Creation data without noise.csv'


try:
    print ("Displaying Dataset Name:", dataset)
    df=pandas.read_csv(dataset,dtype='unicode')
    X=df.drop(['ActiveWindow'],axis=1)
    y=df['ActiveWindow']
    loaded_model = pickle.load(open(model_name, 'rb'))
    result=loaded_model.predict(X)
    result_df=pandas.DataFrame()
    active_window=[]
    for i in range(len(y)):
        if result[i]==y[i]:
            active_window.append(result[i])
            result_df=result_df.append(X.iloc[[i]])
            #result_df['ActiveWindow']=active_window     
    result_df.insert(9,'ActiveWindow',active_window)
    result_df.to_csv(output_file,index=False)
    output_file='HRMS Employee Creation data without noise.csv'  
    print('Output file is: ',output_file)          
             
except getopt.error as err:
    # output error, and return with an error code
    print (str(err))





# %%
