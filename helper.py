import numpy as np 
import pandas as pd
from sklearn.impute import SimpleImputer

def replaceMissingValue(df):
    imr=SimpleImputer(missing_values=np.nan, strategy="mean")
    df_imr = imr.fit_transform(df.values)
    re_df = pd.DataFrame(df_imr,index=df.index,columns=df.columns)
    return re_df

## 이상치를 결측치로 바꾸기전에 이상치를 탐지하는 함수를 만들어야함
def getIq(field) :
    q1 = field.quantile(0.25)
    q3 = field.quantile(0.75)
    iqr = q3 - q1
    low = q1 - 1.5 * iqr
    high = q3 + 1.5 * iqr
    bound = [low,high]
    return bound

## 이상치를 결측치로 바꾸는 함수

def replaceOutlier(df,fieldName):
    cdf=df.copy()
    ## fieldName이 리스트가 아니면 리스트로 벼노한
    if not isinstance(fieldName,list) :
        fieldName = [fieldName]

    for f in fieldName :
        bound = getIq(cdf[f])
        cdf.loc[cdf[f]< bound[0],f] = np.nan
        cdf.loc[cdf[f]>bound[1],f] = np.nan
    return cdf