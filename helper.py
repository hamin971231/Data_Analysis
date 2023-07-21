import numpy as np
import pandas as pd
from math import sqrt
from scipy.stats import t
from sklearn.impute import SimpleImputer
from scipy.stats import shapiro, normaltest, ks_2samp, bartlett, fligner, levene, chi2_contingency


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

def clearStopwords(nouns,stopwords_file_path = "wordcloud/stopwords-ko.txt") :
    with open(stopwords_file_path,"r",encoding="utf-8") as f :
        stopwords = f.readlines()
        
        for i,v in enumerate(stopwords) :
            stopwords[i] = v.strip()
        data_set = []

        for v in nouns :
            if v not in stopwords :
                data_set.append(v)

        return data_set
## 정규성 검정
def normality_test(*any):
    ## 인덱스를 위한 이름 리스트
    names = []

    ## 결과를 저장할 리스트 
    result = {
        "Statistic" :[],
        "p-value":[],
        'Result' :[]
    }

    ## shapiro-wills Test
    for i in any :
        s, p = shapiro(i)
        result['Statistic'].append(s)
        result['p-value'].append(p)
        result['Result'].append(p>0.05)
        names.append(('정규성','Shapiro',i.name))
    
    # normal Test 
    for i in any :
        s,p=normaltest(i)
        result['Statistic'].append(s)
        result['p-value'].append(p)
        result['Result'].append(p>0.05)
        names.append(('정규성','normal',i.name))
    
    ## k-s 검정은 정규성 여부를 각각의 필드 서로서로 해야함 -> 반복 필요
    n = len(any)
    for i in range(0,n):
        j = i + 1 if (i < n-1) else 0
        s,p = ks_2samp(any[i],any[j])
        result['Statistic'].append(s)
        result['p-value'].append(p)
        result['Result'].append(p>0.05)
        names.append(('정규성','k-s_2samp',f'{any[i].name} vs {any[j].name}'))
    
    
    return pd.DataFrame(result,index=pd.MultiIndex.from_tuples(names,names=['Condition','Test','Field']))
        
## 등분산성 검정
def equal_variance_test(*any) :
    # statistic=1.333315753388535, pvalue=0.2633161881599037
    ## 검정결과 저장
    s1,p1 = bartlett(*any)
    s2,p2 = fligner(*any)
    s3,p3 = levene(*any)
    ## 결과를 저장할 리스트

    names= []

    for i in any : 
        names.append(i.name)

    fix =" vs "
    name = fix.join(names) 
    index = [['등분산성','Bartlett',name],['등분산성','Fligner',name],['등분산성','Levene',name]]

    result = pd.DataFrame({
        'Statistic' : [s1, s2, s3],
        "p-value" : [p1, p2, p3],
        "Result" : [p1>0.05,p2>0.05,p3>0.05]
    },index=pd.MultiIndex.from_tuples(index,names=['Condition','Test','Field']))
    return result

## 독립성 검정 

def independence_test(*any):
    df = pd.DataFrame(any).T
    result = chi2_contingency(df)

    names = []
    for i in any :
        names.append(i.name)
    fix = " vs "
    name = fix.join(names)
    index = [['독립성','Chi2',name]]
    df = pd.DataFrame({
        "Statistic":[result.statistic],
        "p-value":[result.pvalue],
        "Result":[result.pvalue>0.05]
    },index = pd.MultiIndex.from_tuples(index, names=['Condition','Test','Field']))
    return df

def all_test(*any) :
    return pd.concat([normality_test(*any),equal_variance_test(*any),independence_test(*any)])