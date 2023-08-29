import numpy as np
import pandas as pd
from math import sqrt
from scipy.stats import t, pearsonr, spearmanr
from sklearn.impute import SimpleImputer
from scipy.stats import shapiro, normaltest, ks_2samp, bartlett, fligner, levene, chi2_contingency
from scipy import stats
from statsmodels.formula.api import ols
#import re
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler,PolynomialFeatures
from sklearn.model_selection import train_test_split
from pca import pca
from statsmodels.formula.api import logit
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, accuracy_score, precision_score, f1_score,recall_score,r2_score,mean_squared_error,mean_absolute_error
from sklearn.linear_model import LinearRegression

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from pmdarima.arima import auto_arima
import seaborn as sb
from matplotlib import pyplot as plt
from tabulate import tabulate

def setCategory(df, fields=[],labelling=True):
    """
    데이터 프레임에서 지정된 필드를 범주형으로 변경한다.

    Parameters
    -------
    - df: 데이터 프레임
    - fields: 범주형으로 변경할 필드명 리스트. 기본값은 빈 리스트(전체 필드 대상)

    Returns
    -------
    - cdf: 범주형으로 변경된 데이터 프레임
    """
    cdf = df.copy()
    # 데이터 프레임의 변수명을 리스트로 변환
    ilist = list(cdf.dtypes.index)
    # 데이터 프레임의 변수형을 리스트로 변환
    vlist = list(cdf.dtypes.values)

    # 변수형에 대한 반복 처리
    for i, v in enumerate(vlist):
        # 변수형이 object이면?
        if v == 'object':
            # 변수명을 가져온다.
            field_name = ilist[i]

            # 대상 필드 목록이 설정되지 않거나(전체필드 대상), 현재 필드가 대상 필드목록에 포함되어 있지 않다면?
            if not fields or field_name not in fields:
                continue

            # 가져온 변수명에 대해 값의 종류별로 빈도를 카운트 한 후 인덱스 이름순으로 정렬
            # vc = cdf[field_name].value_counts().sort_index()
            # print(vc)

            # 인덱스 이름순으로 정렬된 값의 종류별로 반복 처리
            # for ii, vv in enumerate(list(vc.index)):
                # 일련번호값 생성
                # vnum = ii + 1
                # print(vv, " -->", vnum)

                # 일련번호값으로 치환
            # cdf.loc[cdf[field_name] == vv, field_name] = vnum

            # 해당 변수의 데이터 타입을 범주형으로 변환
            cdf[field_name] = cdf[field_name].astype('category')
            if labelling : 
                mydict = {}
                for i,v in enumerate(cdf[field_name].dtypes.categories) :
                    mydict[v]=i
                cdf[field_name] = cdf[field_name].map(mydict).astype(int)

    return cdf
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



def pearson_r(df) :
    names = df.columns 
    n = len(names)
    pv=0.05
    data = []
    for i in range(0,n):
        j=i+1 if i<(n-1) else 0
        fields = names[i] +"vs" +names[j]
        s,p = stats.pearsonr(df[names[i]],df[names[j]])
        result = p<pv
        data.append({'fields': fields, 'statistic': s, 'pvalue': p, 'result': result})
    rdf = pd.DataFrame(data)
    rdf.set_index('fields', inplace=True)
    return rdf

def ext_ols(data,y=None,x=None,expr=None) :
    """
    회귀분석을 수해한다.

    Parameters
    -------
    - data : 데이터 프레임
    - y: 종속변수 이름
    - x: 독립변수의 이름들(리스트)
    """
    df = data.copy()
    if not expr :
        # 독립변수의 이름이 리스트가 아니면 리스트로 변환
        if type(x) != list :
            x= [x]
        ## 종속변수 ~ 독힙변수1 + 독립변수2 + ... 형태의 식을 생성
        expr = "%s ~ %s" % (y,"+".join(x))
    else : 
        x= []
        p = expr.find('~')
        y = expr[:p].strip()
        x_tmp= expr[p+1:]
        x_list = x_tmp.split('+')
        for i in x_list:
            k = i.strip()
            if k :
                x.append(k)
    # 회귀모델 생성
    model = ols(expr,data=data)
    ## q분석수행
    fit = model.fit()
    ## 분석결과를 저장
    summary = fit.summary()

    ## 결과의 첫번쨰, 세번째 내용을 딕셔너리로 분해
    my = {}
    for k in range(0,3,2):
        items = summary.tables[k].data
        for item in items :
            n = len(item)
            for i in range(0,n,2) :
                key = item[i].strip()[:-1]
                value = item[i+1].strip()
                if key and value :
                    my[key]=value
    ## 두번쨰 표의 내용을 딕셔너리에 저장 
    my['variables']=[]
    name_list = list(data.columns)
    for i , v in enumerate(summary.tables[1].data) :
        if i==0 :
            continue
        # 변수의 이름
        name = v[0].strip()
        
        vif = 0

        # 0번쨰인 intercepts는 제외
        if name in name_list :
            j=name_list.index(name)
            vif = variance_inflation_factor(data,j)
        my['variables'].append({
            "name": name,
            "coef": v[1].strip(),
            "std err": v[2].strip(),
            "t": v[3].strip(),
            "P-value": v[4].strip(),
            "Beta": 0,
            "VIF": vif,
        })
        ## 결과표를 데이터 프레임으로 구성
    mylist = []
    yname_list = []
    xname_list = []

    for i in my['variables'] :
        yname_list.append(y)
        xname_list.append(i['name'])

        item = {
            "B": i['coef'],
            "표준오차": i['std err'],
            "β": i['Beta'],
            "t": "%s*" % i['t'],
            "유의확률": i['P-value'],
            "VIF": i["VIF"]
        }

        mylist.append(item)
    table = pd.DataFrame(mylist,
                   index=pd.MultiIndex.from_arrays([yname_list, xname_list], names=['종속변수', '독립변수']))
    # 분석결과
    result = "𝑅(%s), 𝑅^2(%s), 𝐹(%s), 유의확률(%s), Durbin-Watson(%s)" % (my['R-squared'], my['Adj. R-squared'], my['F-statistic'], my['Prob (F-statistic)'], my['Durbin-Watson'])

    # 모형 적합도 보고
    goodness = "%s에 대하여 %s로 예측하는 회귀분석을 실시한 결과, 이 회귀모형은 통계적으로 %s(F(%s,%s) = %s, p < 0.05)." % (y, ",".join(x), "유의하다" if float(my['Prob (F-statistic)']) < 0.05 else "유의하지 않다", my['Df Model'], my['Df Residuals'], my['F-statistic'])

    # 독립변수 보고
    varstr = []

    for i, v in enumerate(my['variables']):
        if i == 0:
            continue
        
        s = "%s의 회귀계수는 %s(p%s0.05)로, %s에 대하여 %s."
        k = s % (v['name'], v['coef'], "<" if float(v['P-value']) < 0.05 else '>', y, '유의미한 예측변인인 것으로 나타났다' if float(v['P-value']) < 0.05 else '유의하지 않은 예측변인인 것으로 나타났다')

        varstr.append(k)

    # 리턴
    return (model, fit, summary, table, result, goodness, varstr)

class RegMetric:
    def __init__(self, y, y_pred):
        # 설명력
        self._r2 = r2_score(y, y_pred)
        # 평균절대오차
        self._mae = mean_absolute_error(y, y_pred)
        # 평균 제곱 오차
        self._mse = mean_squared_error(y, y_pred)
        # 평균 오차
        self._rmse = np.sqrt(self._mse)
        
        # 평균 절대 백분오차 비율
        if type(y) ==pd.Series:
            self._mape = np.mean(np.abs((y.values - y_pred) / y.values) * 100)
        else:
            self._mape = np.mean(np.abs((y - y_pred) / y) * 100)
        
        # 평균 비율 오차
        if type(y) == pd.Series:   
            self._mpe = np.mean((y.values - y_pred) / y.values * 100)
        else:
            self._mpe = np.mean((y - y_pred) / y * 100)

    @property
    def r2(self):
        return self._r2

    @r2.setter
    def r2(self, value):
        self._r2 = value

    @property
    def mae(self):
        return self._mae

    @mae.setter
    def mae(self, value):
        self._mae = value

    @property
    def mse(self):
        return self._mse

    @mse.setter
    def mse(self, value):
        self._mse = value

    @property
    def rmse(self):
        return self._rmse

    @rmse.setter
    def rmse(self, value):
        self._rmse = value

    @property
    def mape(self):
        return self._mape

    @mape.setter
    def mape(self, value):
        self._mape = value

    @property
    def mpe(self):
        return self._mpe

    @mpe.setter
    def mpe(self, value):
        self._mpe = value

class OlsResult :
    def __init__(self):
        self._x_train = None
        self._y_train = None
        self._train_pred = None
        self._x_test = None
        self._y_test = None
        self._test_pred = None




        self._model = None
        self._fit = None
        self._summary = None
        self._table = None
        self._result = None
        self._goodness = None
        self._varstr = None
        self._coef =None
        self._intercept = None
        self._trainRegMetric = None
        self._testRegMetric = None
        
    @property
    def x_train(self):
        return self._x_train

    @x_train.setter
    def x_train(self, value):
        self._x_train = value

    @property
    def y_train(self):
        return self._y_train

    @y_train.setter
    def y_train(self, value):
        self._y_train = value

    @property
    def train_pred(self):
        return self._train_pred

    @train_pred.setter
    def train_pred(self, value):
        self._train_pred = value

    @property
    def x_test(self):
        return self._x_test

    @x_test.setter
    def x_test(self, value):
        self._x_test = value

    @property
    def y_test(self):
        return self._y_test

    @y_test.setter
    def y_test(self, value):
        self._y_test = value

    @property
    def test_pred(self):
        return self._test_pred

    @test_pred.setter
    def test_pred(self, value):
        self._test_pred = value

    @property 
    def model(self) :
        """
        분석모델
        """
        return self._model
    @model.setter
    def model(self,value):
        self._model = value 
    @property
    def fit(self) :
        """
        분석결과 객체
        """
        return self._fit
    @fit.setter
    def fit(self,value):
        self._fit = value
    
    @property
    def summary(self):
        """
        분석결과 요약 보고
        """
        return self._summary
    @summary.setter
    def summary(self,value):
        self._summary = value
    @property 
    def table(self):
        """
        결과표
        """
        return self._table
    @table.setter
    def table(self,value):
        self._table = value
    @property
    def result(self):
        """
        결과표 부가 설명
        """
        return self._result
    @table.setter
    def result(self,value) :
        self._result=value
    @property
    def goodness(self):
        """
        모형 적합도 보고
        """
        return self._goodness
    @goodness.setter
    def goodness(self,value):
        self._goodness=value
    @property
    def varstr(self):
        """
        독립변수 보고
        """
        return self._varstr

    @varstr.setter
    def varstr(self, value):
        self._varstr = value

    @property
    def coef(self):
        return self._coef
    @coef.setter
    def coef(self,value):
        self._coef = value
    
    @property
    def intercept(self):
        return self._intercept
    @intercept.setter
    def intercept(self,value):
        self._intercept= value
    @property
    def trainRegMetric(self):
        return self._trainRegMetric
    @trainRegMetric.setter
    def trainRegMetric(self,value):
        self._trainRegMetric = value
    @property
    def testRegMetric(self):
        return self._testRegMetric
    @testRegMetric.setter
    def testRegMetric(self,value):
        self._testRegMetric = value
    def setRegMetric(self,y_train,y_train_pred,y_test=None,y_test_pred=None):
        self.trainRegMetric = RegMetric(y_train,y_train_pred)

        if y_test is not None and y_test_pred is not None :
            self.testRegMetric = RegMetric(y_test,y_test_pred)           
	
def my_ols(data,x,y,expr=None) :
    model, fit, summary, table, result, goodness, varstr = ext_ols(data, y, x)
    ols_result = OlsResult()
    ols_result.model = model
    ols_result.fit = fit
    ols_result.summary = summary
    ols_result.table = table
    ols_result.result = result
    ols_result.goodness = goodness
    ols_result.varstr = varstr

    return ols_result
    

def scailing(df,yname=None):
    """
    데이터 프레임을 표준화 한다.

    Parameters
    -------
    - df: 데이터 프레임
    - yname: 종속변수 이름

    Returns
    -------
    - x_train_std_df: 표준화된 독립변수 데이터 프레임
    - y_train_std_df: 표준화된 종속변수 데이터 프레임
    """
    x_train = df.drop([yname],axis=1) if yname else df.copy()
    x_train_std = StandardScaler().fit_transform(x_train)
    x_train_std_df = pd.DataFrame(x_train_std,columns=x_train.columns)
    if yname:
        y_train=df.filter([yname])
        y_train_std = StandardScaler().fit_transform(y_train)
        y_train_std_df = pd.DataFrame(y_train_std,columns=y_train.columns)
    if yname :
        result = (x_train_std,y_train_std)
    else :
        result = x_train_std_df
    return result

def get_best_feature(x_train_std_df):
    pca_model=pca()
    fit = pca_model.fit_transform(x_train_std_df)
    topfeat_df = fit['topfeat']
    best = topfeat_df.query("type=='best'")
    feature = list(set(list(best['feature'])))
    return (feature,topfeat_df)
class LogitResult:
    def __init__(self):
        self._model=None
        self._fit=None
        self._summary=None
        self._prs=None
        self._cmdf=None
        self._result_df=None
        self._odds_ratee_df=None

    @property
    def model(self):
        return self._model
    @model.setter
    def model(self,value):
        self._model = value
    @property
    def fit(self) :
        return self._fit
    @fit.setter
    def fit(self,value):
        self._fit = value

    @property
    def summary(self):
        return self._summary

    @summary.setter
    def summary(self, value):
        self._summary = value

    @property
    def prs(self):
        return self._prs

    @prs.setter
    def prs(self, value):
        self._prs = value

    @property
    def cmdf(self):
        return self._cmdf

    @cmdf.setter
    def cmdf(self, value):
        self._cmdf = value

    @property
    def result_df(self):
        return self._result_df

    @result_df.setter
    def result_df(self, value):
        self._result_df = value

    @property
    def odds_rate_df(self):
        return self._odds_rate_df

    @odds_rate_df.setter
    def odds_rate_df(self, value):
        self._odds_rate_df = value    
def my_logit(data,y,x):
    """
    로지스틱 회귀분석을 수행한다.

    Parameters
    -------
    - data : 데이터 프레임
    - y: 종속변수 이름
    - x: 독립변수의 이름들(리스트)
    """

    # 데이터프레임 복사
    df = data.copy()

    # 독립변수의 이름이 리스트가 아니라면 리스트로 변환
    if type(x) !=list:
        x= [x]
    # 종속변수~독립변수1+독립변수2+독립변수3+... 형태의 식을 생성
    expr = "%s~%s" %(y,"+".join(x))
    # 회귀모델 생성
    model = logit(expr,data=df)

    fit= model.fit()
    # 분석결과의 저장
    summary = fit.summary   
    # 의사결정 계수
    prs = fit.prsquared

    ## 예측결과를 데이터 프레임에 추가
    df['예측값'] = fit.predict(df.drop([y],axis=1))
    df['예측결과'] = df['예측값']>0.5

    ## 혼동행력 
    cm = confusion_matrix(df[y],df['예측결과'])
    tn,tp,   fn,fp = cm.ravel()
    cmdf = pd.DataFrame(cm,index =['True', 'False'], columns=['Positive', 'Negative'])
    ## RAS
    ras = roc_auc_score(df[y],df['예측결과'])
    # 위양성율, 재현율, 임계값(사용안함)
    fpr, tpr, thresholds = roc_curve(df[y], df['예측결과'])

    # 정확도
    acc = accuracy_score(df[y], df['예측결과'])

    # 정밀도
    pre = precision_score(df[y], df['예측결과'])
    ## 재현율
    recall = recall_score(df[y],df['예측결과'])
    # F1 score
    f1 = f1_score(df[y],df['예측결과'])
    # 위양성율
    fallout = fp / (fp + tn)

    # 특이성
    spe = 1 - fallout

    result_df = pd.DataFrame({'설명력(Pseudo-Rsqe)': [fit.prsquared], '정확도(Accuracy)':[acc], '정밀도(Precision)':[pre], '재현율(Recall, TPR)':[recall], '위양성율(Fallout, FPR)': [fallout], '특이성(Specificity, TNR)':[spe], 'RAS': [ras], 'f1_score':[f1]})
    # 오즈비
    coef = fit.params
    odds_rate = np.exp(coef)
    odds_rate_df = pd.DataFrame(odds_rate, columns=['odds_rate'])
    
    #return (model, fit, summary, prs, cmdf, result_df, odds_rate_df)
    logit_result = LogitResult()
    logit_result.model = model
    logit_result.fit = fit
    logit_result.summary = summary
    logit_result.prs = prs
    logit_result.cmdf = cmdf
    logit_result.result_df = result_df
    logit_result.odds_rate_df = odds_rate_df

    return logit_result


def exp_timedata(data,yname,sd_model="m",max_diff=1):
    df = data.copy()

    ## 데이터 정상성 여부
    stationality = False

    ## 반복 수행 횟수

    count = 0 

    # 결측치 존재 여부
    na_count = df[yname].isnull().sum()
    print('결측치 수 : ',na_count)

    ## Box Plot
    sb.boxplot(data=df, y=yname)
    plt.show()
    plt.close()

    ## 시계열 분해

    model_name ='multiplicative' if sd_model=='m' else 'additive'
    sd = seasonal_decompose(df[yname],model=model_name)

    figure = sd.plot()
    figure.set_figwidth(15)
    figure.set_figheight(16)
    fig, ax1,ax2,ax3,ax4 = figure.get_children()
    figure.subplots_adjust(hspace = 0.4)
    ax1.set_ylabel("Original")
    ax1.grid(True)
    ax2.grid(True)
    ax3.grid(True)
    ax4.grid(True)

    plt.show()

    while not stationality:
        if count == 0 :
            print("=========== 원본 데이터 ===========")
        else : 
            print("=========== %d차 차분 데이터 ===========" % count)
        
        # ADF test 
        ar = adfuller(df[yname])
        ## 리스트를 원소로 갖는 딕셔너리
        ardict = {
            '검정통계량 (ADF Statistic) ' : [ar[0]],
            'p-value ':[ar[1]],
            '최적 차수 ':[ar[2]],
            '관측치 개수':[ar[3]]
        }
        for key,value in ar[4].items() : 
            ardict['기각값 %s' % key] = value
        
        stationality = ar[1]<0.05
        ardict['데이터 정상성 여부(0=Flase,1=True)'] = stationality

        ardf = pd.DataFrame(ardict,index = ['ADF']).T
        print(tabulate(ardf, headers=["ADF", ""], tablefmt='psql', numalign="right"))
    
        # ACF, PACF 검정
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12))
        fig.subplots_adjust(hspace=0.4)

        ax1.title.set_text("Original")
        sb.lineplot(data=df, x=df.index, y=yname, ax=ax1)

        ax2.title.set_text("ACF Test")
        plot_acf(df[yname], ax=ax2)
        
        ax3.title.set_text("PACF Test")
        plot_pacf(df[yname], ax=ax3)
        
        plt.show()
        plt.close()

        # 차분 수행
        
        df = df.diff().dropna()

        ## 반복을 계속할지 여부 판단
        count +=1
        if count == max_diff:
            break

def set_datetime_index(df,field = None, inplace = False):
    """
        데이터 프레임의 인덱스를 datetime 형식으로 변환

        Parameters
        -------
        - df: 데이터 프레임
        - inplace: 원본 데이터 프레임에 적용 여부

        Returns
        -------
        - 인덱스가 datetime 형식으로 변환된 데이터 프레임
    """
    if inplace:
        if field is not None :
            df.set_index(field,inplace = True)
        df.index = pd.DatetimeIndex(df.index.values,freq = df.index.inferred_freq)
        df.sort_index(inplace = True)
    else : 
        cdf = df.copy()
        if field is not None :
            cdf.set_index(field,inplace = True)

        cdf.index = pd.DatetimeIndex(cdf.index.values,freq = cdf.index.inferred_freq)
        cdf.sort_index(inplace=True)
        return cdf


def convertPoly(data,degree=2,include_bias=False):
    poly = PolynomialFeatures(degree=degree,include_bias=include_bias)
    fit = poly.fit_transform(data)
    x= pd.DataFrame(fit,columns=poly.get_feature_names_out())
    return x 

def getTrend(x,y,degree=2,value_count = 100):
    coeff = np.polyfit(x,y,degree)
    if type(x) == 'list' :
        minx = min(x)
        maxx= max(x)
    else : 
        minx = x.min()
        maxx = x.max()

    
    vtrend = np.linspace(minx,maxx,value_count)
    ttrend = coeff[-1]
    for i in range(0,degree):
        ttrend += coeff[i] * vtrend**(degree - i)
    return vtrend,ttrend


def regplot(x_left, y_left, y_left_pred=None, left_title=None, x_right=None, y_right=None, y_right_pred=None, right_title=None, figsize=(10, 5), save_path=None):
    subcount = 1 if x_right is None else 2
    
    fig, ax = plt.subplots(1, subcount, figsize=figsize)
    
    axmain = ax if subcount == 1 else ax[0]
    
    # 왼쪽 산점도
    sb.scatterplot(x=x_left, y=y_left, label='data', ax=axmain)
    
    # 왼쪽 추세선
    x, y = getTrend(x_left, y_left)
    sb.lineplot(x=x, y=y, color='blue', linestyle="--", ax=axmain)
    
    # 왼쪽 추정치
    if y_left_pred is not None:
        sb.scatterplot(x=x_left, y=y_left_pred, label='predict', ax=axmain)
        # 추정치에 대한 추세선
        x, y = getTrend(x_left, y_left_pred)
        sb.lineplot(x=x, y=y, color='red', linestyle="--", ax=axmain)
    
    if left_title is not None:
        axmain.set_title(left_title)
        
    axmain.legend()
    axmain.grid()
    
    
    if x_right is not None:
        # 오른쪽 산점도
        sb.scatterplot(x=x_right, y=y_right, label='data', ax=ax[1])
        
        # 오른쪽 추세선
        x, y = getTrend(x_right, y_right)
        sb.lineplot(x=x, y=y, color='blue', linestyle="--", ax=ax[1])
    
        # 오른쪽 추정치
        if y_right_pred is not None:
            sb.scatterplot(x=x_right, y=y_right_pred, label='predict', ax=ax[1])
            # 추정치에 대한 추세선
            x, y = getTrend(x_right, y_right_pred)
            sb.lineplot(x=x, y=y, color='red', linestyle="--", ax=ax[1])
        
        if right_title is not None:
            ax[1].set_title(right_title)
            
        ax[1].legend()
        ax[1].grid()
    
    if save_path is not None:
        plt.savefig(save_path, dpi=300)
        
    plt.show()
    plt.close()

def ml_ols(data,xnames,yname,degree=1,test_size=0.25,scailing=False,random_state=777):
    ## 표준화 안되있으면 표준화 수행
    if scailing:
        data =scailing(data)

     # 독립변수 이름이 문자열로 전달되었다면 콤마 단위로 잘라서 리스트로 변환
    if type(xnames) == str:
        xnames = xnames.split(',')
    # 독립변수 추출
    x = data.filter(xnames)
    
    # 종속변수 추출
    y = data[yname]
    # 2차식 이상으로 설정되었다면 차수에 맞게 변환
    if degree > 1:
        x = convertPoly(x, degree=degree)
    
    # 데이터 분할 비율이 0보다 크다면 분할 수행
    if test_size > 0:
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=random_state)
    else:
        x_train = x
        y_train = y
        x_test = None
        y_test = None
    # 회귀분석 수행
    model = LinearRegression()
    fit = model.fit(x_train, y_train)
    
    result = OlsResult()
    result.model = model
    result.fit = fit
    result.coef = fit.coef_
    result.intercept = fit.intercept_

    result.x_train = x_train.copy()
    result.y_train = y_train.copy()
    result.train_pred = result.fit.predict(result.x_train)
    
    if x_test is not None and y_test is not None:
        result.x_test = x_test.copy()
        result.y_test = y_test.copy()
        result.test_pred = result.fit.predict(result.x_test)
        result.setRegMetric(y_train, result.train_pred, y_test, result.test_pred)
        
    else:
        result.setRegMetric(y_train, result.train_pred)
        
    # 절편과 계수를 하나의 배열로 결합
    params = np.append(result.intercept, result.coef)    
    
    # 상수항 추가하기
    designX = x.copy()
    designX.insert(0, '상수', 1)   
    
    # 행렬곱 구하기
    dot = np.dot(designX.T,designX)
    
    # 행렬곱에 대한 역행렬 
    inv = np.linalg.inv(dot)  
    
    # 역행렬의 대각선 반환  
    dia = inv.diagonal()
    
    # 평균 제곱오차 구하기
    predictions = result.fit.predict(x)
    MSE = (sum((y.values-predictions)**2)) / (len(designX)-len(designX.iloc[0]))
    
    # 표준오차
    se_b = np.sqrt(MSE * dia)
    
    # t값
    ts_b = params / se_b
    
    # p값
    p_values = [2*(1-stats.t.cdf(np.abs(i),(len(designX)-len(designX.iloc[0])))) for i in ts_b]
    
    # vif
    vif = []
    # 룬현데이터에 대한 독립변수와 종속변수를 결합한 완전한 데이터프레임 준비
    data= x_train.copy()
    data[yname] = y_train

    for i, v in enumerate(x_train.column):
        j = list(data.columns).index(v)
        vif.append(variance_inflation_factor(data, j))
    
    # 결과표 구성하기
    result.table = pd.DataFrame({
        "종속변수": [yname] * len(x_train.columns),
        "독립변수": x_train.columns,
        "B": result.coef[0],
        "표준오차": se_b[1:],
        "β": 0,
        "t": ts_b[1:],
        "유의확률": p_values[1:],
        "VIF": vif,
    })   
    return result
    