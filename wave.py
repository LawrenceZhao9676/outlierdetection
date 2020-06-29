#导入需要的包
import pandas as pd
import numpy as np
import glob,os
from pyod.models.hbos import HBOS
from pyod.models.knn import KNN
from pyod.models.ocsvm import OCSVM
result=open('wave_result-OCSVM.txt','a')
path=r'.\wave\benchmarks'
file=os.listdir(path)
i=0
for f in file:#读取数据
    df=pd.read_csv(path+'\\'+f)
    label=df['ground.truth']
    df=df.drop(['ground.truth','point.id','motherset','origin'],axis=1)    
    clf = OCSVM()
    score=clf.fit_predict_score(df, label)
    inputf=str(f)+'\t'+str(score)+'\n'
    result.write(inputf)    
    i+=1
    if i%100==0:
        print(str(i)+'\n')
result.close()
