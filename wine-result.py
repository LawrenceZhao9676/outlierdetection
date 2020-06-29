#导入需要的包
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df=pd.read_csv('wine_result-OCSVM.txt',delimiter='\t')
df.columns=['num','result']
plt.figure()
a=[i for i in range(0,len(df['result']))]
plt.scatter(a,df['result'],s=5,alpha=0.7)
plt.xlabel("num of files")
plt.ylabel("auc-roc-scores")
plt.title("wine-OCSVM")
plt.show()
