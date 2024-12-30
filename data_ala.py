import pandas as pd
import numpy as np
from scipy.stats import t,skew,kurtosis,ttest_1samp,ttest_ind,norm,kstest,chi2,levene
import matplotlib.pyplot as plt

data=pd.read_csv('iris.csv')
def descriptive(data):
    features=["Sep_len","Sep_wid","Pet_len","Pet_wid"]
    stats={}
    for i in features:
        means=data[i].mean()
        variance=data[i].var()
        skewness=skew(data[i])
        kurt=kurtosis(data[i])
        std_data=data[i].std()
        stats[i] = {
            "mean": means,
            "variance": variance,
            "skewness": skewness,
            "kurtosis": kurt,
        }       
        print(f"{i}:Mean:{means},variance:{variance},skewness:{skewness},kurtosis:{kurt}")
    plt.hist(data[i],bins=15,color="skyblue",edgecolor="black")
    plt.title(f"Histogram of {i}")
    plt.xlabel(i)
    plt.ylabel("Frequency")



    plt.boxplot(data[i],vert=False,patch_artist=True,boxprops=dict(facecolor="skyblue"))
    plt.title(f"Boxplot of {i}")
    plt.xlabel(i)

    plt.tight_layout()
    plt.show()


    return stats

def test(data,stats):
    alpha=0.05
    length=data["Sep_len"]
    variance=stats["Sep_len"]['variance']
    z_stat =(length.mean()-4.5)/np.sqrt(variance/len(length))
    p_of_z=2*(1-norm.cdf(abs(z_stat)))
    if p_of_z<alpha:
        print("Reject:The mean is different")
    else:
        print("Fail to reject:The mean is not different")
    t_stat,p_of_t=ttest_1samp(length,4.5)
    if p_of_t<alpha:
        print("Reject:The mean is different")
    else:
        print("Fail to reject:The mean is not different")

def compare(data,stats):
    alpha=0.05
    setosa=data[data["Iris_type"]=='Iris-setosa']["Sep_wid"]
    versicolor=data[data["Iris_type"]=='Iris-versicolor']["Sep_wid"]
    stat_var,p_var=levene(setosa,versicolor)
    if p_var>alpha:
        equal_var=True
        print("Yes")
    else:
        equal_var=False
        print("No")


    t_stat,p_t=ttest_ind(setosa,versicolor,equal_var=equal_var)
    if p_t<alpha:
        print("significantly different")
    else:
        print("not significantly different")


def predict(data,stats):
    confi=0.95
    vir_pet=data[data["Iris_type"]=="Iris-virginica"]["Pet_len"]
    vir_mean=vir_pet.mean()
    vir_var=vir_pet.var()
    vir_size=len(vir_pet)
    zxqj_low=vir_mean-(t.ppf(1-(1-confi)/2,df=vir_size-1))*np.sqrt(vir_var/vir_size)
    zxqj_high=vir_mean+(t.ppf(1-(1-confi)/2,df=vir_size-1))*np.sqrt(vir_var/vir_size)
    print(f"point predict:mean={vir_mean},variance={vir_var}")
    print(f"f confidence area:[{zxqj_low},{zxqj_high}]")






def model(data,stats):
    sep_len=data["Sep_len"]
    sep_len_mean=sep_len.mean()
    sep_std=sep_len.std()
    ks_stat,ks_p_value=kstest(sep_len,'norm',args=(sep_len_mean,sep_std))
    print(ks_stat)
    print(ks_p_value)


    prob=norm.cdf(6,loc=sep_len_mean,scale=sep_std)-norm.cdf(5,loc=sep_len_mean,scale=sep_std)
    print(f"probality of sep_len in[5,6]is {prob}")











stats=descriptive(data)
test(data,stats)
compare(data,stats)
predict(data,stats)
model(data,stats)
