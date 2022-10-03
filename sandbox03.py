#!/usr/bin/env python
# coding: utf-8

# # ML macro forecast
# 
# ## 1. Getting data

# In[1]:


import tensorflow as tf
import ipywidgets as ipw
import pandas as pd
import numpy as np
import matplotlib as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sbn
from sklearn.metrics import mean_squared_error
import math as math
import warnings
warnings.filterwarnings("ignore")


# In[2]:


def transform_data(x, flag):
    if flag==1: 
        x=x
    if flag==2:
        x=(x-x.shift(4))
    if flag==3: 
        x=((x-x.shift(1))-(x.shift(4)-x.shift(5)))
    if flag==4: 
        x=np.log(x)
    if flag==5:
        x=np.log(x)-np.log(x.shift(4))
    if flag==6:
        x=np.log(x)-np.log(x.shift(4))
    if flag==7:
        x=((x/x.shift(4))-1)-((x.shift(1)/x.shift(5))-1)
    if flag==10:
        x=np.log((x/100)+1)
    return x


# In[3]:


def get_all_data(date_start, date_end):
    data_csv=pd.read_csv("https://s3.amazonaws.com/files.fred.stlouisfed.org/fred-md/quarterly/current.csv", index_col=0)
    transformation=data_csv.iloc[1,:]
    data=data_csv.iloc[2:,:]
    data=data.loc[data.index.dropna()]
    data.index=pd.to_datetime(data.index, format='%m/%d/%Y')
    colnames=data.columns
    for i in range(len(colnames)):
        data.loc[data.index, colnames[i]]=transform_data(x=data[colnames[i]], flag=transformation[i])
    data=data[(data.index>=date_start) & (data.index<=date_end)]
    return data


# In[4]:


all_data=get_all_data(date_start=pd.to_datetime('01/01/1960', format='%m/%d/%Y'),
                      date_end=pd.to_datetime('12/31/2019', format='%m/%d/%Y'))
sbn.set_style("whitegrid", {'grid.linestyle': '--'})
sbn.lineplot(y=all_data["PCEPILFE"], x=all_data.index)


# In[5]:


groups=pd.read_excel('C:/Users/hrapa/ML macro forecast/groups.xlsx')
groups_dict={}
for gr in groups["Group"].unique(): groups_dict[gr]=[groups["Name"][j] for j in groups[groups["Group"]==gr].index]
#print(groups_dict)


# In[6]:


def forecast_plots(all_dt, train_dt, val_dt, test_dt, 
                  colname, fcs_name, alpha, lw,
                  val_dt_start, val_dt_end,
                  test_dt_start, test_dt_end,
                  rmse_all, rmse_train, rmse_val, rmse_test):
    
    print("RMSE all data: ", rmse_all)
    print("RMSE train data: ", rmse_train)
    print("RMSE val data: ", rmse_val)
    print("RMSE test data: ", rmse_test)
        
    fig, axs = plt.pyplot.subplots(nrows=2, ncols=2, figsize=(20,20))
    fig.suptitle(fcs_name + ' ' + colname + ' forecast', fontsize=48, y=0.95)
    
    axs[0, 0].set_facecolor("mintcream")
    axs[0, 0].axvspan(val_dt_start, val_dt_end, color='y', alpha=alpha, lw=lw)
    axs[0, 0].axvspan(test_dt_start, test_dt_end, color='r', alpha=alpha, lw=lw)
    axs[0, 0].set_title('all data', fontsize=24)
    axs[0, 0].plot(all_dt.index, all_dt.loc[:, colname], color='k')
    axs[0, 0].plot(all_dt.index, all_dt.loc[:, "fcs"], color='b')
    axs[0, 0].grid(True, linestyle='--')
    axs[0, 0].tick_params(axis='both', labelsize=15)
    axs[0, 0].set_ylabel(colname, fontsize=15)
    
    axs[0, 1].set_facecolor("mintcream")
    axs[0, 1].set_title('train data', fontsize=24)
    axs[0, 1].plot(train_dt.index, train_dt.loc[train_dt.index, colname], color='k')
    axs[0, 1].plot(train_dt.index, train_dt.loc[train_dt.index, "fcs"], color='b')
    axs[0, 1].grid(True, linestyle='--')
    axs[0, 1].tick_params(axis='both', labelsize=15)
    axs[0, 1].set_ylabel(colname, fontsize=15)

    axs[1, 0].set_facecolor("mintcream")
    axs[1, 0].set_title('validation data', fontsize=24)
    axs[1, 0].plot(val_dt.index, val_dt.loc[val_dt.index, colname], color='k')
    axs[1, 0].plot(val_dt.index, val_dt.loc[val_dt.index, "fcs"], color='b')
    axs[1, 0].grid(True, linestyle='--')
    axs[1, 0].tick_params(axis='both', labelsize=15)
    axs[1, 0].set_ylabel(colname, fontsize=15)

    axs[1, 1].set_facecolor("mintcream")
    axs[1, 1].set_title('test data', fontsize=24)
    axs[1, 1].plot(test_dt.index, test_dt.loc[test_dt.index, colname], color='k')
    axs[1, 1].plot(test_dt.index, test_dt.loc[test_dt.index, "fcs"], color='b')
    axs[1, 1].grid(True, linestyle='--')
    axs[1, 1].tick_params(axis='both', labelsize=15)
    axs[1, 1].set_ylabel(colname, fontsize=15)
    
    for i in [0,1]: 
        for j in [0,1]: 
            for tick in axs[i,j].get_xticklabels(): 
                tick.set_rotation(20)
    fig.legend(["true", "forecast"], loc = 'upper right', prop={"size":30}, frameon=False)
    
    return


# In[7]:


colname='PCEPILFE'
h=4
train_dt_start=pd.to_datetime('01/01/1960', format='%m/%d/%Y')
train_dt_end=pd.to_datetime('12/31/2005', format='%m/%d/%Y')
val_dt_start=pd.to_datetime('01/01/2006', format='%m/%d/%Y')
val_dt_end=pd.to_datetime('06/30/2012', format='%m/%d/%Y')
test_dt_start=pd.to_datetime('07/01/2012', format='%m/%d/%Y')
test_dt_end=pd.to_datetime('12/31/2019', format='%m/%d/%Y')


# ## 2. Univariate forecasts
# 
# ### 2.1 Univariate statistics forecasts
# 
# #### 2.1.1 Last observation forecast

# In[8]:


def get_last_obs_fcs(data, colname, h,
                     train_dt_start, train_dt_end, 
                     val_dt_start, val_dt_end, 
                     test_dt_start, test_dt_end):
    
    xxx=data.loc[data.index, [colname]]
    xxx.loc[xxx.index, "fcs"]=xxx.loc[xxx.index, colname].shift(h)
    xxx.loc[xxx.index, "err"]=xxx.loc[xxx.index, colname]-xxx.loc[xxx.index, "fcs"]

    xxx_train=xxx.loc[train_dt_start:train_dt_end,:]
    xxx_val=xxx.loc[val_dt_start:val_dt_end,:]
    xxx_test=xxx.loc[test_dt_start:test_dt_end,:]
    
    rmse_all=np.sqrt(np.mean(xxx.loc[xxx.index, "err"]**2))
    rmse_train=np.sqrt(np.mean(xxx_train.loc[xxx_train.index, "err"]**2))
    rmse_val=np.sqrt(np.mean(xxx_val.loc[xxx_val.index, "err"]**2))
    rmse_test=np.sqrt(np.mean(xxx_test.loc[xxx_test.index, "err"]**2))
    
    return xxx, xxx_train, xxx_val, xxx_test, rmse_train, rmse_val, rmse_test, rmse_all


# In[9]:


last_obs_all, last_obs_train, last_obs_val, lat_obs_test, rmse_last_obs_train, rmse_last_obs_val, rmse_last_obs_test, rmse_last_obs_all = get_last_obs_fcs(data=all_data, colname=colname, h=h, 
                                                                            train_dt_start=train_dt_start, 
                                                                            train_dt_end=train_dt_end, 
                                                                            val_dt_start=val_dt_start, 
                                                                            val_dt_end=val_dt_end, 
                                                                            test_dt_start=test_dt_start,
                                                                            test_dt_end=test_dt_end)


forecast_plots(all_dt=last_obs_all, train_dt=last_obs_train, 
               val_dt=last_obs_val, test_dt=lat_obs_test, 
               colname=colname, fcs_name="Last obs", alpha=0.2, lw=0.5, 
               val_dt_start=val_dt_start, val_dt_end=val_dt_end, 
               test_dt_start=test_dt_start, test_dt_end=test_dt_end, 
               rmse_all=rmse_last_obs_all,
               rmse_train=rmse_last_obs_train, 
               rmse_val=rmse_last_obs_val, 
               rmse_test=rmse_last_obs_test)


# #### 2.1.2 Mean forecast

# In[10]:


def get_mean_fcs(data, colname, h, lag,
                 train_dt_start, train_dt_end, 
                 val_dt_start, val_dt_end, 
                 test_dt_start, test_dt_end):
    
    xxx=data.loc[data.index, [colname]]
    xxx.loc[xxx.index, "fcs"]=np.nan
    for i in range((lag+h), len(xxx.index)): xxx.iloc[i,1]=np.mean(xxx.iloc[(i-lag-h):(i-h),0])
    xxx.loc[xxx.index, "err"]=xxx.loc[xxx.index, colname]-xxx.loc[xxx.index, "fcs"]

    xxx_train=xxx.loc[train_dt_start:train_dt_end,:]
    xxx_val=xxx.loc[val_dt_start:val_dt_end,:]
    xxx_test=xxx.loc[test_dt_start:test_dt_end,:]
    
    rmse_all=np.sqrt(np.mean(xxx.loc[xxx.index, "err"]**2))
    rmse_train=np.sqrt(np.mean(xxx_train.loc[xxx_train.index, "err"]**2))
    rmse_val=np.sqrt(np.mean(xxx_val.loc[xxx_val.index, "err"]**2))
    rmse_test=np.sqrt(np.mean(xxx_test.loc[xxx_test.index, "err"]**2))
    
    return xxx, xxx_train, xxx_val, xxx_test, rmse_train, rmse_val, rmse_test, rmse_all


# In[11]:


def get_opt_mean_fcs(data, colname, h, max_lag,
                     train_dt_start, train_dt_end, 
                     val_dt_start, val_dt_end, 
                     test_dt_start, test_dt_end):
    
    all_lags=range(2,max_lag+1)
    all_rmse=[]
    
    for l in all_lags:
        tmp=get_mean_fcs(data=data, colname=colname, h=h, lag=l, 
                         train_dt_start=train_dt_start, train_dt_end=train_dt_end,
                         val_dt_start=val_dt_start, val_dt_end=val_dt_end, 
                         test_dt_start=test_dt_start, test_dt_end=test_dt_end)[5]
        all_rmse.append(tmp)
            
    return all_lags[all_rmse.index(min(all_rmse))]
    
opt_mean_fcs_l=get_opt_mean_fcs(data=all_data, colname=colname, h=h, max_lag=60,
                                train_dt_start=train_dt_start, train_dt_end=train_dt_end, 
                                val_dt_start=val_dt_start, val_dt_end=val_dt_end, 
                                test_dt_start=test_dt_start, test_dt_end=test_dt_end)


# In[12]:


mean_all, mean_train, mean_val, mean_test, rmse_mean_train, rmse_mean_val, rmse_mean_test, rmse_mean_all = get_mean_fcs(data=all_data, colname=colname, h=h, lag=opt_mean_fcs_l,
                                                            train_dt_start=train_dt_start, 
                                                            train_dt_end=train_dt_end, 
                                                            val_dt_start=val_dt_start, 
                                                            val_dt_end=val_dt_end, 
                                                            test_dt_start=test_dt_start,
                                                            test_dt_end=test_dt_end)


forecast_plots(all_dt=mean_all, train_dt=mean_train, 
               val_dt=mean_val, test_dt=mean_test, 
               colname=colname, fcs_name="Mean", alpha=0.2, lw=0.5, 
               val_dt_start=val_dt_start, val_dt_end=val_dt_end, 
               test_dt_start=test_dt_start, test_dt_end=test_dt_end,
               rmse_all=rmse_mean_all,
               rmse_train=rmse_mean_train, 
               rmse_val=rmse_mean_val, 
               rmse_test=rmse_mean_test)


# #### 2.1.3 Simple exponential smoothing forecast

# In[13]:


from statsmodels.tsa.holtwinters import SimpleExpSmoothing
import scipy.optimize as optimize


# In[14]:


def get_exp_smth_fcs(alpha, 
                     data, colname, h,
                     train_dt_start, train_dt_end, 
                     val_dt_start, val_dt_end, 
                     test_dt_start, test_dt_end):
    
    xxx=data.loc[data.index, [colname]]
    xxx.loc[xxx.index, "fcs"]=np.nan
    for i in range((2+h), len(xxx.index)):
        model=SimpleExpSmoothing(np.asarray(xxx.iloc[0:(i-h),0]))
        fit = model.fit(smoothing_level=alpha)
        xxx.iloc[i,1]=fit.forecast(4)[3]
    xxx.loc[xxx.index, "err"]=xxx.loc[xxx.index, colname]-xxx.loc[xxx.index, "fcs"]

    xxx_train=xxx.loc[train_dt_start:train_dt_end,:]
    xxx_val=xxx.loc[val_dt_start:val_dt_end,:]
    xxx_test=xxx.loc[test_dt_start:test_dt_end,:]
    
    rmse_all=np.sqrt(np.mean(xxx.loc[xxx.index, "err"]**2))
    rmse_train=np.sqrt(np.mean(xxx_train.loc[xxx_train.index, "err"]**2))
    rmse_val=np.sqrt(np.mean(xxx_val.loc[xxx_val.index, "err"]**2))
    rmse_test=np.sqrt(np.mean(xxx_test.loc[xxx_test.index, "err"]**2))
   
    return xxx, xxx_train, xxx_val, xxx_test, rmse_train, rmse_val, rmse_test, rmse_all


# In[15]:


def get_opt_exp_smth_fcs(data, colname, h, 
                         train_dt_start, train_dt_end, 
                         val_dt_start, val_dt_end, 
                         test_dt_start, test_dt_end):
    
    all_a=np.arange(0.01, 0.99, 0.1)
    all_rmse=[]
    
    for a in all_a:
        tmp=get_exp_smth_fcs(alpha=a, data=data, colname=colname, h=h,
                             train_dt_start=train_dt_start, train_dt_end=train_dt_end,
                             val_dt_start=val_dt_start, val_dt_end=val_dt_end, 
                             test_dt_start=test_dt_start, test_dt_end=test_dt_end)[5]
        all_rmse.append(tmp)
            
    return all_a[all_rmse.index(min(all_rmse))]
    
opt_exp_smth_fcs_a=get_opt_exp_smth_fcs(data=all_data, colname=colname, h=h,
                                        train_dt_start=train_dt_start, train_dt_end=train_dt_end, 
                                        val_dt_start=val_dt_start, val_dt_end=val_dt_end, 
                                        test_dt_start=test_dt_start, test_dt_end=test_dt_end)


# In[16]:


exp_smth_all, exp_smth_train, exp_smth_val, exp_smth_test, rmse_exp_smth_train, rmse_exp_smth_val, rmse_exp_smth_test, rmse_exp_smth_all = get_exp_smth_fcs(alpha=opt_exp_smth_fcs_a, data=all_data, colname=colname, h=h, 
                                     train_dt_start=train_dt_start, 
                                     train_dt_end=train_dt_end, 
                                     val_dt_start=val_dt_start, 
                                     val_dt_end=val_dt_end, 
                                     test_dt_start=test_dt_start,
                                     test_dt_end=test_dt_end)


forecast_plots(all_dt=exp_smth_all, train_dt=exp_smth_train, 
               val_dt=exp_smth_val, test_dt=exp_smth_test, 
               colname=colname, fcs_name="Simple Exp Smth", alpha=0.2, lw=0.5, 
               val_dt_start=val_dt_start, val_dt_end=val_dt_end, 
               test_dt_start=test_dt_start, test_dt_end=test_dt_end,
               rmse_all=rmse_exp_smth_all,
               rmse_train=rmse_exp_smth_train, 
               rmse_val=rmse_exp_smth_val, 
               rmse_test=rmse_exp_smth_test)


# #### 2.1.3 Holt forecast

# In[17]:


from statsmodels.tsa.holtwinters import Holt


# In[18]:


def get_holt_fcs(alpha, beta, 
                 data, colname, h,
                 train_dt_start, train_dt_end, 
                 val_dt_start, val_dt_end, 
                 test_dt_start, test_dt_end):
    
    xxx=data.loc[data.index, [colname]]
    xxx.loc[xxx.index, "fcs"]=np.nan
    for i in range((2+h), len(xxx.index)):
        model=Holt(np.asarray(xxx.iloc[0:(i-h),0]))
        fit = model.fit(smoothing_level=alpha, smoothing_trend=beta)
        xxx.iloc[i,1]=fit.forecast(4)[3]
    xxx.loc[xxx.index, "err"]=xxx.loc[xxx.index, colname]-xxx.loc[xxx.index, "fcs"]

    xxx_train=xxx.loc[train_dt_start:train_dt_end,:]
    xxx_val=xxx.loc[val_dt_start:val_dt_end,:]
    xxx_test=xxx.loc[test_dt_start:test_dt_end,:]
    
    rmse_all=np.sqrt(np.mean(xxx.loc[xxx.index, "err"]**2))
    rmse_train=np.sqrt(np.mean(xxx_train.loc[xxx_train.index, "err"]**2))
    rmse_val=np.sqrt(np.mean(xxx_val.loc[xxx_val.index, "err"]**2))
    rmse_test=np.sqrt(np.mean(xxx_test.loc[xxx_test.index, "err"]**2))
 
    return xxx, xxx_train, xxx_val, xxx_test, rmse_train, rmse_val, rmse_test, rmse_all


# In[19]:


def get_opt_holt_fcs(data, colname, h,
                     train_dt_start, train_dt_end, 
                     val_dt_start, val_dt_end, 
                     test_dt_start, test_dt_end):
    
    all_a=np.arange(0.01, 0.99, 0.1)
    all_b=np.arange(0.01, 0.99, 0.1)
    rmse, buf_a, buf_b = 1000, [], []
    
    for a in all_a:
        for b in all_b:
            tmp=get_holt_fcs(alpha=a, beta=b, data=data, colname=colname, h=h,
                             train_dt_start=train_dt_start, train_dt_end=train_dt_end,
                             val_dt_start=val_dt_start, val_dt_end=val_dt_end, 
                             test_dt_start=test_dt_start, test_dt_end=test_dt_end)[5]
            if tmp<rmse:
                rmse, buf_a, buf_b = tmp, a, b 
                        
    return buf_a, buf_b
    
opt_holt_fcs_a, opt_holt_fcs_b=get_opt_holt_fcs(data=all_data, colname=colname, h=h,
                                                train_dt_start=train_dt_start, train_dt_end=train_dt_end, 
                                                val_dt_start=val_dt_start, val_dt_end=val_dt_end, 
                                                test_dt_start=test_dt_start, test_dt_end=test_dt_end)


# In[20]:


holt_all, holt_train, holt_val, holt_test, rmse_holt_train, rmse_holt_val, rmse_holt_test, rmse_holt_all=get_holt_fcs(alpha=opt_holt_fcs_a, beta=opt_holt_fcs_b,
                                                          data=all_data, colname=colname, h=h,
                                                          train_dt_start=train_dt_start,
                                                          train_dt_end=train_dt_end,
                                                          val_dt_start=val_dt_start,
                                                          val_dt_end=val_dt_end,
                                                          test_dt_start=test_dt_start,
                                                          test_dt_end=test_dt_end)


forecast_plots(all_dt=holt_all, train_dt=holt_train, 
               val_dt=holt_val, test_dt=holt_test, 
               colname=colname, fcs_name="Holt", alpha=0.2, lw=0.5, 
               val_dt_start=val_dt_start, val_dt_end=val_dt_end, 
               test_dt_start=test_dt_start, test_dt_end=test_dt_end,
               rmse_all=rmse_holt_all,
               rmse_train=rmse_holt_train, 
               rmse_val=rmse_holt_val, 
               rmse_test=rmse_holt_test)


# #### 2.1.5 Damped exponential smoothing forecast

# In[21]:


from statsmodels.tsa.holtwinters import ExponentialSmoothing


# In[22]:


def get_exp_dmp_fcs(alpha, beta, phi, trend_type, 
                    data, colname, h,
                    train_dt_start, train_dt_end, 
                    val_dt_start, val_dt_end, 
                    test_dt_start, test_dt_end):
    
    xxx=data.loc[data.index, [colname]]
    xxx.loc[xxx.index, "fcs"]=np.nan
    for i in range((2+h), len(xxx.index)):
        model=ExponentialSmoothing(np.asarray(xxx.iloc[0:(i-h),0]), damped_trend=True, trend=trend_type)
        fit = model.fit(smoothing_level=alpha, smoothing_trend=beta, damping_trend=phi)
        xxx.iloc[i,1]=fit.forecast(4)[3]
    xxx.loc[xxx.index, "err"]=xxx.loc[xxx.index, colname]-xxx.loc[xxx.index, "fcs"]

    xxx_train=xxx.loc[train_dt_start:train_dt_end,:]
    xxx_val=xxx.loc[val_dt_start:val_dt_end,:]
    xxx_test=xxx.loc[test_dt_start:test_dt_end,:]
    
    rmse_all=np.sqrt(np.mean(xxx.loc[xxx.index, "err"]**2))
    rmse_train=np.sqrt(np.mean(xxx_train.loc[xxx_train.index, "err"]**2))
    rmse_val=np.sqrt(np.mean(xxx_val.loc[xxx_val.index, "err"]**2))
    rmse_test=np.sqrt(np.mean(xxx_test.loc[xxx_test.index, "err"]**2))
    
    return xxx, xxx_train, xxx_val, xxx_test, rmse_train, rmse_val, rmse_test, rmse_all


# In[23]:


def get_opt_exp_dmp_fcs(data, colname, h,
                        train_dt_start, train_dt_end, 
                        val_dt_start, val_dt_end, 
                        test_dt_start, test_dt_end):
    
    all_a=np.arange(0.001, 0.99, 0.2)
    all_b=np.arange(0.001, 0.99, 0.2)
    all_p=np.arange(0.001, 0.99, 0.2)
    all_tt=["add", "mul"]

    rmse=1000
    buf_a, buf_b, buf_p, buf_tt = [], [], [], []
    
    for a in all_a:
        for b in all_b:
            for p in all_p:
                for tt in all_tt:
                    tmp=get_exp_dmp_fcs(alpha=a, beta=b, phi=p, trend_type=tt,
                                        data=data, colname=colname, h=h,
                                        train_dt_start=train_dt_start, train_dt_end=train_dt_end,
                                        val_dt_start=val_dt_start, val_dt_end=val_dt_end, 
                                        test_dt_start=test_dt_start, test_dt_end=test_dt_end)[5]
                    if tmp<rmse:
                        rmse, buf_a, buf_b, buf_p, buf_tt = tmp, a, b, p, tt
                        
    return buf_a, buf_b, buf_p, buf_tt
    
opt_exp_dmp_fcs_a, opt_exp_dmp_fcs_b, opt_exp_dmp_fcs_p, opt_exp_dmp_fcs_tt = get_opt_exp_dmp_fcs(data=all_data, colname=colname, h=h,
                                                            train_dt_start=train_dt_start, train_dt_end=train_dt_end, 
                                                            val_dt_start=val_dt_start, val_dt_end=val_dt_end, 
                                                            test_dt_start=test_dt_start, test_dt_end=test_dt_end)


# In[24]:


exp_dmp_all, exp_dmp_train, exp_dmp_val, exp_dmp_test, rmse_exp_dmp_train, rmse_exp_dmp_val, rmse_exp_dmp_test, rmse_exp_dmp_all = get_exp_dmp_fcs(alpha=opt_exp_dmp_fcs_a, beta=opt_exp_dmp_fcs_b,
                                                                        phi=opt_exp_dmp_fcs_p, trend_type=opt_exp_dmp_fcs_tt,
                                                                        data=all_data, colname=colname, h=h,
                                                                        train_dt_start=train_dt_start,
                                                                        train_dt_end=train_dt_end,
                                                                        val_dt_start=val_dt_start,
                                                                        val_dt_end=val_dt_end,
                                                                        test_dt_start=test_dt_start,
                                                                        test_dt_end=test_dt_end)


forecast_plots(all_dt=exp_dmp_all, train_dt=exp_dmp_train, 
               val_dt=exp_dmp_val, test_dt=exp_dmp_test, 
               colname=colname, fcs_name="Damped exp smth", alpha=0.2, lw=0.5, 
               val_dt_start=val_dt_start, val_dt_end=val_dt_end, 
               test_dt_start=test_dt_start, test_dt_end=test_dt_end,
               rmse_all=rmse_exp_dmp_all, 
               rmse_train=rmse_exp_dmp_train, 
               rmse_val=rmse_exp_dmp_val, 
               rmse_test=rmse_exp_dmp_test)


# #### 2.1.6 ETS forecast

# In[25]:


def get_ets_fcs(alpha, beta, gamma, phi, mod_type, 
                data, colname, h,
                train_dt_start, train_dt_end, 
                val_dt_start, val_dt_end, 
                test_dt_start, test_dt_end):
    
    xxx=data.loc[data.index, [colname]]
    xxx.loc[xxx.index, "fcs"]=np.nan
    for i in range((10+h), len(xxx.index)):
        model=ExponentialSmoothing(np.asarray(xxx.iloc[0:(i-h),0]), damped_trend=True, 
                                   trend=mod_type, seasonal=mod_type, seasonal_periods=4)
        fit = model.fit(smoothing_level=alpha, smoothing_trend=beta, smoothing_seasonal=gamma, damping_trend=phi)
        xxx.iloc[i,1]=fit.forecast(4)[3]
    xxx.loc[xxx.index, "err"]=xxx.loc[xxx.index, colname]-xxx.loc[xxx.index, "fcs"]

    xxx_train=xxx.loc[train_dt_start:train_dt_end,:]
    xxx_val=xxx.loc[val_dt_start:val_dt_end,:]
    xxx_test=xxx.loc[test_dt_start:test_dt_end,:]
    
    rmse_all=np.sqrt(np.mean(xxx.loc[xxx.index, "err"]**2))
    rmse_train=np.sqrt(np.mean(xxx_train.loc[xxx_train.index, "err"]**2))
    rmse_val=np.sqrt(np.mean(xxx_val.loc[xxx_val.index, "err"]**2))
    rmse_test=np.sqrt(np.mean(xxx_test.loc[xxx_test.index, "err"]**2))
    
    return xxx, xxx_train, xxx_val, xxx_test, rmse_train, rmse_val, rmse_test, rmse_all


# In[26]:


def get_opt_ets_fcs(data, colname, h,
                    train_dt_start, train_dt_end, 
                    val_dt_start, val_dt_end, 
                    test_dt_start, test_dt_end):
    
    all_a=np.arange(0.001, 0.99, 0.3)
    all_b=np.arange(0.001, 0.99, 0.3)
    all_p=np.arange(0.001, 0.99, 0.3)
    all_g=np.arange(0.001, 0.99, 0.3)
    all_mt=["add", "mul"]
    
    rmse=1000
    buf_a, buf_b, buf_p, buf_g, buf_tt, buf_st = [], [], [], [], [], []

    for a in all_a:
        for b in all_b:
            for p in all_p:
                for g in all_g:
                    for mt in all_mt:
                        tmp=get_ets_fcs(alpha=a, beta=b, gamma=g, phi=p, mod_type=mt,
                                        data=data, colname=colname, h=h,
                                        train_dt_start=train_dt_start, train_dt_end=train_dt_end,
                                        val_dt_start=val_dt_start, val_dt_end=val_dt_end, 
                                        test_dt_start=test_dt_start, test_dt_end=test_dt_end)[5]
                        if tmp<rmse:
                            rmse, buf_a, buf_b, buf_p, buf_g, buf_mt = tmp, a, b, p, g, mt
                        
    return buf_a, buf_b, buf_p, buf_g, buf_mt
    
opt_ets_fcs_a, opt_ets_fcs_b, opt_ets_fcs_p, opt_ets_fcs_g, opt_ets_fcs_mt = get_opt_ets_fcs(data=all_data, colname=colname, h=h,
                                 train_dt_start=train_dt_start, train_dt_end=train_dt_end, 
                                 val_dt_start=val_dt_start, val_dt_end=val_dt_end, 
                                 test_dt_start=test_dt_start, test_dt_end=test_dt_end)


# In[27]:


ets_all, ets_train, ets_val, ets_test, rmse_ets_train, rmse_ets_val, rmse_ets_test, rmse_ets_all = get_ets_fcs(alpha=opt_ets_fcs_a, beta=opt_ets_fcs_b,
                                                        gamma=opt_ets_fcs_g, 
                                                        phi=opt_ets_fcs_p, mod_type=opt_ets_fcs_mt,
                                                        data=all_data, colname=colname, h=h,
                                                        train_dt_start=train_dt_start,
                                                        train_dt_end=train_dt_end,
                                                        val_dt_start=val_dt_start,
                                                        val_dt_end=val_dt_end,
                                                        test_dt_start=test_dt_start,
                                                        test_dt_end=test_dt_end)


forecast_plots(all_dt=ets_all, train_dt=ets_train, 
               val_dt=ets_val, test_dt=ets_test, 
               colname=colname, fcs_name="ETS", alpha=0.2, lw=0.5, 
               val_dt_start=val_dt_start, val_dt_end=val_dt_end, 
               test_dt_start=test_dt_start, test_dt_end=test_dt_end,
               rmse_all=rmse_ets_all,
               rmse_train=rmse_ets_train, 
               rmse_val=rmse_ets_val, 
               rmse_test=rmse_ets_test)


# #### 2.1.7 ARIMA forecast

# In[28]:


from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX


# In[29]:


plot_pacf(pd.Series.diff(all_data.loc[train_dt_start:train_dt_end, colname])[1:,], alpha=0.05, lags=50)
plot_acf(pd.Series.diff(all_data.loc[train_dt_start:train_dt_end, colname])[1:,], alpha=0.05, lags=50)


# In[30]:


def get_arima_fcs(pp, ii, qq, 
                  data, colname, h,
                  train_dt_start, train_dt_end, 
                  val_dt_start, val_dt_end, 
                  test_dt_start, test_dt_end):
    
    xxx=data.loc[data.index, [colname]]
    xxx.loc[xxx.index, "fcs"]=np.nan
    for i in range((20+h), len(xxx.index)):
        model=SARIMAX(np.asarray(xxx.iloc[0:(i-h),0]), order=(pp,ii,qq))
        fit = model.fit(disp=0)
        xxx.iloc[i,1]=fit.forecast(steps=4)[3]
    xxx.loc[xxx.index, "err"]=xxx.loc[xxx.index, colname]-xxx.loc[xxx.index, "fcs"]

    xxx_train=xxx.loc[train_dt_start:train_dt_end,:]
    xxx_val=xxx.loc[val_dt_start:val_dt_end,:]
    xxx_test=xxx.loc[test_dt_start:test_dt_end,:]
    
    rmse_all=np.sqrt(np.mean(xxx.loc[xxx.index, "err"]**2))
    rmse_train=np.sqrt(np.mean(xxx_train.loc[xxx_train.index, "err"]**2))
    rmse_val=np.sqrt(np.mean(xxx_val.loc[xxx_val.index, "err"]**2))
    rmse_test=np.sqrt(np.mean(xxx_test.loc[xxx_test.index, "err"]**2))
    
    return xxx, xxx_train, xxx_val, xxx_test, rmse_train, rmse_val, rmse_test, rmse_all


# In[31]:


def get_opt_arima_fcs(data, colname, h,
                      train_dt_start, train_dt_end, 
                      val_dt_start, val_dt_end, 
                      test_dt_start, test_dt_end):
    
    all_p=np.arange(1, 5, 1)
    all_i=[0, 1]
    all_q=np.arange(0, 5, 1)
    
    rmse=1000
    buf_p, buf_i, buf_q = [], [], []

    for pp in all_p:
        for ii in all_i:
            for qq in all_q:
                tmp=get_arima_fcs(pp=pp, ii=ii, qq=qq,
                                  data=data, colname=colname, h=h,
                                  train_dt_start=train_dt_start, train_dt_end=train_dt_end,
                                  val_dt_start=val_dt_start, val_dt_end=val_dt_end, 
                                  test_dt_start=test_dt_start, test_dt_end=test_dt_end)[5]
                #print(pp, ii, qq)
                if tmp<rmse:
                    rmse, buf_p, buf_i, buf_q = tmp, pp, ii, qq
                        
    return buf_p, buf_i, buf_q
    
opt_arima_p, opt_arima_i, opt_arima_q = get_opt_arima_fcs(data=all_data, colname=colname, h=h,
                                train_dt_start=train_dt_start, train_dt_end=train_dt_end, 
                                val_dt_start=val_dt_start, val_dt_end=val_dt_end, 
                                test_dt_start=test_dt_start, test_dt_end=test_dt_end)


# In[32]:


arima_all, arima_train, arima_val, arima_test, rmse_arima_train, rmse_arima_val, rmse_arima_test, rmse_arima_all = get_arima_fcs(pp=opt_arima_p, ii=opt_arima_i,
                                                                qq=opt_arima_q, 
                                                                data=all_data, colname=colname, h=h,
                                                                train_dt_start=train_dt_start,
                                                                train_dt_end=train_dt_end,
                                                                val_dt_start=val_dt_start,
                                                                val_dt_end=val_dt_end,
                                                                test_dt_start=test_dt_start,
                                                                test_dt_end=test_dt_end)


forecast_plots(all_dt=arima_all, train_dt=arima_train, 
               val_dt=arima_val, test_dt=arima_test, 
               colname=colname, fcs_name="ARIMA", alpha=0.2, lw=0.5, 
               val_dt_start=val_dt_start, val_dt_end=val_dt_end, 
               test_dt_start=test_dt_start, test_dt_end=test_dt_end,
               rmse_all=rmse_arima_all, 
               rmse_train=rmse_arima_train, 
               rmse_val=rmse_arima_val, 
               rmse_test=rmse_arima_test)


# ### 2.2 Univariate experts forecasts
# 
# #### 2.2.1 Mean experts forecast

# In[33]:


def make_date_from_y_q(y, q):
    if q==1:
        y=y-1
        m=12
    elif q==2:
        m=3
    elif q==3:
        m=6
    elif q==4:
        m=9
    date_to_do=pd.to_datetime(str(m)+'/01/'+str(y), format='%m/%d/%Y')
    return date_to_do


# In[34]:


def get_experts_forecast(data, colname, fcs_type, h, trans_flag,                     
                         train_dt_start, train_dt_end, 
                         val_dt_start, val_dt_end, 
                         test_dt_start, test_dt_end):
    
    url_base="https://www.philadelphiafed.org/-/media/research-and-data/real-time-center/survey-of-professional-forecasters/data-files/files/"
    url=url_base
    
    if colname=="GDPC1":
        if h==1: c_name="RGDP2"
        elif h==2: c_name="RGDP3"
        elif h==3: c_name="RGDP4"
        elif h==4: c_name="RGDP5"
        elif h==5: c_name="RGDP6"
        else: return "Error: No Data"
        transform=5
        if(fcs_type=="median"):
            url=url_base+"median_rgdp_level.xlsx"
        elif(fcs_type=="mean"):
            url=url_base+"mean_rgdp_level.xlsx"
            
    elif colname=="PCEPILFE":
        if h==1: c_name="COREPCE2"
        elif h==2: c_name="COREPCE3"
        elif h==3: c_name="COREPCE4"
        elif h==4: c_name="COREPCE5"
        elif h==5: c_name="COREPCE6"
        else: return "Error: No Data"
        transform=10
        if(fcs_type=="median"):
            url=url_base+"median_corepce_level.xlsx"
        elif(fcs_type=="mean"):
            url=url_base+"mean_corepce_level.xlsx"
            
    elif colname=="UNRATE":
        if h==1: c_name="UNEMP2"
        elif h==2: c_name="UNEMP3"
        elif h==3: c_name="UNEMP4"
        elif h==4: c_name="UNEMP5"
        elif h==5: c_name="UNEMP6"
        else: return "Error: No Data"
        transform=2
        if(fcs_type=="median"):
            url=url_base+"median_unemp_level.xlsx"
        elif(fcs_type=="mean"):
            url=url_base+"mean_unemp_level.xlsx"
                
    elif colname=="INDPRO":
        if h==1: c_name="INDPROD2"
        elif h==2: c_name="INDPROD3"
        elif h==3: c_name="INDPROD4"
        elif h==4: c_name="INDPROD5"
        elif h==5: c_name="INDPROD6"
        else: return "Error: No Data"
        transform=5
        if(fcs_type=="median"):
            url=url_base+"median_indprod_level.xlsx"
        elif(fcs_type=="mean"):
            url=url_base+"mean_indprod_level.xlsx"
                     
    elif colname=="TB3MS":
        if h==1: c_name="TBILL2"
        elif h==2: c_name="TBILL3"
        elif h==3: c_name="TBILL4"
        elif h==4: c_name="TBILL5"
        elif h==5: c_name="TBILL6"
        else: return "Error: No Data"
        transform=2
        if(fcs_type=="median"):
            url=url_base+"median_tbill_level.xlsx"
        elif(fcs_type=="mean"):
            url=url_base+"mean_tbill_level.xlsx"
                     
    elif colname=="GS10":
        if h==1: c_name="TBOND2"
        elif h==2: c_name="TBOND3"
        elif h==3: c_name="TBOND4"
        elif h==4: c_name="TBOND5"
        elif h==5: c_name="TBOND6"
        else: return "Error: No Data"
        transform=2
        if(fcs_type=="median"):
            url=url_base+"median_tbond_level.xlsx"
        elif(fcs_type=="mean"):
            url=url_base+"mean_tbond_level.xlsx"

    elif colname=="AAA":
        if h==1: c_name="BOND2"
        elif h==2: c_name="BOND3"
        elif h==3: c_name="BOND4"
        elif h==4: c_name="BOND5"
        elif h==5: c_name="BOND6"
        else: return "Error: No Data"
        transform=2
        if(fcs_type=="median"):
            url=url_base+"median_bond_level.xlsx"
        elif(fcs_type=="mean"):
            url=url_base+"mean_bond_level.xlsx"

    elif colname=="BAA":
        if h==1: c_name="BAABOND2"
        elif h==2: c_name="BAABOND3"
        elif h==3: c_name="BAABOND4"
        elif h==4: c_name="BAABOND5"
        elif h==5: c_name="BAABOND6"
        else: return "Error: No Data"
        transform=2
        if(fcs_type=="median"):
            url=url_base+"median_baabond_level.xlsx"
        elif(fcs_type=="mean"):
            url=url_base+"mean_baabond_level.xlsx"
            
    elif colname=="GS10TB3Mx":
        url=[]
        c_name="fcs"
        data_fcs_1=get_experts_forecast(data=data, colname="TB3MS", fcs_type=fcs_type, h=h, trans_flag=False,                      
                                        train_dt_start=train_dt_start, train_dt_end=train_dt_end, 
                                        val_dt_start=val_dt_start, val_dt_end=val_dt_end, 
                                        test_dt_start=test_dt_start, test_dt_end=test_dt_end)[0]
        data_fcs_2=get_experts_forecast(data=data, colname="GS10", fcs_type=fcs_type, h=h, trans_flag=False,                      
                                        train_dt_start=train_dt_start, train_dt_end=train_dt_end, 
                                        val_dt_start=val_dt_start, val_dt_end=val_dt_end, 
                                        test_dt_start=test_dt_start, test_dt_end=test_dt_end)[0]
        data_fcs=data_fcs_2[["fcs"]]-data_fcs_1[["fcs"]]
        data_fcs.columns=["fcs"]
    
    if url==url_base: return "Error: No Data"
    
    if colname!="GS10TB3Mx":
        data_xlsx=pd.read_excel(url)
        data_xlsx["sasdate"]=np.vectorize(make_date_from_y_q)(data_xlsx['YEAR'], data_xlsx['QUARTER'])
        data_fcs=data_xlsx[[c_name]]
        data_fcs.index=data_xlsx["sasdate"]
        if trans_flag: data_fcs.loc[data_fcs.index, [c_name]]=transform_data(x=data_fcs[c_name], flag=transform)
    
    xxx=data.loc[data.index, [colname]]
    xxx["fcs"]=data_fcs[c_name]
    xxx.loc[xxx.index, "err"]=xxx.loc[xxx.index, colname]-xxx.loc[xxx.index, "fcs"]

    xxx_train=xxx.loc[train_dt_start:train_dt_end,:]
    xxx_val=xxx.loc[val_dt_start:val_dt_end,:]
    xxx_test=xxx.loc[test_dt_start:test_dt_end,:]
    
    rmse_all=np.sqrt(np.mean(xxx.loc[xxx.index, "err"]**2))
    rmse_train=np.sqrt(np.mean(xxx_train.loc[xxx_train.index, "err"]**2))
    rmse_val=np.sqrt(np.mean(xxx_val.loc[xxx_val.index, "err"]**2))
    rmse_test=np.sqrt(np.mean(xxx_test.loc[xxx_test.index, "err"]**2))
    
    return xxx, xxx_train, xxx_val, xxx_test, rmse_train, rmse_val, rmse_test, rmse_all           
    


# In[35]:


experts_mean_all, experts_mean_train, experts_mean_val, experts_mean_test, rmse_experts_mean_train, rmse_experts_mean_val, rmse_experts_mean_test, rmse_experts_mean_all = get_experts_forecast(data=all_data, colname=colname, fcs_type="mean", h=4, trans_flag=True,                     
                                      train_dt_start=train_dt_start, train_dt_end=train_dt_end, 
                                      val_dt_start=val_dt_start, val_dt_end=val_dt_end, 
                                      test_dt_start=test_dt_start, test_dt_end=test_dt_end)


forecast_plots(all_dt=experts_mean_all, train_dt=experts_mean_train, 
               val_dt=experts_mean_val, test_dt=experts_mean_test, 
               colname=colname, fcs_name="Experts mean", alpha=0.2, lw=0.5, 
               val_dt_start=val_dt_start, val_dt_end=val_dt_end, 
               test_dt_start=test_dt_start, test_dt_end=test_dt_end,
               rmse_all=rmse_experts_mean_all, 
               rmse_train=rmse_experts_mean_train, 
               rmse_val=rmse_experts_mean_val, 
               rmse_test=rmse_experts_mean_test)


# #### 2.2.2 Median experts forecast

# In[36]:


experts_median_all, experts_median_train, experts_median_val, experts_median_test, rmse_experts_median_train, rmse_experts_median_val, rmse_experts_median_test, rmse_experts_median_all = get_experts_forecast(data=all_data, colname=colname, fcs_type="median", h=4, trans_flag=True,                     
                                               train_dt_start=train_dt_start, train_dt_end=train_dt_end, 
                                               val_dt_start=val_dt_start, val_dt_end=val_dt_end, 
                                               test_dt_start=test_dt_start, test_dt_end=test_dt_end)


forecast_plots(all_dt=experts_median_all, train_dt=experts_median_train, 
               val_dt=experts_median_val, test_dt=experts_median_test, 
               colname=colname, fcs_name="Experts median", alpha=0.2, lw=0.5, 
               val_dt_start=val_dt_start, val_dt_end=val_dt_end, 
               test_dt_start=test_dt_start, test_dt_end=test_dt_end,
               rmse_all=rmse_experts_median_all, 
               rmse_train=rmse_experts_median_train, 
               rmse_val=rmse_experts_median_val, 
               rmse_test=rmse_experts_median_test)


# ### 2.3 Univariate ML forecasts
# 
# #### 2.3.1 Single-layer Perceptron

# In[37]:


from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras import regularizers 
import keras.backend as K
from sklearn.preprocessing import MinMaxScaler
tf.keras.backend.set_floatx('float64')
from keras.callbacks import EarlyStopping
from random import sample
from keras.utils import Sequence


# In[38]:


def root_mean_squared_error(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true)))


# In[39]:


def w_init(a_f):
    if a_f == "relu":
        k_i = "he_uniform"
    elif a_f == "tanh":
        k_i = "glorot_uniform"
    return k_i


# In[40]:


def series_to_supervised(data, h, n_in, n_out):
    
    cols = list()
    for ll in range(n_in+h-1, h-1, -1):
        cols.append(data.shift(ll))
    for ll in range(n_out-1, -1,-1):
        cols.append(data.shift(ll))
    agg = pd.concat(cols, axis = 1)
    agg.dropna(inplace = True)
    
    return agg


# In[41]:


def scaling(data):
    
    min_dt = data.min(axis = 0)
    max_dt = data.max(axis = 0)
    
    return min_dt, max_dt, ((data-min_dt)/(max_dt-min_dt))

def rescale(x, min_dt, max_dt):
    
    x = x*(max_dt-min_dt)+min_dt
    
    return x


# In[42]:


def sl_ff_model(n_in, n_out, 
                n_nodes, a_f, l1, l2, 
                n_epoches, n_batch, 
                train_x, train_y, val_x, val_y):

    model = Sequential()
    model.add(Dense(n_nodes, activation = a_f, input_dim = n_in,
               kernel_initializer = w_init(a_f), bias_initializer = "zeros",
               kernel_regularizer = regularizers.l1_l2(l1 = l1, l2 = l2), 
               bias_regularizer = regularizers.l1_l2(l1 = l1, l2 = l2),
               activity_regularizer = regularizers.l1_l2(l1 = l1, l2 = l2),
               dtype="float64"))
    model.add(Dropout(0.5))
    model.add(BatchNormalization(trainable=True))
    model.add(Dense(n_out))

    model.compile(loss = root_mean_squared_error, optimizer = "sgd", metrics = [root_mean_squared_error])

    res=model.fit(train_x, train_y, epochs = n_epoches, batch_size = n_batch, verbose = 0,
              validation_data = (val_x, val_y), use_multiprocessing = True, workers = 8,
              callbacks=EarlyStopping(monitor = 'val_loss', patience = 1000, verbose = 0,
                                      mode = 'auto', restore_best_weights = True))

    return model, res


# In[45]:


def get_s_l_nn_forecast(data, colname, h, 
                        n_in, n_out, n_nodes, n_epoches, 
                        n_batch, a_f, l1, l2, n_times,
                        train_dt_start, train_dt_end, 
                        val_dt_start, val_dt_end, 
                        test_dt_start, test_dt_end):
    
    pre_data = data.loc[data.index, [colname]]
    min_dt, max_dt, pre_data.loc[pre_data.index, [colname]] = scaling(pre_data.loc[pre_data.index, [colname]])
    tensor_all = series_to_supervised(data = pre_data, h = h, n_in = n_in, n_out = n_out)

    tensor_train = tensor_all.loc[train_dt_start:train_dt_end,:]
    tensor_val = tensor_all.loc[val_dt_start:val_dt_end,:]
    tensor_test = tensor_all.loc[test_dt_start:test_dt_end,:]

    all_x, all_y = tensor_all.iloc[:, :-n_out], tensor_all.iloc[:, -n_out:]
    train_x, train_y = tensor_train.iloc[:, :-n_out], tensor_train.iloc[:, -n_out:]
    val_x, val_y = tensor_val.iloc[:, :-n_out], tensor_val.iloc[:, -n_out:]
    test_x, test_y = tensor_test.iloc[:, :-n_out], tensor_test.iloc[:, -n_out:]
    
    yyy = data.loc[data.index, [colname]]
    yyy.loc[yyy.index, "fcs"] = np.nan
    
    for i_time in range(n_times):
        
        yyy.loc[yyy.index, "fcs_" + str(i_time + 1)] = np.nan
        model, res = sl_ff_model(n_in = n_in, n_out = n_out, 
                                 n_nodes = n_nodes, a_f = a_f, l1 = l1, l2 = l2, 
                                 n_epoches = n_epoches, n_batch = n_batch, 
                                 train_x = train_x, train_y = train_y, val_x = val_x, val_y = val_y)
        for ind in all_x.index:
            yyy.loc[ind,"fcs_" + str(i_time + 1)] = np.array(rescale(x = model.predict(all_x.loc[ind:ind,:], verbose = 0)[0][n_out-1],
                                         min_dt=min_dt, max_dt=max_dt))
    
    yyy.loc[yyy.index, "fcs"] = yyy[["fcs_" + str(i_time + 1) for i_time in range(n_times)]].sum(axis=1)/n_times
    
    xxx = data.loc[data.index, [colname]]
    xxx.loc[xxx.index, "fcs"] = yyy.loc[yyy.index, "fcs"] 
    xxx.loc[xxx.index, "err"] = xxx.loc[xxx.index, colname]-xxx.loc[xxx.index, "fcs"]
    
    xxx_train = xxx.loc[train_dt_start:train_dt_end,:]
    xxx_val = xxx.loc[val_dt_start:val_dt_end,:]
    xxx_test = xxx.loc[test_dt_start:test_dt_end,:]
    
    rmse_all = np.sqrt(np.mean(xxx.loc[xxx.index, "err"]**2))
    rmse_train = np.sqrt(np.mean(xxx_train.loc[xxx_train.index, "err"]**2))
    rmse_val = np.sqrt(np.mean(xxx_val.loc[xxx_val.index, "err"]**2))
    rmse_test = np.sqrt(np.mean(xxx_test.loc[xxx_test.index, "err"]**2))
    
    return xxx, xxx_train, xxx_val, xxx_test, rmse_train, rmse_val, rmse_test, rmse_all


# In[47]:


def get_opt_s_l_nn_fcs(data, colname, h, 
                       train_dt_start, train_dt_end, 
                       val_dt_start, val_dt_end, 
                       test_dt_start, test_dt_end):
    
    all_n_in = [60]
    all_n_out = [1, 4]
    all_n_nodes = [4, 20, 40, 60, 120]
    all_n_epoches = [5000000]
    all_n_batch = [32]
    all_a_f = ["relu"]
    all_l1 = [0.001]
    all_l2 = [0.01]
    n_times = 10
    
    rmse=1000
    buf_n_in, buf_n_out, buf_n_nodes, buf_n_epoches,     buf_n_batch, buf_a_f, buf_l1, buf_l2, buf_rmse = [], [], [], [], [], [], [], [], []

    for i_1 in all_n_in:
        for i_2 in all_n_out:
            for i_3 in all_n_nodes:
                for i_4 in all_n_epoches:
                    for i_5 in all_n_batch:
                        for i_6 in all_a_f:
                            for i_7 in all_l1:
                                for i_8 in all_l2:
                                    tmp=get_s_l_nn_forecast(data = data, colname = colname, h = h,
                                                            n_in = i_1, n_out = i_2, n_nodes = i_3, n_epoches = i_4, 
                                                            n_batch = i_5, a_f = i_6, l1 = i_7, l2 = i_8, n_times = n_times,
                                                            train_dt_start = train_dt_start, train_dt_end = train_dt_end,
                                                            val_dt_start = val_dt_start, val_dt_end = val_dt_end, 
                                                            test_dt_start = test_dt_start, test_dt_end = test_dt_end)[5]
                                    print(i_1, i_2, i_3, i_4, i_5, i_6, i_7, i_8, tmp)
                                    
                                    if tmp<rmse:
                                            buf_n_in, buf_n_out, buf_n_nodes, buf_n_epoches,                                             buf_n_batch, buf_a_f, buf_l1, buf_l2 = i_1, i_2, i_3, i_4, i_5, i_6, i_7, i_8
                        
    return buf_n_in, buf_n_out, buf_n_nodes, buf_n_epoches, buf_n_batch, buf_a_f, buf_l1, buf_l2
    
opt_s_l_nn_n_in, opt_s_l_nn_n_out, opt_s_l_nn_n_nodes, opt_s_l_nn_n_epoches, opt_s_l_nn_n_batch, opt_s_l_nn_a_f, opt_s_l_nn_l1, opt_s_l_nn_l2 = get_opt_s_l_nn_fcs(data = all_data, colname = colname, h = h,
                                                  train_dt_start = train_dt_start, train_dt_end = train_dt_end, 
                                                  val_dt_start = val_dt_start, val_dt_end = val_dt_end, 
                                                  test_dt_start = test_dt_start, test_dt_end = test_dt_end)


# In[48]:


s_l_nn_all, s_l_nn_train, s_l_nn_val, s_l_nn_test, rmse_s_l_nn_train, rmse_s_l_nn_val, rmse_s_l_nn_test, rmse_s_l_nn_all = get_s_l_nn_forecast(data = all_data, colname = colname, h = h,
                                      n_in = opt_s_l_nn_n_in, n_out = opt_s_l_nn_n_out,
                                      n_nodes = opt_s_l_nn_n_nodes, n_epoches = opt_s_l_nn_n_epoches, 
                                      n_batch = opt_s_l_nn_n_batch, a_f = opt_s_l_nn_a_f,
                                      l1 = opt_s_l_nn_l1, l2 = opt_s_l_nn_l2, n_times=10,
                                      train_dt_start = train_dt_start, train_dt_end = train_dt_end,
                                      val_dt_start = val_dt_start, val_dt_end = val_dt_end, 
                                      test_dt_start = test_dt_start, test_dt_end = test_dt_end)


forecast_plots(all_dt=s_l_nn_all, train_dt=s_l_nn_train, 
               val_dt=s_l_nn_val, test_dt=s_l_nn_test, 
               colname=colname, fcs_name="Single Layer NN", alpha=0.2, lw=0.5, 
               val_dt_start=val_dt_start, val_dt_end=val_dt_end, 
               test_dt_start=test_dt_start, test_dt_end=test_dt_end,
               rmse_all=rmse_s_l_nn_all, 
               rmse_train=rmse_s_l_nn_train, 
               rmse_val=rmse_s_l_nn_val, 
               rmse_test=rmse_s_l_nn_test)


# ### 2.3 Univariate ML forecasts
# 
# #### 2.3.1 Multi-layer Perceptron

# In[49]:


def ml_ff_model(n_in, n_out, 
                n_nodes_1, n_nodes_2, a_f_1, a_f_2, l1, l2, 
                n_epoches, n_batch, 
                train_x, train_y, val_x, val_y):

    model = Sequential()
    model.add(Dense(n_nodes_1, activation = a_f_1, input_dim = n_in,
               kernel_initializer = w_init(a_f_1), bias_initializer = "zeros",
               kernel_regularizer = regularizers.l1_l2(l1 = l1, l2 = l2), 
               bias_regularizer = regularizers.l1_l2(l1 = l1, l2 = l2),
               activity_regularizer = regularizers.l1_l2(l1 = l1, l2 = l2),
               dtype="float64"))
    model.add(Dropout(0.5))
    model.add(BatchNormalization(trainable=True))
    model.add(Dense(n_nodes_2, activation = a_f_2, input_dim = n_nodes_1,
               kernel_initializer = w_init(a_f_2), bias_initializer = "zeros",
               kernel_regularizer = regularizers.l1_l2(l1 = l1, l2 = l2), 
               bias_regularizer = regularizers.l1_l2(l1 = l1, l2 = l2),
               activity_regularizer = regularizers.l1_l2(l1 = l1, l2 = l2),
               dtype="float64"))
    model.add(Dropout(0.5))
    model.add(BatchNormalization(trainable=True))
    model.add(Dense(n_out))

    model.compile(loss = root_mean_squared_error, optimizer = "sgd", metrics = [root_mean_squared_error])

    res=model.fit(train_x, train_y, epochs = n_epoches, batch_size = n_batch, verbose = 0,
              validation_data = (val_x, val_y), use_multiprocessing = True, workers = 8,
              callbacks=EarlyStopping(monitor = 'val_loss', patience = 2000, verbose = 0,
                                      mode = 'auto', restore_best_weights = True))

    return model, res


# In[50]:


def get_m_l_nn_forecast(data, colname, h, 
                        n_in, n_out, n_nodes_1, n_nodes_2, n_epoches, 
                        n_batch, a_f_1, a_f_2, l1, l2, n_times,
                        train_dt_start, train_dt_end, 
                        val_dt_start, val_dt_end, 
                        test_dt_start, test_dt_end):
    
    pre_data = data.loc[data.index, [colname]]
    min_dt, max_dt, pre_data.loc[pre_data.index, [colname]] = scaling(pre_data.loc[pre_data.index, [colname]])
    tensor_all = series_to_supervised(data = pre_data, h = h, n_in = n_in, n_out = n_out)

    tensor_train = tensor_all.loc[train_dt_start:train_dt_end,:]
    tensor_val = tensor_all.loc[val_dt_start:val_dt_end,:]
    tensor_test = tensor_all.loc[test_dt_start:test_dt_end,:]

    all_x, all_y = tensor_all.iloc[:, :-n_out], tensor_all.iloc[:, -n_out:]
    train_x, train_y = tensor_train.iloc[:, :-n_out], tensor_train.iloc[:, -n_out:]
    val_x, val_y = tensor_val.iloc[:, :-n_out], tensor_val.iloc[:, -n_out:]
    test_x, test_y = tensor_test.iloc[:, :-n_out], tensor_test.iloc[:, -n_out:]
    
    yyy = data.loc[data.index, [colname]]
    yyy.loc[yyy.index, "fcs"] = np.nan
    
    for i_time in range(n_times):
        
        yyy.loc[yyy.index, "fcs_" + str(i_time + 1)] = np.nan
        model, res = ml_ff_model(n_in = n_in, n_out = n_out, 
                                 n_nodes_1 = n_nodes_1, n_nodes_2 = n_nodes_2, 
                                 a_f_1 = a_f_1, a_f_2 = a_f_2, l1 = l1, l2 = l2, 
                                 n_epoches = n_epoches, n_batch = n_batch, 
                                 train_x = train_x, train_y = train_y, val_x = val_x, val_y = val_y)
        for ind in all_x.index:
            yyy.loc[ind,"fcs_" + str(i_time + 1)] = np.array(rescale(x = model.predict(all_x.loc[ind:ind,:], verbose = 0)[0][n_out-1],
                                         min_dt=min_dt, max_dt=max_dt))
    
    yyy.loc[yyy.index, "fcs"] = yyy[["fcs_" + str(i_time + 1) for i_time in range(n_times)]].sum(axis=1)/n_times
    
    xxx = data.loc[data.index, [colname]]
    xxx.loc[xxx.index, "fcs"] = yyy.loc[yyy.index, "fcs"] 
    xxx.loc[xxx.index, "err"] = xxx.loc[xxx.index, colname]-xxx.loc[xxx.index, "fcs"]
    
    xxx_train = xxx.loc[train_dt_start:train_dt_end,:]
    xxx_val = xxx.loc[val_dt_start:val_dt_end,:]
    xxx_test = xxx.loc[test_dt_start:test_dt_end,:]
    
    rmse_all = np.sqrt(np.mean(xxx.loc[xxx.index, "err"]**2))
    rmse_train = np.sqrt(np.mean(xxx_train.loc[xxx_train.index, "err"]**2))
    rmse_val = np.sqrt(np.mean(xxx_val.loc[xxx_val.index, "err"]**2))
    rmse_test = np.sqrt(np.mean(xxx_test.loc[xxx_test.index, "err"]**2))
    
    return xxx, xxx_train, xxx_val, xxx_test, rmse_train, rmse_val, rmse_test, rmse_all


# In[51]:


def get_opt_m_l_nn_fcs(data, colname, h, 
                       train_dt_start, train_dt_end, 
                       val_dt_start, val_dt_end, 
                       test_dt_start, test_dt_end):
    
    all_n_in = [60]
    all_n_out = [1, 4]
    all_n_nodes_1 = [4, 20, 40, 120]
    all_n_nodes_2 = [4, 20, 40, 120]
    all_n_epoches = [5000000]
    all_n_batch = [32]
    all_a_f_1 = ["relu"]
    all_a_f_2 = ["relu"]
    all_l1 = [0.001]
    all_l2 = [0.01]
    n_times = 10
    
    rmse=1000
    buf_n_in, buf_n_out, buf_n_nodes_1, buf_n_nodes_2, buf_n_epoches,     buf_n_batch, buf_a_f_1, buf_a_f_2, buf_l1, buf_l2, buf_rmse = [], [], [], [], [], [], [], [], [], [], []

    for i_1 in all_n_in:
        for i_2 in all_n_out:
            for i_3 in all_n_nodes_1:
                for i_4 in all_n_nodes_2:
                    for i_5 in all_n_epoches:
                        for i_6 in all_n_batch:
                            for i_7 in all_a_f_1:
                                for i_8 in all_a_f_2:
                                    for i_9 in all_l1:
                                        for i_10 in all_l2:
                                            tmp=get_m_l_nn_forecast(data = data, colname = colname, h = h,
                                                                    n_in = i_1, n_out = i_2, n_nodes_1 = i_3, n_nodes_2 = i_4,
                                                                    n_epoches = i_5, n_batch = i_6, a_f_1 = i_7, a_f_2 = i_8,
                                                                    l1 = i_9, l2 = i_9, n_times = n_times,
                                                                    train_dt_start = train_dt_start, train_dt_end = train_dt_end,
                                                                    val_dt_start = val_dt_start, val_dt_end = val_dt_end, 
                                                                    test_dt_start = test_dt_start, test_dt_end = test_dt_end)[5]
                                            print(i_1, i_2, i_3, i_4, i_5, i_6, i_7, i_8, i_9, i_10, tmp)

                                            if tmp<rmse:
                                                buf_n_in, buf_n_out, buf_n_nodes_1, buf_n_nodes_2, buf_n_epoches,                                                 buf_n_batch, buf_a_f_1, buf_a_f_2,                                                 buf_l1, buf_l2 = i_1, i_2, i_3, i_4, i_5, i_6, i_7, i_8, i_9, i_10

    return buf_n_in, buf_n_out, buf_n_nodes_1, buf_n_nodes_2, buf_n_epoches, buf_n_batch, buf_a_f_1, buf_a_f_2, buf_l1, buf_l2
    
opt_m_l_nn_n_in, opt_m_l_nn_n_out, opt_m_l_nn_n_nodes_1, opt_m_l_nn_n_nodes_2, opt_m_l_nn_n_epoches, opt_m_l_nn_n_batch, opt_m_l_nn_a_f_1, opt_m_l_nn_a_f_2, opt_m_l_nn_l1, opt_m_l_nn_l2 = get_opt_m_l_nn_fcs(data = all_data, colname = colname, h = h,
                                                  train_dt_start = train_dt_start, train_dt_end = train_dt_end, 
                                                  val_dt_start = val_dt_start, val_dt_end = val_dt_end, 
                                                  test_dt_start = test_dt_start, test_dt_end = test_dt_end)


# #### To be continued...

