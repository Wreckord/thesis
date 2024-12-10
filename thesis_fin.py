import polars as pl
import numpy as np
from scipy import stats
import statsmodels.regression.mixed_linear_model as sm
import statsmodels.api as sp
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold


#Funtion for statistical analysis
def statistical_analysis(metric,merge,mix_merge):
    n_splits=5
    group_kfold=GroupKFold(n_splits=n_splits)
    colsize=merge.height
    model_gds = sm.MixedLM.from_formula("GDTOTAL ~ 1", groups="RID", data=mix_merge)
    result_gds = model_gds.fit()
    resid_gds=pl.from_pandas(result_gds.resid).to_numpy()
    #ANALYSIS
    print("---------------"+metric+"---------------")
    # Correlation (Pearson & Spearman)
    prarray=[]
    speararray=[]
    for i in range(colsize):
        prarray.append(stats.pearsonr(np.array(merge['GDTOTAL'].to_list()[i]),np.array(merge[metric].to_list()[i]))[0])
        speararray.append(stats.spearmanr(np.array(merge['GDTOTAL'].to_list()[i]),np.array(merge[metric].to_list()[i]))[0])
    #Confidence Interval (Pearson)
    prarray=np.array([pr for pr in prarray if np.isnan(pr)==False])
    mean=np.mean(prarray)
    dev=np.std(prarray,ddof=1)
    n=len(prarray)
    print(n)
    conf=0.95
    t_crit=stats.t.ppf((1+conf)/2,df=n-1)
    err=dev/np.sqrt(n)
    margin=t_crit*err
    ci=(mean-margin,mean+margin)
    t_stat,p_value=stats.ttest_1samp(prarray,0)
    print('The mean, confidence interval for a='+str(1-conf)+' and the p-value for Pearson-r=0 for GDS vs '+metric+' are:')
    print(mean)
    print(ci)
    print(p_value)
    #Confidence Interval (Spearman)
    speararray=np.array([sp for sp in speararray if np.isnan(sp)==False])
    mean=np.mean(speararray)
    dev=np.std(speararray,ddof=1)
    n=len(speararray)
    print(n)
    conf=0.95
    t_crit=stats.t.ppf((1+conf)/2,df=n-1)
    err=dev/np.sqrt(n)
    margin=t_crit*err
    ci=(mean-margin,mean+margin)
    t_stat,p_value=stats.ttest_1samp(speararray,0)
    print('The mean, confidence interval for a='+str(1-conf)+' and the p-value for Spearman-r=0 for GDS vs '+metric+' are:')
    print(mean)
    print(ci)
    print(p_value)
    #Correlation through MixedLM Residuals
    formula=metric+" ~ 1"
    model_metric = sm.MixedLM.from_formula(formula, groups="RID", data=mix_merge)
    result_metric = model_metric.fit()
    resid_metric=pl.from_pandas(result_metric.resid).to_numpy()
    correlation=pl.DataFrame({"GDS Residuals":resid_gds,"Metric Residuals":resid_metric}).corr()['GDS Residuals'][1]
    print(f"Pearson Correlation between GDS and {metric} through residuals of Mixed Effect Model: {correlation}")
    #Mixed LM 
    formula=metric+" ~ GDTOTAL+PTGENDER+AGE_bl+TAU+ABETA"
    model=sm.MixedLM.from_formula(formula,groups='RID',data=mix_merge)
    result=model.fit()
    print(result.summary())
    #Calculating Bias for MixedLM
    predictions=result.predict()
    bias=np.mean(np.array(predictions)-mix_merge[metric].to_numpy())
    print(f"The bias of the MixedLM for {metric} is: {bias}")
    #Plotting Residual QQ-Plot for MixedLM
    residuals=result.resid
    residuals=residuals/np.std(residuals)   
    expected_values = np.sort(stats.norm.ppf(np.linspace(0.01, 0.99, len(residuals)))) 
    observed_values = np.sort(residuals)
    n_simulations = 1000
    simulated_bands = np.zeros((n_simulations, len(residuals)))
    for i in range(n_simulations):
        simulated_residuals = np.random.normal(0, 1, len(residuals))
        simulated_bands[i, :] = np.sort(simulated_residuals)
    lower_band = np.percentile(simulated_bands, 2.5, axis=0)
    upper_band = np.percentile(simulated_bands, 97.5, axis=0)  
    out_of_bounds=(observed_values < lower_band) | (observed_values > upper_band)
    full=len(np.array(observed_values))
    invalid=len(np.where(out_of_bounds)[0])
    percentage=100*invalid/full  
    print(f"The percentage of the out of bounds values is {percentage}")
    plt.figure(figsize=(8, 6))
    plt.scatter(expected_values, observed_values, alpha=0.7)
    plt.plot(expected_values, expected_values, color='red', linestyle='--')
    plt.fill_between(expected_values, lower_band, upper_band, color='blue', alpha=0.2)
    plt.title("MixedLM QQ-Plot for "+metric)
    plt.xlabel("Expexted Values")
    plt.ylabel("Observed Values")
    plt.savefig(metric+"_qqplot_mixedlm_biomarkers"+'.png')
    plt.show()
    #Plotting residuals against predicted for MixedLM
    t_val=stats.t.ppf(1-(1-0.95)/2,df=len(residuals)-1)
    lower_bound = np.mean(residuals) - t_val * np.std(residuals)
    upper_bound = np.mean(residuals) + t_val * np.std(residuals)
    full=len(np.array(residuals))
    invalid=len(residuals[np.where((residuals<lower_bound)|(residuals>upper_bound))[0]])
    percentage=100*invalid/full
    print(f"The percentage of the out of bounds values is {percentage}")
    plt.figure(figsize=(8,6))
    plt.scatter(predictions,residuals)
    plt.axhline(lower_bound,color='blue',linestyle='--')
    plt.axhline(0, color='red', linestyle='--')
    plt.axhline(upper_bound,color='blue',linestyle='--')
    plt.title('Residuals vs Fitted Values (MixedLM) for '+metric)
    plt.xlabel('Fitted Values')
    plt.ylabel('Residuals')
    plt.savefig(metric+"_residuals_predicted_mixedlm_biomarkers"+'.png')
    plt.show()
    #Check for overfitting/underfitting for MixedLM
    mse_list = []
    X=mix_merge.drop('RID','M','MMSE','MOCA','ADAS13','DX')
    for train_index, test_index in group_kfold.split(X, mix_merge[metric],mix_merge['RID']):
        groups_train=mix_merge['RID'][train_index]
        model=sm.MixedLM.from_formula(formula,groups=groups_train,data=mix_merge[train_index])
        result=model.fit()
        y_pred=result.predict(mix_merge[test_index])
        y_test=mix_merge[metric][test_index]
        mse=mean_squared_error(y_test,y_pred)
        mse_list.append(mse)
    avmse=np.mean(mse_list)
    print(f'Average MSE across {n_splits} folds: {avmse}')
    #GEE
    formula=metric+" ~ GDTOTAL+PTGENDER+AGE_bl+TAU+ABETA"
    model=sp.GEE.from_formula(formula, groups='RID', data=mix_merge, cov_struct=sp.cov_struct.Exchangeable())
    result=model.fit()
    print(result.summary())
    #Calculating Bias for GEE 
    predictions = result.predict()
    bias=np.mean(np.array(predictions)-mix_merge[metric].to_numpy())
    print(f"The bias of the GEE for {metric} is: {bias}")
    #Plotting Residual QQ-Plot for GEE
    residuals=result.resid_response
    residuals=residuals/np.std(residuals)   
    expected_values = np.sort(stats.norm.ppf(np.linspace(0.01, 0.99, len(residuals)))) 
    observed_values = np.sort(residuals)
    n_simulations = 1000
    simulated_bands = np.zeros((n_simulations, len(residuals)))
    for i in range(n_simulations):
        simulated_residuals = np.random.normal(0, 1, len(residuals))
        simulated_bands[i, :] = np.sort(simulated_residuals)
    lower_band = np.percentile(simulated_bands, 2.5, axis=0)
    upper_band = np.percentile(simulated_bands, 97.5, axis=0)
    out_of_bounds=(observed_values < lower_band) | (observed_values > upper_band)
    full=len(np.array(observed_values))
    invalid=len(np.where(out_of_bounds)[0])
    percentage=100*invalid/full  
    print(f"The percentage of the out of bounds values is {percentage}")     
    plt.figure(figsize=(8, 6))
    plt.scatter(expected_values, observed_values, alpha=0.7)
    plt.plot(expected_values, expected_values, color='red', linestyle='--')
    plt.fill_between(expected_values, lower_band, upper_band, color='blue', alpha=0.2)
    plt.title("GEE QQ-Plot for "+metric)
    plt.xlabel("Expexted Values")
    plt.ylabel("Observed Values")
    plt.savefig(metric+"_qqplot_gee_biomarkers"+'.png')
    plt.show()
    #Plotting residuals against predicted for GEE
    t_val=stats.t.ppf(1-(1-0.95)/2,df=len(residuals)-1)
    lower_bound = np.mean(residuals) - t_val * np.std(residuals)
    upper_bound = np.mean(residuals) + t_val * np.std(residuals)
    full=len(np.array(residuals))
    invalid=len(residuals[np.where((residuals<lower_bound)|(residuals>upper_bound))[0]])
    percentage=100*invalid/full
    print(f"The percentage of the out of bounds values is {percentage}")
    plt.figure(figsize=(8,6))
    plt.scatter(predictions,residuals)
    plt.axhline(lower_bound,color='blue',linestyle='--')
    plt.axhline(0, color='red', linestyle='--')
    plt.axhline(upper_bound,color='blue',linestyle='--')
    plt.title('Residuals vs Fitted Values (GEE) for '+metric)
    plt.xlabel('Fitted Values')
    plt.ylabel('Residuals')
    plt.savefig(metric+"_residuals_predicted_gee_biomarkers"+'.png')
    plt.show()
    #Check for overfitting/underfitting for GEE
    mse_list = []
    X=mix_merge.drop('RID','M','MMSE','MOCA','ADAS13','DX')
    for train_index, test_index in group_kfold.split(X, mix_merge[metric],mix_merge['RID']):
        groups_train=mix_merge['RID'][train_index]
        model=sp.GEE.from_formula(formula,groups=groups_train,data=mix_merge[train_index],cov_struct=sp.cov_struct.Exchangeable())
        result=model.fit()
        y_pred=result.predict(mix_merge[test_index])
        y_test=mix_merge[metric][test_index]
        mse=mean_squared_error(y_test,y_pred)
        mse_list.append(mse)
    avmse=np.mean(mse_list)
    print(f'Average MSE across {n_splits} folds: {avmse}')


#data transformations
mix_merge=pl.read_csv('mix_merge.csv')
merge=mix_merge.group_by('RID',maintain_order=True).all()
age=merge['AGE_bl'].to_list()
months=merge['M'].to_list()
gender=merge['PTGENDER'].to_list()
for i in range(len(age)):
    age[i]=age[i][0]+months[i][0]/12
    gender[i]=gender[i][0]
    months[i]=np.array(months[i])-months[i][0]
age=pl.Series('AGE_bl',age)
gender=pl.Series('PTGENDER',gender)
months=pl.Series('M',months)
merge.replace_column(1,age)
merge.replace_column(2,gender)
merge.replace_column(3,months)

for i,metric in enumerate(metrics):
    mat=mix_merge[metric].to_numpy()
    mat=(mat-np.mean(mat))/np.std(mat)
    mat=pl.Series(metric,mat)
    if(metric!='TAU' and metric!='ABETA'): mix_merge.replace_column(i+4,mat)
    else: mix_merge.replace_column(i+5,mat)
    
#Define metric
metric='MMSE'

#Run program
statistical_analysis(metric,merge,mix_merge)