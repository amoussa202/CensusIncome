
# coding: utf-8

# # Census-Income (KDD) Data Set 
# 
# ## Data Set Information
# 
# This data set contains weighted census data extracted from the 1994 and 1995 Current Population Surveys conducted by the U.S. Census Bureau. The data contains 41 demographic and employment related variables.   
#   
# One instance per line with comma delimited fields. There are 199523 instances in the data file and 99762 in the test file.   
#   
# The data was split into train/test in approximately 2/3, 1/3 proportions using MineSet's MIndUtil mineset-to-mlc.  
# 
# Source: https://archive.ics.uci.edu/ml/datasets/Census-Income+%28KDD%29
# 
#   
# 
# ## Attribute Information
# 
# More information detailing the meaning of the attributes can be found in the Census Bureau's documentation To make use of the data descriptions at this site, the following mappings to the Census Bureau's internal database column names will be needed:   
#   
# age	'Age'    
# class of worker 'ClassOfWorker'   
# industry code	'IndustryCode'    
# occupation code	'OccupationCode'  
# education	'Education'     
# wage per hour	'WagePerHour'   
# enrolled in edu inst last wk	'EnrolledEducation'     
# marital status	'MaritalStatus'       
# major industry code	'MajorIndustryCode'    
# major occupation code	'MajorOccupationCode'        
# Race	'Race'    
# hispanic Origin	'HispanicOrigin'    
# sex	'Sex'    
# member of a labor union	'LabourUnion'     
# reason for unemployment	'ReasonUnemployed'           
# full or part time employment stat	'FullOrPartTime'       
# capital gains	'CapitalGains'    
# capital losses	'CapitalLosses'    
# divdends from stocks  'StockDividends' 
# federal income tax liability	     
# tax filer status	'TaxFilerStat'     
# region of previous residence	'PrevResidenceRegion'     
# state of previous residence	'PrevResidenceState'     
# detailed household and family stat	'HouseholdFamilyStatus'    
# detailed household summary in household	'HouseholdSummary'    
# instance weight	'InstanceWeight'   
# migration code-change in msa	'MigrationCodeChangeMSA'   
# migration code-change in reg	'MigrationCodeChangeReg'   
# migration code-move within reg	'MigrationCodeMoveWithinRegion'   
# live in this house 1 year ago	'LiveInHouse1Y'   
# migration prev res in sunbelt	'MigPrevResidenceSunbelt'   
# num persons worked for employer	'NumPersonsWorkedEmployer'   
# family members under 18	'FamilyMembersU18'   
# country of birth father	'CountryBirthFather'   
# country of birth mother	'CountryBirthMother'   
# country of birth self	'CountryBirthSelf'   
# citizenship	'Citizenship'   
# own business or self employed	'OwnBusiness'   
# fill inc questionnaire for veteran's admin	'QuestionnaireVeteran'   
# veterans benefits	'VeteranBenefits'   
# weeks worked in year	'WeeksWorkedInY'   
#   
# Incomes 'Income'  
# 
# 
# Note that Incomes have been binned at the $50K level to present a binary classification problem, much like the original UCI/ADULT database. The goal field of this data, however, was drawn from the "total person income" field rather than the "adjusted gross income" and may, therefore, behave differently than the original ADULT goal field. 
# 
# 
# ## Problem Statement
# 
# The goal of this analysis is to try to find which factors can be used to predict an individual's annual income (higher or lower than 50k USD), and then predict the income level based on these factors.
# 
# ## Summary
# 
# As part of the analysis, I will be going through the following steps:
# 
# **1- Data Extraction:**      
# Data is available in CSV files that can be downloaded at the source mentioned above.
# 
# **2- Data Cleaning**
# 
# **3- Exploratory Data Analysis**
# 
# **4- Modeling:**     
# a) Feature Selection: I will use Random Forest Classification for feature selection      
# b) Model Selection: I will use the selected features and apply Decision Trees and Logistic Regression. I will then proceed with the one with better rates.     
# 
# **5- Oversampling:**
# Given that the data is skewed (only 8% are positive), I will also try oversampling using SMOTE to enhance the recall rates. I will then compare both results (with and without oversampling).

# In[437]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# # Read the data

# In[438]:


#Training and Test data is already split into two CSV's. Hence we will read them separately, and apply all changes to both data sets

columns = ['Age', 'ClassOfWorker', 'IndustryCode', 'OccupationCode','Education', 'WagePerHour', 'EnrolledEducation',
          'MaritalStatus', 'MajorIndustryCode', 'MajorOccupationCode', 'Race', 'HispanicOrigin', 'Sex', 'LabourUnion',
          'ReasonUnemployed', 'FullOrPartTime', 'CapitalGains', 'CapitalLosses', 'StockDividends', 'TaxFilerStat',
          'PrevResidenceRegion', 'PrevResidenceState', 'HouseholdFamilyStatus', 'HouseholdSummary', 'InstanceWeight', 
          'MigrationCodeChangeMSA','MigrationCodeChangeReg', 'MigrationCodeMoveWithinRegion', 'LiveInHouse1Y', 
            'MigPrevResidenceSunbelt', 'NumPersonsWorkedEmployer', 'FamilyMembersU18', 'CountryBirthFather',
          'CountryBirthMother', 'CountryBirthSelf', 'Citizenship', 'OwnBusiness', 'QuestionnaireVeteran',
          'VeteranBenefits','WeeksWorkedInY', 'Year','Income']

df = pd.read_csv('Data/census-income.data', header=None)
df.columns = columns
df.drop(['InstanceWeight'], axis=1, inplace=True)

df_test = pd.read_csv('Data/census-income.test', header=None)
df_test.columns = columns
df_test.drop(['InstanceWeight'], axis=1, inplace=True)


# In[439]:


df.shape


# In[440]:


df_test.shape


# In[441]:


df.head()


# In[442]:


#Trim the strings in the data
for i in df.columns:
    if type(df[i][0]) == str:
        df[i] = df[i].apply(lambda x: str(x).strip())
        df_test[i] = df_test[i].apply(lambda x: str(x).strip())


# In[443]:


#Drop duplicates
df.drop_duplicates(inplace=True)
df_test.drop_duplicates(inplace=True)


# # Data Cleaning

# In[444]:


#Check missing data
for i in df.columns:
    print(df[i].unique())


# In[445]:


#Some missing data is represented as '?', others are 'Not in universe'
df.replace("?", np.nan, inplace=True)
df.replace("Not in universe", np.nan, inplace=True)
df_test.replace("?", np.nan, inplace=True)
df_test.replace("Not in universe", np.nan, inplace=True)


# In[446]:


df.isnull().sum()


# In[447]:


#Delete the columns with almost completely missing values
df.drop(['EnrolledEducation','LabourUnion', 'ReasonUnemployed', 'PrevResidenceRegion','PrevResidenceState', 
         'MigPrevResidenceSunbelt','QuestionnaireVeteran','FamilyMembersU18','MigrationCodeChangeMSA',
         'MigrationCodeChangeReg','MigrationCodeMoveWithinRegion'], axis=1, inplace=True)

df_test.drop(['EnrolledEducation','LabourUnion', 'ReasonUnemployed', 'PrevResidenceRegion','PrevResidenceState', 
         'MigPrevResidenceSunbelt','QuestionnaireVeteran','FamilyMembersU18','MigrationCodeChangeMSA',
         'MigrationCodeChangeReg','MigrationCodeMoveWithinRegion'], axis=1, inplace=True)


# In[448]:


#Drop Rows with few missing column values
df.dropna(subset=['CountryBirthFather','CountryBirthMother','CountryBirthSelf'], inplace=True)
df_test.dropna(subset=['CountryBirthFather','CountryBirthMother','CountryBirthSelf'], inplace=True)


# In[449]:


df.isnull().sum()


# In[450]:


#ClassOfWorker and MajorOccupationCode are missing for those who do not work. I will replace them with 'NA'
df[['ClassOfWorker','MajorOccupationCode']] = df[['ClassOfWorker','MajorOccupationCode']].fillna('NA')
df_test[['ClassOfWorker','MajorOccupationCode']] = df_test[['ClassOfWorker','MajorOccupationCode']].fillna('NA')


# In[451]:


df['Income>50k'] = np.where(df['Income'] == '- 50000.', 0, 1)
df.drop('Income', axis=1, inplace=True)

df_test['Income>50k'] = np.where(df_test['Income'] == '- 50000.', 0, 1)
df_test.drop('Income', axis=1, inplace=True)


# In[452]:


df.head(100)


# In[453]:


df.info()


# In[454]:


df.describe()


# Looks like there are some wrong values, such as WagePerHour (9999). I will look into those during EDA

# # Exploratory Data Analysis

# In[455]:


df['Income>50k'].value_counts()


# In[456]:


df['Income>50k'].value_counts(normalize=True)


# <b>Age

# In[457]:


df['Age'].describe()


# In[458]:


sns.distplot(df['Age'], bins=10)
plt.grid()


# In[459]:


sns.boxplot(x='Income>50k', y='Age',data=df)


# In[460]:


sns.violinplot(x='Income>50k', y='Age',data=df)


# In[461]:


plt.figure(figsize=(10,10))
g = sns.FacetGrid(data=df, col='Income>50k', row='Year', height=6)
g.map(sns.distplot, 'Age', bins=5)
for ax in g.axes.flat:
    labels = ax.get_xticklabels() # get x labels
    ax.set_xticklabels(labels, rotation=90) # set new labels


# <b>ClassOfWorker

# In[462]:


df['ClassOfWorker'].describe()


# In[463]:


df['ClassOfWorker'].value_counts()


# In[464]:


plt.figure(figsize=(12,8))
sns.barplot(y='ClassOfWorker', x='Income>50k', data=df, orient="h", hue='Year', dodge=True)
plt.grid(True)


# In[465]:


#Rename Never Worked and Without Pay to NA
#Join State and Local Govt
df['ClassOfWorker'] = np.where(df['ClassOfWorker'] == 'Never worked', 'NA', df['ClassOfWorker'])
df['ClassOfWorker'] = np.where(df['ClassOfWorker'] == 'Without pay', 'NA', df['ClassOfWorker'])
df['ClassOfWorker'] = np.where(df['ClassOfWorker'] == 'Local government', 'Non Federal Government', df['ClassOfWorker'])
df['ClassOfWorker'] = np.where(df['ClassOfWorker'] == 'State government', 'Non Federal Government', df['ClassOfWorker'])

df_test['ClassOfWorker'] = np.where(df_test['ClassOfWorker'] == 'Never worked', 'NA', df_test['ClassOfWorker'])
df_test['ClassOfWorker'] = np.where(df_test['ClassOfWorker'] == 'Without pay', 'NA', df_test['ClassOfWorker'])
df_test['ClassOfWorker'] = np.where(df_test['ClassOfWorker'] == 'Local government', 'Non Federal Government', df_test['ClassOfWorker'])
df_test['ClassOfWorker'] = np.where(df_test['ClassOfWorker'] == 'State government', 'Non Federal Government', df_test['ClassOfWorker'])

df['ClassOfWorker'].value_counts()


# In[466]:


sns.boxplot(y='ClassOfWorker', x='CapitalGains', data=df, orient='h')


# <b> Industry Code - Major Industry Code

# In[467]:


df['IndustryCode'].unique()


# In[468]:


df['IndustryCode'].value_counts()


# In[469]:


plt.figure(figsize=(20,20))
sns.countplot(y='Income>50k', hue='IndustryCode', data=df[df['Income>50k']==1], palette=sns.color_palette("hls", 8))
plt.grid()
plt.legend(fontsize='large')


# In[470]:


plt.figure(figsize=(12,20))
sns.barplot(y='IndustryCode', x='Income>50k', data=df, orient="h", dodge=True)
plt.grid(True)


# In[471]:


df['MajorIndustryCode'].value_counts()


# In[472]:


plt.figure(figsize=(15,10))
sns.countplot(y='Income>50k', hue='MajorIndustryCode', data=df[df['Income>50k']==1],palette=sns.color_palette("husl", 8))
plt.grid()
plt.legend(fontsize='medium')


# In[473]:


plt.figure(figsize=(12,30))
sns.barplot(y='MajorIndustryCode', x='Income>50k', data=df, orient="h", hue='Year')
plt.grid(True)


# In[474]:


df.groupby('MajorIndustryCode')['IndustryCode'].unique()


# In[475]:


df.groupby(['IndustryCode','MajorIndustryCode'])['Income>50k'].sum().sort_values()


# In[476]:


#Industry Code is too scattered, and the values are too small for each Code. I will drop it and use Major Industry Code only
df.drop('IndustryCode', axis=1, inplace=True)
df_test.drop('IndustryCode', axis=1, inplace=True)


# <b> OccupationCode and MajorOccupationCode

# In[477]:


df['OccupationCode'].describe()


# In[478]:


df['OccupationCode'].value_counts()


# In[479]:


df['MajorOccupationCode'].unique()


# In[480]:


plt.figure(figsize=(12,20))
sns.barplot(y='OccupationCode', x='Income>50k', data=df, orient="h", hue='Year', dodge=True)
plt.grid(True)


# In[481]:


df.groupby('MajorOccupationCode')['OccupationCode'].unique()


# In[482]:


g = sns.FacetGrid(data=df, col='Income>50k', row='Year', height=6)
g.map(sns.countplot, 'OccupationCode', order=df['OccupationCode'].unique())
for ax in g.axes.flat:
    labels = ax.get_xticklabels() # get x labels
    ax.set_xticklabels(labels, rotation=60) # set new labels


# In[483]:


plt.figure(figsize=(15,15))
sns.countplot(y='Income>50k', hue='OccupationCode', data=df[df['Income>50k']==1],palette=sns.color_palette("husl", 8))
plt.grid()
plt.legend(fontsize='medium')


# In[484]:


plt.figure(figsize=(12,20))
sns.barplot(y='MajorOccupationCode', x='Income>50k', data=df, orient="h", hue='Year', dodge=True)
plt.grid(True)


# In[485]:


plt.figure(figsize=(15,10))
sns.countplot(y='Income>50k', hue='MajorOccupationCode', data=df[df['Income>50k']==1],palette=sns.color_palette("husl", 8))
plt.grid()
plt.legend(fontsize='medium')


# In[486]:


#Occupation Code is too scattered, and the values are too small for each Code. I will drop it and use Major Occupation Code only
df.drop('OccupationCode', axis=1, inplace=True)
df_test.drop('OccupationCode', axis=1, inplace=True)


# <b>Education

# In[487]:


df['Education'].value_counts()


# In[488]:


g = sns.FacetGrid(data=df, col='Income>50k', row='Year', height=6)
g.map(sns.countplot, 'Education', order=df['Education'].unique() )
for ax in g.axes.flat:
    labels = ax.get_xticklabels() # get x labels
    ax.set_xticklabels(labels, rotation=90) # set new labels


# In[489]:


plt.figure(figsize=(12,12))
sns.barplot(y='Education', x='Income>50k', data=df, orient="h", hue='Year', dodge=True)
plt.grid(True)


# In[490]:


plt.figure(figsize=(10,10))
sns.countplot(y='Income>50k', hue='Education', data=df[df['Income>50k']==1],palette=sns.color_palette("husl", 8))
plt.grid()
plt.legend(fontsize='medium')


# In[491]:


df['Education'].unique()


# In[492]:


#I will group all Education levels other than 'High school graduate', 'Some college but no degree',
#Bachelors degree(BA AB BS)' and 'Masters degree(MA MS MEng MEd MSW MBA)' as 'Other'
#I will also rename 'Some college but no degree' as 'High school graduate'.
    
df['Education'] = df['Education'].apply(lambda x: 'Other' if x not in ['High school graduate', 'Some college but no degree',
        'Bachelors degree(BA AB BS)','Masters degree(MA MS MEng MEd MSW MBA)', 
        'Prof school degree (MD DDS DVM LLB JD)', 'Doctorate degree(PhD EdD)'] else x)

df['Education'] = np.where(df['Education'] == 'Some college but no degree', 'High school graduate', df['Education'] )

df_test['Education'] = df_test['Education'].apply(lambda x: 'Other' if x not in ['High school graduate', 'Some college but no degree',
        'Bachelors degree(BA AB BS)','Masters degree(MA MS MEng MEd MSW MBA)', 
        'Prof school degree (MD DDS DVM LLB JD)', 'Doctorate degree(PhD EdD)'] else x)

df_test['Education'] = np.where(df_test['Education'] == 'Some college but no degree', 'High school graduate', df_test['Education'] )


# <b>WagePerHour

# In[493]:


df.columns


# In[494]:


plt.figure(figsize=(10,10))
sns.boxplot(x='Income>50k', y='WagePerHour', data=df)


# In[495]:


df.groupby('WagePerHour')['Age'].count()


# In[496]:


plt.figure(figsize=(10,10))

g = sns.FacetGrid(data=df, col='Income>50k', row='Year', height=6)
g.map(sns.distplot, 'WagePerHour', bins=5)
for ax in g.axes.flat:
    labels = ax.get_xticklabels() # get x labels
    ax.set_xticklabels(labels, rotation=90) # set new labels


# In[497]:


sns.distplot(df['WagePerHour'], bins=4)


# In[498]:


sns.countplot(x='Income>50k', data=df[df['WagePerHour']>2000])


# #It looks like the WagePerHouse has some wrong inputs. I might delete the rows, but a big oercentage of them are positive 
# for Income>50K. I will cap the Wages at $2000 per hour.

# In[499]:


#I will cap the WagePerHour to $2,000. Higher numbers do not seem to be correct.

df['WagePerHour'] = np.where(df['WagePerHour'] > 2000, 2000, df['WagePerHour'])
df_test['WagePerHour'] = np.where(df_test['WagePerHour'] > 2000, 2000, df_test['WagePerHour'])


# In[500]:


sns.countplot(x='Income>50k', data=df[df['WagePerHour']== 2000])


# <b> Marital Status

# In[501]:


df['MaritalStatus'].value_counts()


# In[502]:


sns.countplot(x='Income>50k', hue='MaritalStatus', data=df)


# In[503]:


g = sns.FacetGrid(data=df, col='Income>50k', row='Year', height=6)
g.map(sns.countplot, 'MaritalStatus', order=df['MaritalStatus'].unique())
g.set_xticklabels(rotation=30)

for ax in g.axes.flat:
    labels = ax.get_xticklabels() # get x labels
    ax.set_xticklabels(labels, rotation=90) # set new labels


# In[504]:


plt.figure(figsize=(10,10))
sns.countplot(x='Income>50k', hue='MaritalStatus', data=df[df['Income>50k']==1],palette=sns.color_palette("husl", 8))
plt.grid()
plt.legend(fontsize='medium')


# In[505]:


plt.figure(figsize=(12,12))
sns.barplot(y='MaritalStatus', x='Income>50k', data=df, orient="h", hue='Year', dodge=True)
plt.grid(True)


# In[506]:


df['MaritalStatus'].unique()


# In[507]:


#I will join some values under two: Married and Divorced
df['MaritalStatus'] = df['MaritalStatus'].apply(lambda x: 'Divorced' if x in ['Divorced','Separated'] 
                else 'Married' if x in ['Married-civilian spouse present', 'Married-spouse absent', 'Married-A F spouse present']
                else x)

df_test['MaritalStatus'] = df_test['MaritalStatus'].apply(lambda x: 'Divorced' if x in ['Divorced','Separated'] 
                else 'Married' if x in ['Married-civilian spouse present', 'Married-spouse absent', 'Married-A F spouse present']
                else x)


# In[508]:


df['MaritalStatus'].value_counts()


# <b>Race

# In[509]:


df['Race'].value_counts()


# In[510]:


sns.countplot(x='Income>50k', hue='Race', data=df)


# In[511]:


g = sns.FacetGrid(data=df, col='Income>50k', row='Year', height=6)
g.map(sns.countplot, 'Race', order=df['Race'].unique())
g.set_xticklabels(rotation=30)

for ax in g.axes.flat:
    labels = ax.get_xticklabels() # get x labels
    ax.set_xticklabels(labels, rotation=90) # set new labels


# In[512]:


plt.figure(figsize=(12,12))
sns.barplot(y='Race', x='Income>50k', data=df, orient="h", hue='Year', dodge=True)
plt.grid(True)


# In[513]:


#I will add 'Amer Indian Aleut or Eskimo' to 'Other'
df['Race'] = df['Race'].apply(lambda x: 'Other' if x == 'Amer Indian Aleut or Eskimo' else x)
df_test['Race'] = df_test['Race'].apply(lambda x: 'Other' if x == 'Amer Indian Aleut or Eskimo' else x)


# <b>HispanicOrigin

# In[514]:


df['HispanicOrigin'].value_counts(normalize=True)


# In[515]:


g = sns.FacetGrid(data=df, col='Income>50k', row='Year', height=6)
g.map(sns.countplot, 'HispanicOrigin', order=df['HispanicOrigin'].unique())
g.set_xticklabels(rotation=30)

for ax in g.axes.flat:
    labels = ax.get_xticklabels() # get x labels
    ax.set_xticklabels(labels, rotation=90) # set new labels


# In[516]:


df['HispanicOrigin'].replace('All other', 'NA', inplace=True)
df_test['HispanicOrigin'].replace('All other', 'NA', inplace=True)


# In[517]:


g = sns.FacetGrid(data=df[df['HispanicOrigin'] !='NA'], col='Income>50k', row='Year', height=6)
g.map(sns.countplot, 'HispanicOrigin', order=df['HispanicOrigin'].unique())
g.set_xticklabels(rotation=30)

for ax in g.axes.flat:
    labels = ax.get_xticklabels() # get x labels
    ax.set_xticklabels(labels, rotation=90) # set new labels


# In[518]:


plt.figure(figsize=(12,12))
sns.barplot(y='HispanicOrigin', x='Income>50k', data=df[df['HispanicOrigin'] !='NA'], orient="h", dodge=True)
plt.grid(True)


# In[519]:


#I will drop this column. I will only be relying on Race
df.drop('HispanicOrigin', axis=1, inplace=True)
df_test.drop('HispanicOrigin', axis=1, inplace=True)


# <b>Sex

# In[520]:


g = sns.FacetGrid(data=df, col='Income>50k', row='Year', height=6)
g.map(sns.countplot, 'Sex', order=df['Sex'].unique())

for ax in g.axes.flat:
    labels = ax.get_xticklabels() # get x labels
    ax.set_xticklabels(labels, rotation=90) # set new labels


# In[521]:


plt.figure(figsize=(12,12))
sns.barplot(x='Sex', y='Income>50k', data=df,  dodge=True)
plt.grid(True)


# In[522]:


df.rename(columns={'Sex':'Male'}, inplace=True)
df['Male'] = np.where(df['Male']=='Male',1,0)
df_test.rename(columns={'Sex':'Male'}, inplace=True)
df_test['Male'] = np.where(df_test['Male']=='Male',1,0)


# <b>FullOrPartTime

# In[523]:


df['FullOrPartTime'].value_counts()


# In[524]:


g = sns.FacetGrid(data=df, col='Income>50k', row='Year', height=6)
g.map(sns.countplot, 'FullOrPartTime', order=df['FullOrPartTime'].unique() )
for ax in g.axes.flat:
    labels = ax.get_xticklabels() # get x labels
    ax.set_xticklabels(labels, rotation=90) # set new labels


# In[525]:


plt.figure(figsize=(12,12))
sns.barplot(x='FullOrPartTime', y='Income>50k', data=df,  hue='Year', dodge=True)
plt.xticks(rotation=45)
plt.grid(True)


# Looks like the data for this column is missing for 1994. It doesn't make sense that all those who answered the survey in 1994 are of the same type
# 

# In[526]:


sns.countplot(x='Income>50k', hue='FullOrPartTime', data=df[df['Year']==94])


# In[527]:


plt.figure(figsize=(10,10))
sns.countplot(x='Income>50k', hue='FullOrPartTime', data=df[df['Income>50k']==1],palette=sns.color_palette("husl", 8))
plt.grid()
plt.legend(fontsize='medium')


# Even though this columns seems to be correlated with Income levels, but I will have to drop it as I cannot impute the values of 1994

# In[528]:


df.drop('FullOrPartTime', axis=1, inplace=True)
df_test.drop('FullOrPartTime', axis=1, inplace=True)


# <b>CapitalGains & CapitalLosses

# In[529]:


df['CapitalGains'].describe()


# In[530]:


plt.figure(figsize=(10,10))
g = sns.FacetGrid(data=df, col='Income>50k', row='Year', height=6)
g.map(sns.distplot, 'CapitalGains', bins=5)
for ax in g.axes.flat:
    labels = ax.get_xticklabels() # get x labels
    ax.set_xticklabels(labels, rotation=90) # set new labels


# In[531]:


sns.distplot(df['CapitalGains'], bins=4)


# In[532]:


plt.figure(figsize=(8,8))
sns.boxplot(y='CapitalGains', x='Income>50k', data=df[df['CapitalGains']>0])


# In[533]:


df['CapitalLosses'].describe()


# In[534]:


plt.figure(figsize=(10,10))
g = sns.FacetGrid(data=df, col='Income>50k', row='Year', height=6)
g.map(sns.distplot, 'CapitalLosses', bins=5)
for ax in g.axes.flat:
    labels = ax.get_xticklabels() # get x labels
    ax.set_xticklabels(labels, rotation=90) # set new labels


# In[535]:


sns.distplot(df['CapitalLosses'], bins=4)


# In[536]:


plt.figure(figsize=(8,8))
sns.boxplot(y='CapitalLosses', x='Income>50k', data=df[df['CapitalLosses']>0])


# In[537]:


plt.figure(figsize=(8,8))
sns.violinplot(y='CapitalLosses', x='Income>50k', data=df[df['CapitalLosses']>0])


# In[538]:


plt.figure(figsize=(8,8))
sns.swarmplot(y='CapitalLosses', x='Income>50k', data=df[df['CapitalLosses']>0])


# This is unexpected, since higher income should probably mean lower Capital Losses.

# <b>StockDividends

# In[539]:


df['StockDividends'].describe()


# In[540]:


plt.figure(figsize=(10,10))
g = sns.FacetGrid(data=df, col='Income>50k', row='Year', height=6)
g.map(sns.distplot, 'StockDividends', bins=5)
for ax in g.axes.flat:
    labels = ax.get_xticklabels() # get x labels
    ax.set_xticklabels(labels, rotation=90) # set new labels


# In[541]:


sns.distplot(df['StockDividends'], bins=4)
plt.xlim(0,30000)


# In[542]:


plt.figure(figsize=(8,8))
sns.boxplot(y='StockDividends', x='Income>50k', data=df[df['StockDividends']>0])


# <b>TaxFilerStat

# In[543]:


df['TaxFilerStat'].describe()


# In[544]:


df['TaxFilerStat'].value_counts()


# In[545]:


g = sns.FacetGrid(data=df, col='Income>50k', row='Year', height=6)
g.map(sns.countplot, 'TaxFilerStat', order=df['TaxFilerStat'].unique() )
for ax in g.axes.flat:
    labels = ax.get_xticklabels() # get x labels
    ax.set_xticklabels(labels, rotation=90) # set new labels


# In[546]:


plt.figure(figsize=(12,12))
sns.barplot(x='TaxFilerStat', y='Income>50k', data=df,  hue='Year', dodge=True)
plt.xticks(rotation=45)
plt.grid(True)


# In[547]:


plt.figure(figsize=(10,10))
sns.countplot(x='Income>50k', hue='TaxFilerStat', data=df[df['Income>50k']==1],palette=sns.color_palette("husl", 8))
plt.grid()
plt.legend(fontsize='medium')


# In[548]:


#Group the values: ['Head of household', 'Joint both 65+', 'Joint one under 65 & one 65+'] as 'Other'
df['TaxFilerStat'] = df['TaxFilerStat'].apply(lambda x: 'Other' if x in ['Head of household', 'Joint both 65+', 'Joint one under 65 & one 65+'] else x)
df_test['TaxFilerStat'] = df_test['TaxFilerStat'].apply(lambda x: 'Other' if x in ['Head of household', 'Joint both 65+', 'Joint one under 65 & one 65+'] else x)


# <b>HouseholdFamilyStatus

# In[549]:


df['HouseholdFamilyStatus'].describe()


# In[550]:


df['HouseholdFamilyStatus'].value_counts()


# In[551]:


plt.figure(figsize=(12,12))
sns.barplot(x='HouseholdFamilyStatus', y='Income>50k', data=df, dodge=True)
plt.xticks(rotation=90)
plt.grid(True)


# In[552]:


plt.figure(figsize=(10,10))
sns.countplot(x='Income>50k', hue='HouseholdFamilyStatus', data=df[df['Income>50k']==1],palette=sns.color_palette("husl", 8))
plt.grid()
plt.legend(fontsize='medium')


# In[553]:


g = sns.FacetGrid(data=df, col='Income>50k', row='Year', height=6)
g.map(sns.countplot, 'HouseholdFamilyStatus', order=df['HouseholdFamilyStatus'].unique() )
for ax in g.axes.flat:
    labels = ax.get_xticklabels() # get x labels
    ax.set_xticklabels(labels, rotation=90) # set new labels


# In[554]:


#To simplify the model, I will replace all small values with Other
df['HouseholdFamilyStatus'] = df['HouseholdFamilyStatus'].apply(lambda x: 'Other' if x not in ['Householder','Spouse of householder','Nonfamily householder'] else x)
df_test['HouseholdFamilyStatus'] = df_test['HouseholdFamilyStatus'].apply(lambda x: 'Other' if x not in ['Householder','Spouse of householder','Nonfamily householder'] else x)


# In[555]:


df['HouseholdFamilyStatus'].value_counts()


# <b>LiveInHouse1Y

# In[556]:


df['LiveInHouse1Y'].describe()


# In[557]:


df['LiveInHouse1Y'].value_counts()


# In[558]:


#Too much missing info, I will drop this column
df.drop('LiveInHouse1Y', axis=1, inplace=True)
df_test.drop('LiveInHouse1Y', axis=1, inplace=True)


# <b>NumPersonsWorkedEmployer

# In[559]:


df['NumPersonsWorkedEmployer'].describe()


# In[560]:


g = sns.FacetGrid(data=df, col='Income>50k', row='Year', height=6)
g.map(sns.countplot, 'NumPersonsWorkedEmployer', order=df['NumPersonsWorkedEmployer'].unique() )
for ax in g.axes.flat:
    labels = ax.get_xticklabels() # get x labels
    ax.set_xticklabels(labels, rotation=90) # set new labels


# In[561]:


plt.figure(figsize=(12,12))
sns.barplot(x='NumPersonsWorkedEmployer', y='Income>50k', data=df, dodge=True)
plt.xticks(rotation=90)
plt.grid(True)


# In[562]:


plt.figure(figsize=(10,10))
sns.countplot(x='Income>50k', hue='NumPersonsWorkedEmployer', data=df[df['Income>50k']==1],palette=sns.color_palette("husl", 8))
plt.grid()
plt.legend(fontsize='medium')


# In[563]:


sns.boxplot(x='Income>50k', y="NumPersonsWorkedEmployer", data=df)


# <b>CountryBirthFather

# In[564]:


df['CountryBirthFather'].describe()


# In[565]:


plt.figure(figsize=(10,10))
sns.countplot(y='Income>50k', hue='CountryBirthFather', data=df, orient='H',palette=sns.color_palette("husl", 8))
plt.grid()
plt.legend(fontsize='x-small')


# In[566]:


plt.figure(figsize=(10,10))
sns.countplot(y='Income>50k', hue='CountryBirthFather', data=df[df['Income>50k']==1], orient='H',palette=sns.color_palette("husl", 8))
plt.grid()
plt.legend(fontsize='x-small')


# In[567]:


plt.figure(figsize=(12,12))
sns.barplot(x='CountryBirthFather', y='Income>50k', data=df, dodge=True)
plt.xticks(rotation=90)
plt.grid(True)


# In[568]:


plt.figure(figsize=(12,12))
sns.barplot(x='CountryBirthMother', y='Income>50k', data=df, dodge=True)
plt.xticks(rotation=90)
plt.grid(True)


# In[569]:


plt.figure(figsize=(12,12))
sns.barplot(x='CountryBirthSelf', y='Income>50k', data=df, dodge=True)
plt.xticks(rotation=90)
plt.grid(True)


# In[570]:


#BirthCountry columns don't seem to be relevant. I will drop them for now.
df.drop(['CountryBirthFather','CountryBirthMother','CountryBirthSelf'], axis=1, inplace=True)
df_test.drop(['CountryBirthFather','CountryBirthMother','CountryBirthSelf'], axis=1, inplace=True)


# <b>Citizenship

# In[571]:


df['Citizenship'].describe()


# In[572]:


df['Citizenship'].value_counts()


# In[573]:


df['Citizenship'].value_counts(normalize=True)


# In[574]:


plt.figure(figsize=(10,10))
sns.countplot(x='Income>50k', hue='Citizenship', data=df[df['Income>50k']==1],palette=sns.color_palette("husl", 8))
plt.grid()
plt.legend(fontsize='x-small')


# In[575]:


plt.figure(figsize=(12,12))
sns.barplot(x='Citizenship', y='Income>50k', data=df, dodge=True)
plt.xticks(rotation=90)
plt.grid(True)


# In[576]:


#I will replace the values with 1 = US Citizen, and 0 = Non-US Citizen   
df['Citizenship'] = df['Citizenship'].apply(lambda x: 0 if x == 'Foreign born- Not a citizen of U S' else 1)
df_test['Citizenship'] = df_test['Citizenship'].apply(lambda x: 0 if x == 'Foreign born- Not a citizen of U S' else 1)


# <b>OwnBusiness

# In[577]:


df['OwnBusiness'].value_counts()


# In[578]:


plt.figure(figsize=(12,12))
sns.barplot(x='OwnBusiness', y='Income>50k', data=df, dodge=True)
plt.xticks(rotation=90)
plt.grid(True)


# In[579]:


#Very clear correlation between Owning a Business and Income.
#I will rename '2' to 1.
df['OwnBusiness'] = np.where(df['OwnBusiness'] == 0, 0, 1)
df_test['OwnBusiness'] = np.where(df_test['OwnBusiness'] == 0, 0, 1)

df['OwnBusiness'].value_counts(normalize=True)


# <b>VeteranBenefits'

# In[580]:


df['VeteranBenefits'].describe()


# In[581]:


df['VeteranBenefits'].value_counts()


# In[582]:


plt.figure(figsize=(12,12))
sns.barplot(x='VeteranBenefits', y='Income>50k', data=df, dodge=True)
plt.xticks(rotation=90)
plt.grid(True)


# In[583]:


#Couldn't find any explanations for this column, what it means, what are the digits. hence I will drop it for now
df.drop('VeteranBenefits', axis=1, inplace=True)
df_test.drop('VeteranBenefits', axis=1, inplace=True)


# <b>WeeksWorkedInY

# In[584]:


df['WeeksWorkedInY'].describe()


# In[585]:


plt.figure(figsize=(12,12))
sns.boxplot(x='Income>50k', y='WeeksWorkedInY', data=df)


# In[586]:


sns.distplot(df['WeeksWorkedInY'])


# In[587]:


g = sns.FacetGrid(data=df, col='Income>50k', height=6)
g.map(sns.distplot, 'WeeksWorkedInY' )
for ax in g.axes.flat:
    labels = ax.get_xticklabels() # get x labels
    ax.set_xticklabels(labels, rotation=90) # set new labels


# In[588]:


plt.figure(figsize=(12,12))
sns.violinplot(x='Income>50k', y='WeeksWorkedInY', data=df)


# <b>Year

# In[589]:


plt.figure(figsize=(12,12))
sns.barplot(x='Year', y='Income>50k', data=df, dodge=True)
plt.xticks(rotation=90)
plt.grid(True)


# In[590]:


plt.figure(figsize=(12,12))
sns.countplot(x='Year', data=df, hue='Income>50k',dodge=True)
plt.xticks(rotation=90)
plt.grid(True)


# In[591]:


#Year doesn't seem to be correlated with Income levels. Even the above charts were very similar for both years
df.drop('Year', axis=1, inplace=True)
df_test.drop('Year', axis=1, inplace=True)


# In[592]:


df.info()


# In[593]:


df.describe()


# In[594]:


df.head()


# In[595]:


plt.subplots(figsize=(10, 10))
df_cor = df.corr()
sns.heatmap(df_cor, annot=True, fmt = ".1f", cmap = "coolwarm")


# In[596]:


df_test.shape


# In[597]:


df.shape


# # MODELING

# ## Encoding
# 

# In[598]:


df_train_encoded=pd.get_dummies(df, columns=['ClassOfWorker', 'Education', 'MaritalStatus', 'MajorIndustryCode', 
                                       'MajorOccupationCode', 'Race', 'TaxFilerStat', 'HouseholdFamilyStatus',
                                      'HouseholdSummary'],
                         prefix=['ClassOfWorker', 'Education', 'MaritalStatus', 'MajorIndustryCode', 
                                       'MajorOccupationCode', 'Race', 'TaxFilerStat', 'HouseholdFamilyStatus',
                                      'HouseholdSummary'])

df_test_encoded=pd.get_dummies(df_test, columns=['ClassOfWorker', 'Education', 'MaritalStatus', 'MajorIndustryCode', 
                                       'MajorOccupationCode', 'Race', 'TaxFilerStat', 'HouseholdFamilyStatus',
                                      'HouseholdSummary'],
                         prefix=['ClassOfWorker', 'Education', 'MaritalStatus', 'MajorIndustryCode', 
                                       'MajorOccupationCode', 'Race', 'TaxFilerStat', 'HouseholdFamilyStatus',
                                      'HouseholdSummary'])


# In[599]:


df_train_encoded.shape


# In[600]:


X_train = df_train_encoded.loc[:,df_train_encoded.columns != 'Income>50k']
y_train = df_train_encoded['Income>50k']
X_test = df_test_encoded.loc[:,df_test_encoded.columns != 'Income>50k']
y_test = df_test_encoded['Income>50k']


# ## Use RandomForest for Feature Selection

# In[601]:


#First, scale the data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)
X_train = pd.DataFrame(scaler.transform(X_train), columns=X_train.columns)
X_test = pd.DataFrame(scaler.transform(X_test),  columns=X_test.columns)


# In[602]:


from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=300)
rfc.fit(X_train, y_train)


# In[603]:


from sklearn.feature_selection import SelectFromModel
sel = SelectFromModel(rfc, prefit=True)
selected_feat= X_train.columns[(sel.get_support())]


# In[604]:


print('Selected Features:\n',*selected_feat, sep='\n')


# In[605]:


X_train = X_train[selected_feat]
X_test = X_test[selected_feat]


# In[606]:


df2 = pd.concat([X_train,y_train.reset_index()], axis=1)
df2.drop('index', axis = 1 ,inplace=True)

plt.subplots(figsize=(10, 10))
df_cor = df2.corr()
sns.heatmap(df_cor, annot=True, fmt = ".1f", cmap = "coolwarm")


# ## Model Selection

# In[607]:


X_train.shape


# In[608]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, learning_curve


# In[609]:


kfold = StratifiedKFold(n_splits=10)


# In[610]:


#Doing 10-fold cross validation, using Decision Tree and Logistic Regression

rs=42

classifiers = [] # list of classifiers tested
classifiers.append(LogisticRegression(random_state = rs))
classifiers.append(DecisionTreeClassifier(random_state = rs))

cv_results = []
for classifier in classifiers :
    cv_results.append(cross_val_score(classifier, X_train, y_train, scoring = 'roc_auc', cv = kfold, n_jobs=-1))

cv_means = []
cv_std = []
for cv_result in cv_results:
    cv_means.append(cv_result.mean())
    cv_std.append(cv_result.std())

cv_res = pd.DataFrame({'CV_score':cv_means, 'CV_stddev':cv_std, 'Algorithm':['LogisticRegression','DecisionTree']})


# In[611]:


cv_res


# In[612]:


plt.subplots(figsize=(10, 10))
g = sns.barplot('CV_score','Algorithm', data = cv_res, palette='Set2', orient = 'h',**{'xerr':cv_std})
g.set_xlabel('Mean AUC score')
g = g.set_title('Cross validation scores')


# ## Logistic Regression

# In[613]:


from sklearn.metrics import roc_curve, auc

LR = LogisticRegression(random_state=42)
y_score = LR.fit(X_train, y_train).decision_function(X_test)
fpr = dict()
tpr = dict()
roc_auc = dict()

fpr[0], tpr[0], _ = roc_curve(y_test, y_score)
roc_auc[0] = auc(fpr[0], tpr[0])

# Compute micro-average ROC curve and ROC area
fpr[1], tpr[1], _ = roc_curve(y_test.ravel(), y_score.ravel())
roc_auc[1] = auc(fpr[1], tpr[1])


# In[614]:


plt.figure(figsize=(10,10))
lw = 2
plt.plot(fpr[0], tpr[0], color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[0])
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC - Logistic Regression')
plt.legend(loc="lower right")
plt.show()


# In[615]:


import itertools
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# In[616]:


# use trained model to make predictions on test set
y_pred = LR.predict(X_test)


# In[617]:


# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, y_pred)
np.set_printoptions(precision=2)

class_names = ['Income < 50k', 'Income > 50k']

# Plot non-normalized confusion matrix
plt.figure(figsize=(10,10))
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix (counts)')

# Plot normalized confusion matrix
plt.figure(figsize=(10,10))
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title='Confusion matrix (percent)')


# In[618]:


from sklearn.metrics import accuracy_score, precision_score, recall_score

print("Accuracy: " , "%.2f" % (accuracy_score(y_test, y_pred)*100),'%')
print("Precision: " , "%.2f" % (precision_score(y_test, y_pred)*100),'%')
print("Recall: " , "%.2f" % (recall_score(y_test, y_pred)*100),'%')


# <b>Accuracy is 93.70%, i.e. better than simply predicting that everybody make <50k.   
# Precision 70.32% means that 70.32 of the one predicted >50k are actually >50k   
# Recall 33% means that out of the ones who actually make >50k, the model could only find 33% of them.   
# Given the skewness of the data, I will to use Oversampling, hoping to get better results.   

# ## Oversampling

# Due to the unbalanced data I use SMOTE to oversample the training data.

# In[619]:


from imblearn.over_sampling import SMOTE

print("Number transactions X_train dataset: ", X_train.shape)
print("Number transactions y_train dataset: ", y_train.shape)
print("Number transactions X_test dataset: ", X_test.shape)
print("Number transactions y_test dataset: ", y_test.shape)


# In[620]:


print("Before OverSampling, counts of label '1': {}".format(sum(y_train==1)))
print("Before OverSampling, counts of label '0': {} \n".format(sum(y_train==0)))

sm = SMOTE(random_state=2)
X_train_res, y_train_res = sm.fit_sample(X_train, y_train.ravel())

print('After OverSampling, the shape of train_X: {}'.format(X_train_res.shape))
print('After OverSampling, the shape of train_y: {} \n'.format(y_train_res.shape))

print("After OverSampling, counts of label '1': {}".format(sum(y_train_res==1)))
print("After OverSampling, counts of label '0': {}".format(sum(y_train_res==0)))


# In[621]:


X_train_res = pd.DataFrame(X_train_res, columns = X_train.columns)


# In[622]:


#Using Random Forest for feature selection
rfc.fit(X_train_res, y_train_res)
sel = SelectFromModel(rfc, prefit=True)
selected_feat= X_train_res.columns[(sel.get_support())]
print('Selected Features:\n',*selected_feat, sep='\n')
X_train_res = X_train_res[selected_feat]
X_test = X_test[selected_feat]


# In[623]:


#Doing 10-fold cross validation, using Decision Tree and Logistic Regression

rs=42

classifiers = [] # list of classifiers tested
classifiers.append(LogisticRegression(random_state = rs))
classifiers.append(DecisionTreeClassifier(random_state = rs))

cv_results = []
for classifier in classifiers :
    cv_results.append(cross_val_score(classifier, X_train_res, y_train_res, scoring = 'roc_auc', cv = kfold, n_jobs=-1))

cv_means = []
cv_std = []
for cv_result in cv_results:
    cv_means.append(cv_result.mean())
    cv_std.append(cv_result.std())

cv_res = pd.DataFrame({'CV_score':cv_means, 'CV_stddev':cv_std, 'Algorithm':['LogisticRegression','DecisionTree']})


# In[624]:


cv_res


# In[625]:


plt.subplots(figsize=(10, 10))
g = sns.barplot('CV_score','Algorithm', data = cv_res, palette='Set2', orient = 'h',**{'xerr':cv_std})
g.set_xlabel('Mean AUC score')
g = g.set_title('Cross validation scores')


# In[626]:


# define LogisticRegression
LR = DecisionTreeClassifier(random_state=42)

# fit LR model to (oversampled) training data
LR.fit(X_train_res, y_train_res)


# In[627]:


# use trained model to make predictions on test set
y_pred = LR.predict(X_test)


# In[628]:


# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, y_pred)
np.set_printoptions(precision=2)

class_names = ['Income < 50k', 'Income > 50k']

# Plot non-normalized confusion matrix
plt.figure(figsize=(10,10))
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix (counts)')

# Plot normalized confusion matrix
plt.figure(figsize=(10,10))
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title='Confusion matrix (percent)')


# In[629]:


print("Accuracy: " , "%.2f" % (accuracy_score(y_test, y_pred)*100),'%')
print("Precision: " , "%.2f" % (precision_score(y_test, y_pred)*100),'%')
print("Recall: " , "%.2f" % (recall_score(y_test, y_pred)*100),'%')


# <b>Recall rate got much better after oversampling, but accuracy and precisoin went down.

# ## Conclusion

# Based on the above two scenarios (with and without oversampling), the user can select the one that matches his requirements. Since Recall and precision varied significantly, then it will depend on what is more important:  
# 1- Making correct predictions => No oversampling  
# 2- Finding as many >50k as possible => Oversampling  
# 3- Correctly predicting >50k => No Oversampling  
