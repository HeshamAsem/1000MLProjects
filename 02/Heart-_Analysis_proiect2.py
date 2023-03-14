#!/usr/bin/env python
# coding: utf-8

# In[66]:


#import librraries 
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
import numpy as np


# In[67]:


#read csv file  
df=pd.read_csv(r"D:\data science course\1000ml project\dataset1\heart_2020_cleaned.csv")


# In[68]:


df


# In[4]:


df.info()


# In[5]:


#chek null in data 
df.isnull().sum()


# ### Data Preprocessing

# In[6]:


#check duplicated data 
df.duplicated().sum()


# In[7]:


df[df.duplicated()]


# In[8]:


#drop dublicated data
df.drop_duplicates(inplace=True)


# In[9]:


# shape of data after remove duplicate
df.shape


# In[10]:


#look at each column values 
for col in df.columns:
    print(f'column {col}')
    print('***********************************')
    print(df[col].value_counts(),'\n')

# We Find that we have 4cloumns have numeric values [BMI,PhysicalHealth,MentalHealth,SleepTime]
#and the another column have text value we will convet them into numeric values


# In[11]:


#5 number summary of numerical columns
df.describe() 


# In[12]:


# drow boxplot of PhysicalHealth
def boxplot_drawer(Column_Name):
    sns.boxplot(y=Column_Name,data=df)

#we find outliers we shoud remove it we should remove values which out of upper limit and lower limit 
#lower limit < data <upper limit 


# In[13]:


#draw box plot PhysicalHealth column
boxplot_drawer("PhysicalHealth")


# In[14]:


# Remove outliers
def Remove_outliers(Column_Name):
    q1=df[Column_Name].quantile(0.25)
    q3=df[Column_Name].quantile(0.75)
    iqr=q3-q1
    df[Column_Name][(df[Column_Name]<(q1-1.5*iqr))|(df[Column_Name]>(q3+1.5*iqr))]=np.nan


# In[15]:


#remove outliers in  PhysicalHealth column
Remove_outliers("PhysicalHealth")


# In[16]:


df.isnull().sum()


# In[17]:


df.dropna(inplace=True)


# In[18]:


df["PhysicalHealth"].describe()


# In[19]:


#chech outliesrs in MentalHealth column
df["MentalHealth"].describe()


# In[20]:


#draw boxplot of "MentalHealth" column 
boxplot_drawer("MentalHealth")


# In[21]:


#remove outliers which in "MentalHealth"  
Remove_outliers("MentalHealth")


# In[22]:


df.dropna(inplace=True)


# In[23]:


#  MentalHealth after remove outliers
df["MentalHealth"].describe()


# In[24]:


# describe text value data
df.describe(include=['O'])
#df["HeartDisease"].value_counts()


# In[25]:


#we find that diabitic column have 4 categories we will convert it  
df["Diabetic"].unique()


# In[ ]:


#we should conver text value into 2 classes 1,0


# In[26]:


#convert data type of Heartdisease
df.loc[df.HeartDisease=="Yes","HeartDisease"]=1
df.loc[df.HeartDisease=="No","HeartDisease"]=0
df["HeartDisease"]=df["HeartDisease"].astype("int32")
df["HeartDisease"]


# In[27]:


#replace remaining columns classes by this way into 1,0
df =  df[df.columns].replace({'Yes':1, 'No':0,'No, borderline diabetes':0,'Yes (during pregnancy)':1,"Male":1,"Female":0})

df=df.apply(pd.to_numeric,errors="ignore")


# ### Exploring & Analysis  Data 

# In[28]:


#plot describe data histogram of each column   
df.hist(figsize=(20,15))
plt.show()


# In[29]:


#correlation betwwen coulmns 
cor =df.corr()


# In[30]:


#show it in heatmap
plt.figure(figsize = (14,7))
sns.heatmap(cor.rank(axis="columns"),annot=True) # fix size of fig


# ### Visulization Class

# In[32]:


class Visulization():
    # function to return  distribution of cases yes or no heartdisease  according do another column
    def distribution(column1_name,column2_name,class_label):
        #num of female and male have-------------------Heartdisese
        Have_DeartDisease_column2_name=pd.DataFrame(df[df[column1_name]==class_label][column2_name])
        print(Have_DeartDisease_column2_name.value_counts())
        Have_DeartDisease_column2_name.hist(figsize=(20,15))
        return plt.show()



    #suplot_Histoggramfunction
    def Subplot_Histogram(column1_name,column2_name):
        fig, ax = plt.subplots(figsize = (13,6))

        ax.hist(df[df[column1_name]==1][column2_name], bins=20,alpha=1 ,color="green", label="HeartDisease")
        ax.hist(df[df[column1_name]==0][column2_name], bins=20,alpha=.5 ,color="yellow", label="Normal")

        ax.set_xlabel(column2_name)
        ax.set_ylabel("Frequency")

        fig.suptitle(f"Distribution of Cases with Yes/No heartdisease according to {column2_name} ")

        ax.legend()
        

    #count_plotfunction
    def count_plot(column_name1,column_name2):

        plt.figure(figsize = (13,6))
        sns.countplot(x = df[column_name2], hue =column_name1, data = df)
        plt.xlabel(column_name2)
        plt.legend(['Normal',column_name1])
        plt.ylabel('Frequency')
        plt.show()


# ### 1 According to Sex Have a HeartDisease

# In[33]:


Visulization.distribution("HeartDisease","Sex",1)
# num of Male whoes have heartdisease > num of female


# ### 2 According to Sex Not Have HeartDisease

# In[36]:


Visulization.distribution("HeartDisease","Sex",0)
# num of Male whoes Not have heartdisease < num of female


# In[34]:


Visulization.Subplot_Histogram("HeartDisease","Sex")


# ### 3 According to Smoking ====> Have HeartDisease

# In[37]:


#find num of people who have Heartdisease according to he smoke or not 1 refer to have heartdisease
Visulization.distribution("HeartDisease","Smoking",1)
#we find that the num of  people who smoking and have aheartdisease largerthan whose not smooking 


# ### 4 According to Smoking ====>Not  Have HeartDisease

# In[38]:


#find num of people who have not Heartdisease according to he smoke or not 0 refer to not have heartdisease
Visulization.distribution("HeartDisease","Smoking",0)
#we find that the num of people whose not smoking and not have a heartdisease larger than whose smoking


# In[39]:


Visulization.Subplot_Histogram("HeartDisease","Smoking")


# ###  5 According to  drink (alchol or not)  Have a HeartDisease

# In[40]:


#we want to fin num of people who had aheart disese acoording to drink or not alchol 
Visulization.distribution("HeartDisease","AlcoholDrinking",1)

#we find that num of people who had heartdisease and not drink alchol > whose drink 


# ### 6 According to  drink (alchol or not)  Not Have a HeartDisease

# In[41]:


# we want to find num of  people whose hadnot aheart disease according to drink alchol 
Visulization.distribution("HeartDisease","AlcoholDrinking",0)
# we find that num of not dink and not have a heatdisease > drinking and not have 


# In[42]:


Visulization.Subplot_Histogram("HeartDisease","AlcoholDrinking")


# ### 7 According to stroke  Have a HeartDisease

# In[43]:


Visulization.distribution("HeartDisease","Stroke",1)
# number of people whoese not had stroke and have HeartDisease >num of people whoese had stroke and HeartDisease


# ### 8  According to stroke  not Have a HeartDisease

# In[44]:


Visulization.distribution("HeartDisease","Stroke",0)
# اكتشفنا ان عدد الناس ال مجلهاش سكتات قلبيه قبل كدا عدد عدم اصابتهم ب مرض القلب اكتر من عدد الناس ال اتعرضوا لسكاتات دماغيه 


# In[45]:


Visulization.Subplot_Histogram("HeartDisease","Stroke")


# ### 9  According to DiffWalking  Have a HeartDisease

# In[46]:


Visulization.distribution("HeartDisease","DiffWalking",1)
# اكتشفنا ان عدد الناس ال معنداش صعوبه فى المشي وعندها القلب اكثر من عدد الناس ال عندها مشاكل فى المشي 


# In[47]:


Visulization.distribution("HeartDisease","DiffWalking",0)
#هنا عدد الناش ال معندهاش مشاكل فى المشي ومعندهاش القلب اكثر من عدد  الناس ال عندها مشاكل فى المشي 


# In[48]:


Visulization.Subplot_Histogram("HeartDisease","DiffWalking")
# هنلاحظ ان نسبة الناس ال بتعانى من مرض القلب اعلى فى حالة ان كنت انت بتعانى من مشاكل فى المشي


#  ### 10 According to Diabetic 

# In[49]:


Visulization.distribution("HeartDisease","Diabetic",1)


# In[50]:


Visulization.Subplot_Histogram("HeartDisease","Diabetic")
# عدد الاشخاص ال غير مصابين ب السكرى ومصابين ب القلب اكثر من عدد الاشخاص المصابين ب السكرى 


# ### 11 According to PhysicalActivity

# In[51]:


Visulization.Subplot_Histogram("HeartDisease","PhysicalActivity")
# اكتشفنا ان عدد الناس ال مصابون ب القلب فى الفئه ال مارست نشاط رياضى اكثر من الفئه الاخرى


# ### 12 According to Asthma

# In[52]:


Visulization.Subplot_Histogram("HeartDisease","Asthma")
# عدد الناس ال معندهمش ال ربو وعندهم القلب اكثر من ال عندهم ال ربو


# ### 13 According to KidneyDisease 	

# In[53]:


Visulization.Subplot_Histogram("HeartDisease","KidneyDisease")
#عدد المصابين ب القلب فى حالة ال معندهوش امراض كلى اكثر من ال عنده امراض كلى


# ### 14 According to Race

# In[54]:


Visulization.Subplot_Histogram("HeartDisease","Race")
# الاشخاص البيض هم اكثر فئه مصابه بمرض القلب 


# In[35]:


Visulization.count_plot("HeartDisease","Race")


# ### 15 According to AgeCategory

# In[55]:


encode_AgeCategory= {'55-59':57, '80 or older':80, '65-69':67,
                      '75-79':77,'40-44':42,'70-74':72,'60-64':62,
                      '50-54':52,'45-49':47,'18-24':21,'35-39':37,
                      '30-34':32,'25-29':27}
df['AgeCategory'] = df['AgeCategory'].apply(lambda x: encode_AgeCategory[x])
df['AgeCategory'] = df['AgeCategory'].astype('float')


# In[56]:


Visulization.Subplot_Histogram("HeartDisease","AgeCategory")


# In[57]:


Visulization.count_plot("HeartDisease","AgeCategory")


# ### 15 According to GenHealth

# In[58]:


Visulization.Subplot_Histogram("HeartDisease","GenHealth")


# In[59]:


Visulization.count_plot("HeartDisease","GenHealth")


# ### 16 According to Sleep Time

# In[60]:


Visulization.count_plot("HeartDisease","SleepTime")
# اكثر ناس مصابه ب القلب هى الفئة التى تنم  7و8 ساعات


# ### 17 According to BMI

# In[61]:


fig, ax = plt.subplots(figsize = (13,5))
sns.kdeplot(df[df["HeartDisease"]==1]["BMI"], alpha=0.5,shade = True, color="green", label="HeartDisease", ax = ax)
sns.kdeplot(df[df["HeartDisease"]==0]["BMI"], alpha=0.5,shade = True, color="#fccc79", label="Normal", ax = ax)
plt.title('Distribution of Body Mass Index', fontsize = 18)
ax.set_xlabel("BodyMass")
ax.set_ylabel("Frequency")
ax.legend();
plt.show()

# اكتشفت ان اكثر ناس عندهم القلب هم ال  مؤشر كتلتهم 42 فوق


# ### According to PhysicalHealth

# In[62]:


Visulization.count_plot("HeartDisease","PhysicalHealth")
# اكثر فئة كانت مريضة ب القلب هى الفئة ال ماشاعرتش بتعب جسدى خلال اخر 30 يوم 


# ### According to MentalHealth

# In[63]:


Visulization.count_plot("HeartDisease","MentalHealth")


# In[64]:


df.columns=[x.lower() for x in df.columns]
#list comprehension we used it to convert column name int small


# In[65]:


df


# In[ ]:




