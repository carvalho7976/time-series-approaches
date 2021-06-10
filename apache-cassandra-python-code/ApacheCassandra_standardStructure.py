#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('run', 'auxStandardStructure.ipynb')


# In[2]:


import tensorflow as tf
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

if tf.test.gpu_device_name():
    print('GPU found')
else:
    print("No GPU found")


# ## Preprocessing

# In[3]:


all_releases_df = pd.read_csv('all_releases.csv')


# In[4]:


X, y = generateStandardTimeSeriesStructure(all_releases_df, 2)


# In[5]:


print("Declaring a dictionary to save results...")
results_dict = dict()
print("... DONE!")

print("Splitting dataset into train and test sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
print("General information:")
print("X Train set:", X_train.shape[0], "X Test set:", X_test.shape[0])
print("y Train set:", y_train.shape[0], "y Test set:", y_test.shape[0])
print("... DONE!")

print("Scaling features...")
scaler = MinMaxScaler()
X_train = pd.DataFrame(scaler.fit_transform(X_train))
X_test = pd.DataFrame(scaler.fit_transform(X_test))
print("... DONE!")

print("Setting stratified k-fold...")
k=10
kf = StratifiedKFold(n_splits=k, shuffle=False, random_state=42)
print("k =", k)
print("... DONE!\n")


# In[6]:


y_test = pd.DataFrame(y_test)
y_train = pd.DataFrame(y_train)


# ## Statistical Analysis

# In[7]:


all_releases_df.describe()


# In[8]:


ax = y_train.groupby([0])[0].count().plot.bar(title="Class Distribution", figsize=(5,5))
y_train.groupby([0])[0].count()


# In[9]:


X_train.corr(method='spearman').style.format("{:.2}").background_gradient(cmap=plt.get_cmap('coolwarm'), axis=1)


# ## Imbalanced baseline

# In[10]:


get_ipython().run_cell_magic('time', '', 'LogisticRegr_(X_train, y_train, X_test, y_test)')


# In[11]:


get_ipython().run_cell_magic('time', '', 'DecisionTree_(X_train, y_train, X_test, y_test)')


# In[12]:


get_ipython().run_cell_magic('time', '', 'RandomForest_(X_train, y_train, X_test, y_test)')


# In[13]:


get_ipython().run_cell_magic('time', '', 'NN_(X_train, y_train, X_test, y_test)')


# ## Undersampling

# In[14]:


print("Resampling dataset using Random UnderSampling (RUS)...")
X_RUS, y_RUS = RandomUnderSampler(random_state=42).fit_sample(X_train, y_train.values.ravel())
print("... DONE!")
print("X and Y RUS:", len(X_RUS), len(y_RUS))


# In[15]:


get_ipython().run_cell_magic('time', '', 'LogisticRegr_(X_RUS, y_RUS, X_test, y_test)')


# In[16]:


get_ipython().run_cell_magic('time', '', 'DecisionTree_(X_RUS, y_RUS, X_test, y_test)')


# In[17]:


get_ipython().run_cell_magic('time', '', 'RandomForest_(X_RUS, y_RUS, X_test, y_test)')


# In[18]:


get_ipython().run_cell_magic('time', '', 'NN_(X_RUS, y_RUS, X_test, y_test)')


# In[19]:


print("Resampling dataset using Edited Nearest Neighbour (ENN)...")
X_ENN, y_ENN = EditedNearestNeighbours(random_state=42).fit_sample(X_train, y_train.values.ravel())
print("... DONE!")
print("X and Y ENN:", len(X_ENN), len(y_ENN))


# In[20]:


get_ipython().run_cell_magic('time', '', 'LogisticRegr_(X_ENN, y_ENN, X_test, y_test)')


# In[21]:


get_ipython().run_cell_magic('time', '', 'DecisionTree_(X_ENN, y_ENN, X_test, y_test)')


# In[22]:


get_ipython().run_cell_magic('time', '', 'RandomForest_(X_ENN, y_ENN, X_test, y_test)')


# In[23]:


get_ipython().run_cell_magic('time', '', 'NN_(X_ENN, y_ENN, X_test, y_test)')


# In[24]:


print("Resampling dataset using Tomek's Link (TL)...")
X_TL, y_TL = TomekLinks(random_state=42).fit_sample(X_train, y_train.values.ravel())
print("... DONE!")
print("X and Y TL:", len(X_TL), len(y_TL))


# In[25]:


get_ipython().run_cell_magic('time', '', 'LogisticRegr_(X_TL, y_TL, X_test, y_test)')


# In[26]:


get_ipython().run_cell_magic('time', '', 'DecisionTree_(X_TL, y_TL, X_test, y_test)')


# In[27]:


get_ipython().run_cell_magic('time', '', 'RandomForest_(X_TL, y_TL, X_test, y_test)')


# In[28]:


get_ipython().run_cell_magic('time', '', 'NN_(X_TL, y_TL, X_test, y_test)')


# ## Oversampling

# In[29]:


print("Resampling dataset using Random OverSampling (ROS)...")
ros = RandomOverSampler(random_state=42)
X_ROS, y_ROS = ros.fit_resample(X_train, y_train)
print("X and Y ROS:", len(X_ROS), len(y_ROS))


# In[30]:


get_ipython().run_cell_magic('time', '', 'LogisticRegr_NoIloc(X_ROS, y_ROS, X_test, y_test)')


# In[31]:


get_ipython().run_cell_magic('time', '', 'DecisionTree_NoIloc(X_ROS, y_ROS, X_test, y_test)')


# In[32]:


get_ipython().run_cell_magic('time', '', 'RandomForest_NoIloc(X_ROS, y_ROS, X_test, y_test)')


# In[33]:


get_ipython().run_cell_magic('time', '', 'NN_NoIloc(X_ROS, y_ROS, X_test, y_test)')


# In[34]:


print("Resampling dataset using SMOTE (SMO)...")
sm = SMOTE(random_state=42)
X_SMO, y_SMO = sm.fit_resample(X_train, y_train)
print("X and Y SMO:", len(X_SMO), len(y_SMO))


# In[35]:


get_ipython().run_cell_magic('time', '', 'LogisticRegr_NoIloc(X_SMO, y_SMO, X_test, y_test)')


# In[36]:


get_ipython().run_cell_magic('time', '', 'DecisionTree_NoIloc(X_SMO, y_SMO, X_test, y_test)')


# In[37]:


get_ipython().run_cell_magic('time', '', 'RandomForest_NoIloc(X_SMO, y_SMO, X_test, y_test)')


# In[38]:


get_ipython().run_cell_magic('time', '', 'NN_NoIloc(X_SMO, y_SMO, X_test, y_test)')


# In[39]:


print("Resampling dataset using ADASYN (ADA)...")
ada = ADASYN(random_state=42)
X_ADA, y_ADA = ada.fit_resample(X_train, y_train)
print("X and Y ADA:", len(X_ADA), len(y_ADA))


# In[40]:


get_ipython().run_cell_magic('time', '', 'LogisticRegr_NoIloc(X_ADA, y_ADA, X_test, y_test)')


# In[41]:


get_ipython().run_cell_magic('time', '', 'DecisionTree_NoIloc(X_ADA, y_ADA, X_test, y_test)')


# In[42]:


get_ipython().run_cell_magic('time', '', 'RandomForest_NoIloc(X_ADA, y_ADA, X_test, y_test)')


# In[43]:


get_ipython().run_cell_magic('time', '', 'NN_NoIloc(X_ADA, y_ADA, X_test, y_test)')

