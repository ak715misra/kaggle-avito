
# coding: utf-8

# Load all the libraries necessary for the project 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD

import plotly.offline as py
py.init_notebook_mode(connected=True)
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.offline as offline
offline.init_notebook_mode()
import plotly.tools as tls
color = sns.color_palette()
from sklearn import preprocessing, model_selection, metrics
#import lightgbm as lgb

# Results are saved as output.

# Read the necessary data
print('Reading Data ...')
train = pd.read_csv("train.csv", parse_dates=["activation_date"])
print('training data size: ', train.shape)
test = pd.read_csv("test.csv", parse_dates=["activation_date"])
print('test data size: ', test.shape)
periods_train = pd.read_csv("periods_train.csv", parse_dates=["activation_date", "date_from", "date_to"])
print('periods_train data size: ', periods_train.shape)
periods_test = pd.read_csv("periods_test.csv", parse_dates=["activation_date", "date_from", "date_to"])
print('periods_test data size: ', periods_test.shape)
print('Finished Reading Data ...')

# Check the data in train dataset
train.head()

# test dataset should not have deal probability column
test.head()

# Also check periods_train dataset
periods_train.head()

# Training dataset overview
train.info()

train.describe()

# Use df.isnull.sum() to get the count of missing values in each column of df.
# Use df.isnull.count() to get the count of rows for each column in df 
total = train.isnull().sum().sort_values(ascending=False)
percent = (train.isnull().sum()/train.isnull().count()*100).sort_values(ascending=False)
missing_train_data = pd.concat([total, percent], axis=1, keys=["Total", "Percent"])
missing_train_data.head(10)

# Also check if we have missing values in Periods_train dataset
total = periods_train.isnull().sum().sort_values(ascending=False)
percent = (periods_train.isnull().sum()/periods_train.isnull().count()*100).sort_values(ascending=False)
missing_train_data = pd.concat([total, percent], axis=1, keys=["Total", "Percent"])
missing_train_data.head()

# data exploration - let us explore some of the data in the dataset
# deal probability is our target variable with float value between 0 and 1
# histogram and distribution of deal_probability
plt.figure(figsize = (12, 8))
sns.distplot(train['deal_probability'], bins=100, kde=False)
plt.xlabel('likelihood that ad actually sold something', fontsize=12)
plt.title('Histogram of likelihood that ad sold something')
plt.show()

plt.figure(figsize = (12, 8))
plt.scatter(range(train.shape[0]), np.sort(train.deal_probability.values))
plt.ylabel('likelihood that ad actually sold something', fontsize=12)
plt.title('Distribution of likelihood that ad sold something')
plt.show()


# The plots show that almost 1000000 ads had a probability of 0 (means sold nothing), while few had a probability of 1, and the rest were in the middle.

# Check the distribution of non-zero deal_probability
train['deal_class'] = train['deal_probability'].apply(lambda x: ">=0.5" if x >= 0.5 else "<0.5")
temp = train['deal_class'].value_counts()
labels = temp.index
sizes = (temp / temp.sum()) * 100
trace = go.Pie(labels=labels, values=sizes, hoverinfo='label+percent')
layout = go.Layout(title='Distribution of deal class')
data = [trace]
fig = go.Figure(data=data, layout=layout)
py.iplot(fig)

del train['deal_class']


# 88% of the non-zero deal_probability ads have greater than 50% chances of selling something, while 12% have less than 50% chances.

# convert the Russian columns into English using Yandex Translator and merge into train dataset
from io import StringIO
temp_region = StringIO("""
region,region_en
Свердловская область, Sverdlovsk oblast
Самарская область, Samara oblast
Ростовская область, Rostov oblast
Татарстан, Tatarstan
Волгоградская область, Volgograd oblast
Нижегородская область, Nizhny Novgorod oblast
Пермский край, Perm Krai
Оренбургская область, Orenburg oblast
Ханты-Мансийский АО, Khanty-Mansi Autonomous Okrug
Тюменская область, Tyumen oblast
Башкортостан, Bashkortostan
Краснодарский край, Krasnodar Krai
Новосибирская область, Novosibirsk oblast
Омская область, Omsk oblast
Белгородская область, Belgorod oblast
Челябинская область, Chelyabinsk oblast
Воронежская область, Voronezh oblast
Кемеровская область, Kemerovo oblast
Саратовская область, Saratov oblast
Владимирская область, Vladimir oblast
Калининградская область, Kaliningrad oblast
Красноярский край, Krasnoyarsk Krai
Ярославская область, Yaroslavl oblast
Удмуртия, Udmurtia
Алтайский край, Altai Krai
Иркутская область, Irkutsk oblast
Ставропольский край, Stavropol Krai
Тульская область, Tula oblast
""")
region_df = pd.read_csv(temp_region)
train = pd.merge(train, region_df, how="left", on="region")

# Use Pie Charts to plot the regional distribution of ads
regions_in_eng = train['region_en'].value_counts()
labels = regions_in_eng.index
sizes = (regions_in_eng/regions_in_eng.sum())*100
trace = go.Pie(labels=labels, values=sizes)
layout = go.Layout(title="Region Distribution", width=900, height=900)
data = [trace]
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename="region")

# plot deal_probability by the regions
plt.figure(figsize=(12,8))
sns.boxplot(y='region_en', x='deal_probability', data=train)
plt.xlabel('deal probability', fontsize=12)
plt.ylabel('region', fontsize=12)
plt.title('Deal Probability by Region')
plt.xticks(rotation='vertical')
plt.show()

# The top 3 regions are: Kresnodar Krai, Sverdlovsk oblast, and Rostov oblast

# Convert parent_category_name from Russian to English
temp_parent_category = StringIO("""
parent_category_name,parent_category_name_en
Личные вещи,Personal belongings
Для дома и дачи,For the home and garden
Бытовая электроника,Consumer electronics
Недвижимость,Real estate
Хобби и отдых,Hobbies & leisure
Транспорт,Transport
Услуги,Services
Животные,Animals
Для бизнеса,For business
""")

temp_df = pd.read_csv(temp_parent_category)
train = pd.merge(train, temp_df, on="parent_category_name", how="left")

# Check parent category name distribution
temp_parent_category_colmn = train['parent_category_name_en'].value_counts()
labels = (np.array(temp_parent_category_colmn.index))
sizes = (np.array((temp_parent_category_colmn / temp_parent_category_colmn.sum())*100))

trace = go.Pie(labels=labels, values=sizes)
layout = go.Layout(
    title='Parent Category distribution',
    width=900,
    height=900,
)
data = [trace]
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename="parentcategory")

plt.figure(figsize=(12,8))
sns.boxplot(x="parent_category_name_en", y="deal_probability", data=train)
plt.ylabel('Deal probability', fontsize=12)
plt.xlabel('Parent Category', fontsize=12)
plt.title("Deal probability by parent category", fontsize=14)
plt.xticks(rotation='vertical')
plt.show()


# 46.4% of the ads are for Personal belongings, 11.9% are for home and garden and 11.5% for consumer electronics.

# Consider distribution of category name after converting category_names from Russian to English
temp_category_name = StringIO("""
category_name,category_name_en
"Одежда, обувь, аксессуары","Clothing, shoes, accessories"
Детская одежда и обувь,Children's clothing and shoes
Товары для детей и игрушки,Children's products and toys
Квартиры,Apartments
Телефоны,Phones
Мебель и интерьер,Furniture and interior
Предложение услуг,Offer services
Автомобили,Cars
Ремонт и строительство,Repair and construction
Бытовая техника,Appliances
Товары для компьютера,Products for computer
"Дома, дачи, коттеджи","Houses, villas, cottages"
Красота и здоровье,Health and beauty
Аудио и видео,Audio and video
Спорт и отдых,Sports and recreation
Коллекционирование,Collecting
Оборудование для бизнеса,Equipment for business
Земельные участки,Land
Часы и украшения,Watches and jewelry
Книги и журналы,Books and magazines
Собаки,Dogs
"Игры, приставки и программы","Games, consoles and software"
Другие животные,Other animals
Велосипеды,Bikes
Ноутбуки,Laptops
Кошки,Cats
Грузовики и спецтехника,Trucks and buses
Посуда и товары для кухни,Tableware and goods for kitchen
Растения,Plants
Планшеты и электронные книги,Tablets and e-books
Товары для животных,Pet products
Комнаты,Room
Фототехника,Photo
Коммерческая недвижимость,Commercial property
Гаражи и машиноместа,Garages and Parking spaces
Музыкальные инструменты,Musical instruments
Оргтехника и расходники,Office equipment and consumables
Птицы,Birds
Продукты питания,Food
Мотоциклы и мототехника,Motorcycles and bikes
Настольные компьютеры,Desktop computers
Аквариум,Aquarium
Охота и рыбалка,Hunting and fishing
Билеты и путешествия,Tickets and travel
Водный транспорт,Water transport
Готовый бизнес,Ready business
Недвижимость за рубежом,Property abroad
""")

temp_df = pd.read_csv(temp_category_name)
train = pd.merge(train, temp_df, on="category_name", how="left")

category_name_cnt = train['category_name_en'].value_counts()
trace = go.Bar(
    y=category_name_cnt.index[::-1],
    x=category_name_cnt.values[::-1],
    orientation = 'h',
    marker=dict(
        color=category_name_cnt.values[::-1],
        colorscale = 'Blues',
        reversescale = True
    ),
)

layout = dict(
    title='Category Name of Ads - Count',
    height=900
    )
data = [trace]
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename="category name")


# Top 3 categories are:
# 
#     1. Clothes, shoes, accessories
#     2. Children's clothing and footwear
#     3. Goods for children and toys
# 

# Fill the missing values in price column with Mean and check the distribution
train["price_new"] = train["price"].values
train["price_new"].fillna(np.nanmean(train["price"].values), inplace=True)

plt.figure(figsize=(12,8))
sns.distplot(np.log1p(train["price_new"].values), bins=100, kde=False)
plt.xlabel('Log of price', fontsize=12)
plt.title("Log of Price Histogram", fontsize=14)
plt.show()

# Check the number of words in the title
train["title_nwords"] = train["title"].apply(lambda x: len(x.split()))
test["title_nwords"] = test["title"].apply(lambda x: len(x.split()))

title_words_cnt = train['title_nwords'].value_counts()

trace = go.Bar(
    x=title_words_cnt.index,
    y=title_words_cnt.values,
    marker=dict(
        color="blue",
        reversescale = True
    ),
)

layout = go.Layout(
    title='Number of words in title column'
)

data = [trace]
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename="title_nwords") 


# Mostly titles have 2 or 3 words.

# Check the variability in the title column by taking the TF-IDF of the title, getting the top SVD of this TF-IDF and then plotting the distribution of SVD components with deal_probability

### TFIDF Vectorizer ###
tfidf_vec = TfidfVectorizer(ngram_range=(1,1))
full_tfidf = tfidf_vec.fit_transform(train['title'].values.tolist() + test['title'].values.tolist())
train_tfidf = tfidf_vec.transform(train['title'].values.tolist())
test_tfidf = tfidf_vec.transform(test['title'].values.tolist())

### SVD Components ###
n_comp = 3
svd_obj = TruncatedSVD(n_components=n_comp, algorithm='arpack')
svd_obj.fit(full_tfidf)
train_svd = pd.DataFrame(svd_obj.transform(train_tfidf))
test_svd = pd.DataFrame(svd_obj.transform(test_tfidf))
train_svd.columns = ['svd_title_'+str(i+1) for i in range(n_comp)]
test_svd.columns = ['svd_title_'+str(i+1) for i in range(n_comp)]
train = pd.concat([train, train_svd], axis=1)
test = pd.concat([test, test_svd], axis=1)
del full_tfidf, train_tfidf, test_tfidf, train_svd, test_svd

# 1st svd comp #
plt.figure(figsize=(8,8))
sns.jointplot(x=train["svd_title_1"].values, y=train["deal_probability"].values, size=10)
plt.ylabel('Deal Probability', fontsize=12)
plt.xlabel('First SVD component on Title', fontsize=12)
plt.title("Deal Probability distribution for First SVD component on title", fontsize=15)
plt.show()

# 2nd svd comp #
plt.figure(figsize=(8,8))
sns.jointplot(x=train["svd_title_2"].values, y=train["deal_probability"].values, size=10)
plt.ylabel('Deal Probability', fontsize=12)
plt.xlabel('Second SVD component on Title', fontsize=12)
plt.title("Deal Probability distribution for Second SVD component on title", fontsize=15)
plt.show()

# 3rd svd comp #
plt.figure(figsize=(8,8))
sns.jointplot(x=train["svd_title_3"].values, y=train["deal_probability"].values, size=10)
plt.ylabel('Deal Probability', fontsize=12)
plt.xlabel('Third SVD component on Title', fontsize=12)
plt.title("Deal Probability distribution for Third SVD component on title", fontsize=15)
plt.show()


# Plot the number of words in description column after filling the missing values

## Filling missing values ##
train["description"].fillna("NA", inplace=True)
test["description"].fillna("NA", inplace=True)

train["desc_nwords"] = train["description"].apply(lambda x: len(x.split()))
test["desc_nwords"] = test["description"].apply(lambda x: len(x.split()))

desc_words_cnt = train['desc_nwords'].value_counts().head(100)

trace = go.Bar(
    x=desc_words_cnt.index,
    y=desc_words_cnt.values,
    marker=dict(
        color="blue",
        reversescale = True
    ),
)

layout = go.Layout(
    title='Number of words in Description column'
)

data = [trace]
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename="desc_nwords")  


# In order to build the model, we will :
# 
#     Create a new feature for week day
#     Label encode the categorical variables
#     Drop the un-needed columns.
# 

# Target and ID variable
train_y = train['deal_probability'].values
test_id = test['item_id'].values

# New variable of weekday
train['activation_weekday'] = train['activation_date'].dt.weekday
test['activation_weekday'] = test['activation_date'].dt.weekday

# Label encode the categorical variables
cat_vars = ["region", "city", "parent_category_name", "category_name", "user_type", "param_1", "param_2", "param_3"]
for col in cat_vars:
    lbl = preprocessing.LabelEncoder()
    lbl.fit(list(train[col].values.astype('str')) + list(test[col].values.astype('str')))
    train[col] = lbl.transform(list(train[col].values.astype('str')))
    test[col] = lbl.transform(list(test[col].values.astype('str')))

cols_to_drop = ['item_id', 'user_id', 'title', 'description', 'activation_date', 'image']
train_X = train.drop(cols_to_drop + ["region_en", "parent_category_name_en", "category_name_en", "price_new", "deal_probability"], axis=1)
test_X = test.drop(cols_to_drop, axis=1)

# Trying to implement XGBoost Model
import xgboost as xgb
dev_X = train_X.iloc[:-200000,:]
val_X = train_X.iloc[-200000:,:]
dev_y = train_y[:-200000]
val_y = train_y[-200000:]
print(dev_X.shape, val_X.shape, test_X.shape)

params = {'eta': 0.3,
          'tree_method': "hist",
          'grow_policy': "lossguide",
          'max_leaves': 1400,  
          'max_depth': 0, 
          'subsample': 0.9, 
          'colsample_bytree': 0.7, 
          'colsample_bylevel':0.7,
          'min_child_weight':0,
          'alpha':4,
          'objective': 'reg:logistic', 
          'eval_metric': 'rmse', 
          'random_state': 99, 
          'silent': True}

tr_data = xgb.DMatrix(dev_X, dev_y)
va_data = xgb.DMatrix(val_X, val_y)

eval_list = [(tr_data, 'train'), (va_data, 'valid')]
num_boosting_itr = 1000
model = xgb.train(params, tr_data, num_boosting_itr, eval_list, maximize=False, early_stopping_rounds = 25, verbose_eval=5)

# Top features
from xgboost import plot_importance
fig, ax = plt.subplots(figsize=(12,18))
plot_importance(model, max_num_features=50, height=0.8, ax=ax)
ax.grid(False)
plt.title("XGB - Feature Importance", fontsize=15)
plt.show()
plt.gcf().savefig('feature_importance_xgb.png')


# Price seems to be the most important feature, followed by city, and title.

# We will try to improve the model by:
# 
#     Adding more new features
#     Tuning the parameters as these are not tuned
#     Blending with other models.
# 
