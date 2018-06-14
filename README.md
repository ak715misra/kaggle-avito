# Machine Learning Engineer Nanodegree
# Capstone Project
## Project: Predicting deal probability for Avito advertisements

### Install

This project requires **Python** and the following Python libraries installed:

- [NumPy](http://www.numpy.org/)
- [Pandas](http://pandas.pydata.org/)
- [matplotlib](http://matplotlib.org/)
- [scikit-learn](http://scikit-learn.org/stable/)

You will also need to have software installed to run and execute a [Jupyter Notebook](http://ipython.org/notebook.html)

If you do not have Python installed yet, it is highly recommended that you install the [Anaconda](http://continuum.io/downloads) distribution of Python, which already has the above packages and more included. 

In addition, this project makes use of plotly and xgboost libraries. You may download them by using commands like:
pip install plotly
conda -c anaconda install xgboost

### Code

Code is provided in the `avito.ipynb` notebook file. You will also be required to use the `train.csv`, 'test.csv', 'periods_train.csv', and 'periods_test.csv' dataset files from Avito to complete the task.

### Run

In a terminal or command window, navigate to the top-level project directory `kaggle-avito/` (that contains this README) and run one of the following commands:

```bash
ipython notebook avito.ipynb
```  
or
```bash
jupyter notebook avito.ipynb
```
or 
```bash
python avito.py
```

This will open the Jupyter Notebook software and project file in your browser.

### Data

The Avito data is broken into train and test datasets. The train datasets are train.csv, periods_train.csv, train_active.csv, while test datasets are test.csv, periods_test.csv, test_active.csv. In addition, the ads images are provided in train_jpg.zip. There are more than 1.5 million data points, with each datapoint having 18 features. Due to large size of the datasets (~ 9 Gigabytes), we are using only train, test, periods_train and periods_test datasets for this project. These datasets can be found on Kaggle site (https://www.kaggle.com/c/avito-demand-prediction/data).

**Features**
1.	`item_id`: Ad id
2.	`user_id`: User id
3.	`region`: Ad region
4.	`city`: Ad city
5.	`parent_category_name`: Top level ad category as classified by Avito's ad model
6.	`category_name`: Fine grain ad category as classified by Avito's ad model
7.	`param_1`: Optional parameter from Avito's ad model
8.	`param_2`: Optional parameter from Avito's ad model
9.	`param_3`: Optional parameter from Avito's ad model
10.	`title`: Ad title
11.	`description`: Ad description
12.	`price`: Ad price
13.	`item_seq_number`: Ad sequential number for user
14.	`activation_date`: Date ad was placed
15.	`user_type`: User type
16.	`image`: Id code of image. Ties to a jpg file in train_jpg. Not every ad has an image
17.	`image_top_1`: Avito's classification code for the image


**Target Variable**
18. `deal_probability`: likelihood that an ad actually sold something