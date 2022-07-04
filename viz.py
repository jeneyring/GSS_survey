#####################################
# This file holds the visualization functions for the Debunk the Zodiac project.
#####################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats



"""This function runs the distribution charts of train data"""
def dist_data1(df):
    train_demo = df[['zodiac','race','age','sex','sexornt','marital','res16','reg16','class']]
    plt.figure(figsize = (9, 13 / 3))
    train_view = train_demo.nunique()
    s = sns.barplot(x = train_view.values, y = train_view.index)
    s.set_xlim(0, 100)
    # s.bar_label(s.containers[0])
    s.set(xlabel = "Demographic # of unique values", ylabel = "column")
    plt.tight_layout()
    plt.show()

def dist_data2(df):
    train_demo2 = df[['income','wrkslf','satjob','occ10','degree','relig','postlifev','postlifenv','sprtprsn','sprtconnct',
                    'sprtlrgr','sprtpurp']]
    plt.figure(figsize = (9, 13 / 3))
    train_view = train_demo2.nunique()
    s = sns.barplot(x = train_view.values, y = train_view.index)
    s.set_xlim(0, 100)
    # s.bar_label(s.containers[0])
    s.set(xlabel = "Demographic # of unique values", ylabel = "column")
    plt.tight_layout()
    plt.show()

def dist_data3(df):
    train_opin = df[['fairv','fairnv','helpfulv','helpfulnv','trustv','trustnv',
                    'conmedic','contv','conpress','consci','conjudge','conlegis','happy',
                    'life','obey','popular','thnkself','workhard','helpoth','grtwrks','freemind'
                   ]]
    # Number of unique values in the training set columns by count
    plt.figure(figsize = (9, 13 / 3))
    train_view5 = train_opin.nunique()
    s = sns.barplot(x = train_view5.values, y = train_view5.index)
    s.set_xlim(0, 100)
    # s.bar_label(s.containers[0])
    s.set(xlabel = "Opinion # of unique values", ylabel = "column")
    plt.tight_layout()
    plt.show()

"""This function gives a visual of the target variable, zodiac"""

def zodiac_cnt(df):
    sns.set(rc={'figure.figsize':(11.7,8.27)})
    sns.barplot(x=df.zodiac.value_counts().index, y=df.zodiac.value_counts())


"""This function runs through each variable giving a variable count """

def variable_count(df):
    columns = df.columns
    for col in columns:
        plt.figure(figsize = (15,  5))
        sns.countplot(data = df, y = col)
        plt.tight_layout()
        plt.show()

"""This function conducts single chi^2 test"""
def chi_square_test(var_one, var_two, alpha = 0.05):
    """"
    Performs a chi squared test on two variables passed
    """
    outputs = []
    observed = pd.crosstab(var_one, var_two)
    chi2, p, degf, expected = stats.chi2_contingency(observed)
    output = {
        'χ^2' : chi2,
        'p' : p,
        'reject_null': p < alpha
    }
    outputs.append(output)
    return pd.DataFrame(outputs)


"""This function shows marital of certain zodiac signs breakdown"""
def mar_viz(df):
    #define dimensions of subplots (rows, columns)
    sns.set(rc={'figure.figsize':(18.7,13.27)})
    fig, axes = plt.subplots(2, 2)
    sns.set_theme(style="whitegrid")

    #create chart in each subplot

    sns.barplot( data= df, x='marital', y="zodiac_capricorn", ax=axes[0,0])
    sns.barplot(data= df, x='marital', y='zodiac_sagittarius', ax=axes[0,1])
    sns.barplot(data= df, x='marital', y='zodiac_aquarius', ax=axes[1,0])
    sns.barplot(data= df, x='marital', y='zodiac_scorpio', ax=axes[1,1])

"""This function shows political views of certain zodiac signs breakdown"""
def pol_viz(df):
    #define dimensions of subplots (rows, columns)
    sns.set(rc={'figure.figsize':(18.7,13.27)})
    fig, axes = plt.subplots(2, 2)
    sns.set_theme(style="whitegrid")

    #create chart in each subplot

    sns.barplot( data= df, x="zodiac_libra", y='polviews', ax=axes[0,0])
    sns.barplot(data= df, x='zodiac_cancer', y='polviews', ax=axes[0,1])
    sns.barplot(data= df, x='zodiac_aries', y='polviews', ax=axes[1,0])
    sns.barplot(data= df, x='zodiac_taurus', y='polviews', ax=axes[1,1])