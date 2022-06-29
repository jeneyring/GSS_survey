########################################################
# This is a file for the GSS 2021 Survey in relation to participants Zodiac.
# To use this file:
## 1) in a new notebook, type: 'df = acquire_prep()'  , then 'df.head()' to view.
## 2) in same notebook, to split: 'train, validate, test = split(df)'
#imports:
import pandas as pd
from sklearn.model_selection import train_test_split

################################################

"""This function pulls in the 2021 GSS results and columns wanted/mentioned in
    the walkthrough/ReadMe file:"""

def get_gss():
    df = pd.read_stata('gss2021.dta')
    return df

def get_columns(df):
    df = df[['zodiac','born', 'race', 
          'ethnic', 'age', 'sex', 'sexornt', 
          'marital', 'martype','paocc10', 'maocc10', 'res16', 
          'reg16', 'degree',  
          'income', 'wrkslf', 'satjob', 'occ10','partyid', 
          'if16who', 'polviews', 'gunlaw', 'grassv','relidesc', 'relig', 
          'postlifev', 'postlifenv', 'sprtprsn', 'sprtconnct', 'sprtlrgr', 
          'sprtpurp','happy', 'life', 'obey', 'popular', 'thnkself', 
          'workhard', 'helpoth', 'grtwrks', 'freemind', 'decevidc', 'advfmsci',
         'mditate1', 'health', 'hlthphys', 'hlthmntl', 'enjoynat',  
          'eatmeat', 'recycle', 'nobuygrn','fairv', 'fairnv', 'helpfulv', 
          'helpfulnv', 'trustv', 'trustnv', 'conmedic', 'contv', 'conpress', 
          'consci', 'conjudge', 'conlegis', 'socbar', 
          'socrel', 'socommun', 'socfrend', 'satsoc', 'class', 'satfin', 
          'quallife', 'partners', 'partnrs5'
        ]]
    return df

"""This function handles the nulls and dtypes of the 2021 GSS results: """
def wrangle_gss(df):
    df = df.dropna(axis=0, subset=['zodiac'])
    df = df.astype(object)
    df = df.fillna('unknown')
    return df


#### Taking above functions into one:

def acquire_prep():
    df = get_gss()
    df = get_columns(df)
    df = wrangle_gss(df)
    return df

#######################################################
#Train/Test Split:

"""This function splits the dataset into Train, Validate, Test """
def split(df):
    train_and_validate, test = train_test_split(df, random_state=13, test_size=.15)
    train, validate = train_test_split(train_and_validate, random_state=13, test_size=.2)

    print('Train: %d rows, %d cols' % train.shape)
    print('Validate: %d rows, %d cols' % validate.shape)
    print('Test: %d rows, %d cols' % test.shape)

    return train, validate, test

