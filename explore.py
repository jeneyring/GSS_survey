import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

##########################################################
# This file holds functions for the EDA (exploration) phase of the 
# Debunking the Zodiac model project
###################################################

"""This function loops through each variable for a viz of value_counts"""

def variable_count(df):
    columns = df.columns
    for col in columns:
        plt.figure(figsize = (15,  5))
        sns.countplot(data = df, y = col)
        plt.tight_layout()
        plt.show()

"""This get_dummies and encodes the zodiac column per zodiac sign and then concats
it back to the original df"""
def dummy_zodiac(df):
    # get the dummies and store it in a variable
    dummy_df2 = pd.get_dummies(df[['zodiac']], dummy_na=False, drop_first= False, dtype=int)
    df = pd.concat([df, dummy_df2], axis=1)
    return df


"""This function creates a column that holds zodiac birthdates and concats it back
to the original df"""

def zodiac_dates(row):  
    if row['zodiac'] == 'libra':
        return 'Sept 22-Oct 23'
    elif row['zodiac'] == 'aries':
        return 'March 21-April 19'
    elif row['zodiac'] == 'taurus':
        return 'April 20-May 20'
    elif row['zodiac'] == 'gemini':
        return 'May 21-June 21'
    elif row['zodiac'] == 'cancer':
        return 'June 22-July 22'
    elif row['zodiac'] == 'leo':
        return 'July 23-Aug 22'
    elif row['zodiac'] == 'virgo':
        return 'Aug 23-Sept 22'
    elif row['zodiac'] == 'scorpio':
        return 'Oct 23-Nov 21'
    elif row['zodiac'] == 'sagittarius':
        return 'Nov 22-Dec 21'
    elif row['zodiac'] == 'capricorn':
        return 'Dec 22-Jan 19'
    elif row['zodiac'] == 'aquarius':
        return 'Jan 20-Feb 18'
    return 'Feb 19-March 20'
    
#adding this row with the date function:
def add_zdates(df):
    df['zodiac_dates'] = df.apply(lambda row: zodiac_dates(row), axis=1)
    return df

#function (made by Stephen Fitzsimon), to run variables against target by zodiac sub categories
def chi_square_mass_test(df, cat_cols, target_col = 'target_outcome', alpha=0.05):
    """
    Performs a chi square test for all the aubcategories pass to cat_cols against the
    target_col
    """
    outputs = []
    for cat in cat_cols:
        for subcat in list(df[cat].unique()):
            for target_col_subcat in list(df[target_col].unique()):
                #get the crosstab between the two variables
                observed = pd.crosstab(df[target_col]==target_col_subcat, df[cat]==subcat)
                #calculate the statistic
                chi2, p, degf, expected = stats.chi2_contingency(observed)
                #save the calculation
                output = {
                        'target_column':target_col,
                        'column':cat,
                        'target_col_subcat':target_col_subcat,
                        'column_subcat':subcat,
                        'null_hypothesis':f"{target_col_subcat} independent of {subcat}",
                        'chi2':chi2,
                        'p':p,
                        'reject_null':p < alpha
                }
                outputs.append(output)
    #return the dataframe
    return pd.DataFrame(outputs)

