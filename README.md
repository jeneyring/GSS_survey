# Zodiac Prediction Model using the GSS 2021 survey
## About the Project:

The GSSurvey asks US adults questions related to current demographics, opinions, religious/spiritual beliefs, political and life view points, social class opinions, as well even recording peoples zodiac signs.

This project is an interest project, taking the GSS participant responses and comparing their zodiac responses to their other survey answers to see if a Classification Model can accurately predict their Zodiac sign/birth month.

### Project Goals
The goals of this project are to build a Classification Model that can beat the Baseline model of 10% accuracy prediction in predicting a person's zodiac sign when answering the GSSurvey questions. 

This project goes through the full Data Science pipeline in acquiring the 2021 GSS survey responses, cleaning and preparing this dataset, exploring and conducting statistical chi^2 tests for 6 different hypothesis questions, and then conducting 5 different Classification Models on the found main drivers.

### Background Data information:
ABOUT THE GSS:
Since 1972, the General Social Survey (GSS) has been conducted annually by the Univeristy of Chicago to measure the responses of contemporary American Society--through the lens of adult survey participants. This project uses the <a href="https://gss.norc.org/get-the-data " title="GSS 2021">GSS 2021</a> dataset.

As the GSS 2021 is a general survey of each participant's response on their beliefs, opinions, demographics, and relationships, this model will be using the Zodiac responses as a general way to see if one's sign can actually be predicted by certain responses, and if a model could actually be built on this.

The columns selected pulled from the GSS data in this project where done so by targeting key points of interest that are often covered in astrology:

- Career
- Relationships
- Opinions
- Health (physical and mental)
- Political stance/view points
- Religion/Spirituality
- Interests
and Overall outlook of others and life


ABOUT THE ZODIAC SIGNS:
As many viewpoints on Astrology overall seem to be changing, oftentimes in terms of self-identity and decision factors, this project is taking a look at zodiac signs (using participants Sun Sign) to determine if the responses of the US adult partipants of the GSS actually correlates to the astrological sun signs they provided and align with.

### Data Dictionary:
Some of the columns and datasets may not fully makes sense to those first meeting this data. Below is a table that helps define the features and terminology used in this project.

|Target|Datatype|Definition|
|:-------|:--------|:----------|
| GSS 2021 | 4032 non-null: Category | General Social Survey of 2021 |

|Feature|Datatype|Definition|
|:-------|:--------|:----------|
| zodiac          |  4032 non-null: Category | participant's zodiac birth sign |
| sexornt         |  4032 non-null: Category  | participant's sexual orientation|
| marital   | 4032 non-null: Category | participant's marital status|
| res16       | 4032 non-null: Category | participant's region of living |
| degree   |  4032 non-null: Category | participant's highest earned education degree|
| class |  4032 non-null: Category | participant's self rated social class|
| relig |  4032 non-null: Category | participant's religion|
| postlifev |  4032 non-null: Category | participant's view of life after death|
| sprtprsn |  4032 non-null: Category | participant's self rating of being a spiritual person|
| fairv |  4032 non-null: Category | participant's view of others being fair|
| trustv |  4032 non-null: Category | participant's view of others being trustworthy|
| conmedic |  4032 non-null: Category | participant's trust of medical system|
| happy |  4032 non-null: Category | participant's overall happiness with life|
| workhard |  4032 non-null: Category | participant's 1-5 rating of most important life lesson of working hard|
| occ10 |  4032 non-null: Category | participant's occupation trade|
| hlthmntl |  4032 non-null: Category | participant's self rating of their mental health|
| socrel |  4032 non-null: Category | participant's rating of how often they spend time with relatives/family|
| decevidc |  4032 non-null: Category | participant's rating of how much they agree of needing sufficient evidence for decisions|
| polviews |  4032 non-null: Category | participant's rating of their political viewpoints|

# Sample of Hypothesis' Questions asked:
For the key features in comparison as categorical features of zodiac predictors.
### Question: Relationship/marital status can help classify a person's zodiac?
- H_0: Marital status is independant of zodiac sign.
- H_a: Marital status is dependant of zodiac sign.

### Question: Opinions on working hard in life can help classify a person's zodiac?
- H_0: workhard opinion is independant of zodiac sign.
- H_a: workhard opinion is dependant of zodiac sign.

### Question: Political viewpoints can help classify a person's zodiac?
- H_0: Political viewpoints is independant of zodiac sign.
- H_a: Political viewpoints is dependant of zodiac sign.

# Project Steps:

## Acquire
Within the wrangle.py file there are functions that will help to:
- Pull in the 2017, Single Family Property residents with no duplicates
- Read the SQL acquire query to a csv file
- Assign the data to a variable

## Prepare
Within the wrangle.py file there are functions for the prepare stage that:
- Clean up the nulls by dropping all from zodiac column and creating unanswered columns as 'unknown' to keep data that would otherwise be lost.
- Drop any unneccessary and duplicated columns
- Resets dtypes to Objects and int 
- Splits data into train, validate and test

## Explore
Within the explore.ipynb file, there is a step-by-step breakdown of my Exploration methods.
Main features found and mentioned are:
- workhard: how participants rate (1-5) 'working hard' as a key life lesson they would teach a child.
- hlthmntl: each participant's self rating of their mental health
- socrel: how often participants spend time with relatives/family.
- decevidc: how participants rate needing sufficient evidence in making decisions

## Model
The model.ipnyb notebook holds each method of model used in this project. Models include steps to produce: Decision Tree Classifier| Random Forest| and 5 different KNN Models (with varying n_number of nearest neighbors)

The function for the final chosen model can be found in model.py and Final Report.

## Conclusion
Using the best Classification model of Random Forest, the results did not beat the baseline prediction of 10%. 

Moving forward, I would want to try creating cluster features and utilize feature engineering like SelectKBest to find better predictors for this model. 

I would also want to add past GSS responses to add in more data and see if past years might help with adding to predictions.