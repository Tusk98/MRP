import numpy as np 
import pandas as pd 

import matplotlib.pyplot as plt
import matplotlib

from wordcloud import WordCloud
from wordcloud import ImageColorGenerator
from wordcloud import STOPWORDS

import seaborn as sns


# Reading in the data by chunks and putting it into DataFrame named df
counter = 0 
recorder = []

for chunk in pd.read_json("yelp_academic_dataset_review.json", lines=True, chunksize=1000): 
    recorder.append(chunk)
    
    ''' #For smaller sized chunks to test code
    
    if counter == 10: 
        break
    '''
    counter += 1
    if counter % 1000 == 0: 
        print(counter)

df = pd.concat(recorder)


# Bar graph of number of reviews per star
review_count = df.groupby(['stars'], as_index=False).size()
review_count.plot.bar(x=0, y=1, title="Number of reviews by star")
plt.show

# Getting percentages for review pie chart
total_reviews = len(df)
percentage_reviews = [i/total_reviews for i in review_count['size']]
pie_df = pd.DataFrame({'star_reviews': percentage_reviews}, index=['1-star', '2-star', '3-star', '4-star', '5-star'])
percentage_pie = pie_df.plot(kind='pie', title="Percentage of reviews", y='star_reviews', autopct="%.0f%%", legend=False)


# Count of reviews by years  
by_year = df.groupby(df.date.dt.year)['stars'].sum()
by_year.plot.line(x=by_year.keys(), y=1, title="Reviews by year")

# Count of reviews by years  
by_month = df.groupby(df.date.dt.month)['stars'].sum()
by_month.plot.line(x=by_month.keys(), y=1, title="Reviews by month")

# Review length per star 
df['review_length'] = df['text'].apply(len)
sns.set(rc={"figure.figsize":(11.7,8.27)})
length_graph = sns.FacetGrid(data = df, col='stars')
length_graph.map(plt.hist, 'review_length', bins=50)

# Correlation table 
stval = df.groupby('stars').mean()
correlations = stval.corr() 
correlations

text = " ".join(i for i in df['text'])
stopwords = set(STOPWORDS) 
wordcloud = WordCloud(stopwords=stopwords, background_color="white").generate(text)
plt.figure( figsize=(15,10))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()



print("ENDDDDDDDD")






















