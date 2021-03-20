#!/usr/bin/env python
# coding: utf-8

# In[1]:


#09/01/2020

import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import re
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from matplotlib import pyplot as plt


# In[43]:


df = pd.read_csv(r'C:\Users\italo\OneDrive\Desktop\wos.csv') #creates a dataframe


# In[44]:


stop_words = set(stopwords.words('english'))
##Creating a list of custom stopwords
new_words = ["fig","figure","image","sample","using", 
             "show", "result", "large", 
             "also", "one", "two", "three", 
             "four", "five", 'six', "seven","eight","nine",
             "covid", "health", "cov", "sars"]
stop_words = list(stop_words.union(new_words))


# In[45]:


print(type(df['Abstract'][0]))
print(df['Abstract'][0])


# In[46]:


abstract_df = df.iloc[: , [22]]
print(abstract_df)


# In[47]:


abstract_df = abstract_df[abstract_df['Abstract'].notna()]
print(abstract_df) #800 circa in meno


# In[48]:


abstract_df['Abstract'] = abstract_df['Abstract'].astype(str) 


# In[49]:


print(abstract_df['Abstract'])


# In[9]:


def pre_process(text):
    '''This function preprocesses a text'''
    text=text.lower() # lowercase
    text=re.sub("&lt;/?.*?&gt;"," &lt;&gt; ",text) #remove tags
    text=re.sub("(\\d|\\W)+"," ",text) # remove special characters and digits
    text = text.split() #Convert to list from string
    text = [word for word in text if word not in stop_words] # remove stopwords
    text = [word for word in text if len(word) >= 3] # remove words less than three letters
    lmtzr = WordNetLemmatizer() # lemmatize
    text = [lmtzr.lemmatize(word) for word in text]
    return ' '.join(text)

docs = abstract_df['Abstract'].apply(lambda x:pre_process(x))


# In[62]:


print(docs) #proof
#docs.columns


# In[11]:


len(docs)


# In[12]:


abstract_df = abstract_df.reset_index(drop=True)


# In[13]:


docs = docs.reset_index(drop=True)


# In[14]:


docs


# In[15]:


abstract_df


# ### TF-IDF stands for Text Frequency Inverse Document Frequency. The importance of each word increases in proportion to the number of times a word appears in the document (Text Frequency – TF) but is offset by the frequency of the word in the corpus (Inverse Document Frequency – IDF).

# In[16]:


cv=CountVectorizer(max_df=0.95,         # ignore words that appear in 95% of documents
                   max_features=10000,  # the size of the vocabulary
                   ngram_range=(1,3)    # vocabulary contains single words, bigrams, trigrams
                  )
word_count_vector=cv.fit_transform(docs)


# ### Now I’m going to use the TfidfTransformer in Scikit-learn to calculate the reverse frequency of documents:

# In[17]:


tfidf_transformer=TfidfTransformer(smooth_idf=True,use_idf=True)
tfidf_transformer.fit(word_count_vector)


# ### Now, we are ready for the final step. In this step, I will create a function for the task of Keyword Extraction with Python by using the Tf-IDF vectorization:

# In[18]:


def sort_coo(coo_matrix):
    tuples = zip(coo_matrix.col, coo_matrix.data)
    return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)

def extract_topn_from_vector(feature_names, sorted_items, topn=10):
    """get the feature names and tf-idf score of top n items"""
    
    #use only topn items from vector
    sorted_items = sorted_items[:topn]

    score_vals = []
    feature_vals = []

    for idx, score in sorted_items:
        fname = feature_names[idx]
        
        #keep track of feature name and its corresponding score
        score_vals.append(round(score, 3))
        feature_vals.append(feature_names[idx])

    #create a tuples of feature,score
    #results = zip(feature_vals,score_vals)
    results= {}
    for idx in range(len(feature_vals)):
        results[feature_vals[idx]]=score_vals[idx]
    
    return results

# get feature names
feature_names=cv.get_feature_names()

def get_keywords(idx, docs):
    tf_idf_vector=tfidf_transformer.transform(cv.transform([docs[idx]])) #generate tf-idf for the given document
    sorted_items=sort_coo(tf_idf_vector.tocoo()) #sort the tf-idf vectors by descending order of scores
    keywords=extract_topn_from_vector(feature_names,sorted_items,10) #extract only the top n; n here is 10
    #print(type(keywords))
    return keywords
    

def print_results(idx,keywords, df):
    # now print the results
    print("\n=====Abstract=====")
    print(abstract_df['Abstract'][idx])
    print("\n===Keywords===")
    for k in keywords:
        print(k,keywords[k])
        #print(keywords)
    
idx=0
keywords=get_keywords(idx, docs)
print_results(idx,keywords, df)
#listone(keywords)
#print(keywords.keys()[0])


# In[19]:


def get_all_keywords(idx, docs):
    '''This function gets all principal keywords'''
    tf_idf_vector=tfidf_transformer.transform(cv.transform([docs[idx]])) #generate tf-idf for the documents
    sorted_items=sort_coo(tf_idf_vector.tocoo()) #sort the tf-idf vectors by descending order of scores
    keywords=extract_topn_from_vector(feature_names,sorted_items,1) #extract only the first
    #print(type(keywords))
    return keywords


# In[20]:


def principal_k_list(docs):
    '''It returns list of principal keywords'''
    lst = []
    for idx in range(len(docs)):
        keywords=get_keywords(idx, docs)
        lst.append(list(keywords.keys())[0])
        #lst.append(keywords)
    return lst


# In[21]:


K_list = principal_k_list(docs)


# In[22]:


def principal_k_dict(generic_list):
    '''It creates a dictionary with occurencies of every word in a list'''
    string = str(generic_list)
    d = {}
    words = string.replace(',','').replace("'",'').replace("[",'').split()
    for word in words:
        d[word] = d.get(word, 0) + 1
    return d


# In[23]:


K_dict = principal_k_dict(K_list)
#print(dizionario)


# In[24]:


def orderer(dictionary):
    '''It sorts a dictionary'''
    return sorted(dictionary.items(), key = lambda x:x[1], reverse = True)
    
final_K_dict = orderer(K_dict)


# In[25]:


X=[final_K_dict[i] [0] for i in range(20)]
Y=[final_K_dict[i] [1] for i in range(20)]

fig = plt.figure(figsize=(15, 7))
ax = fig.add_subplot(111)
ax.bar(X, Y)
plt.title('Twelve most frequent Keyword')
plt.xlabel('Words')
plt.ylabel('Frequencies')
plt.xticks(rotation=30)
#plt.tight_layout(pad=0.005)
plt.savefig('Twelve most frequent Keyword')
plt.show()


# ## Wordclouds

# In[26]:


from wordcloud import WordCloud, STOPWORDS


# In[33]:


#df['Abstract'] = df['Abstract'].astype('|S80')
#df['Abstract'].dtypes


# In[63]:


lst = []
for rows in docs:
    lst.append(rows)


# In[64]:


lst


# In[77]:


# Define a function to plot word cloud
import numpy as np
from PIL import Image

def plot_cloud(wordcloud):
    # Set figure size
    plt.figure(figsize=(40, 30))
    # Display image
    plt.imshow(wordcloud) 
    # No axis details
    plt.axis("off")
    plt.title('Word Cloud with a """Covid 19""" shape')
    plt.savefig('Word Cloud')
    plt.show;

mask = np.array(Image.open(r'C:\Users\italo\OneDrive\Desktop\covid_19.png'))
wordcloud = WordCloud(width = 1919, height = 2400, random_state=1, collocations=False, background_color='white', stopwords = STOPWORDS, mask = mask).generate(str(lst))
plot_cloud(wordcloud)


# In[ ]:




