#!/usr/bin/env python
# coding: utf-8

# In[1]:


#10/01/2021

import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.cluster.util import cosine_distance
import re
import networkx as nx


# In[2]:


df = pd.read_csv(r'C:\Users\italo\OneDrive\Desktop\wos.csv') #creates a dataframe
df['Abstract'] = df['Abstract'].astype(str) # converts abstracts to string
#df = df['Abstract'].notna()
df = df.reset_index(drop=True) # resets indices
df.columns # shows columns
#df.head()


# ## Algorithm for Text Summarization with TextRank

# In[3]:


def read_article(file_name):
    '''This function opens an article and reads it'''
    file = open(file_name, "r")
    filedata = file.readlines() 
    article = filedata[0].split(". ")
    sentences = []
    for sentence in article:
        sentences.append(sentence.replace("[^a-zA-Z]", " ").split(" "))
    sentences.pop() 
    return sentences

def sentence_similarity(sent1, sent2, stopwords=None):
    '''It looks at similarity of sentences'''
    if stopwords is None:
        stopwords = []
    sent1 = [w.lower() for w in sent1]
    sent2 = [w.lower() for w in sent2]
    all_words = list(set(sent1 + sent2))
    vector1 = [0] * len(all_words)
    vector2 = [0] * len(all_words)
    for w in sent1: # build the vector for the first sentence
        if w in stopwords:
            continue
        vector1[all_words.index(w)] += 1
    for w in sent2: # build the vector for the second sentence
        if w in stopwords:
            continue
        vector2[all_words.index(w)] += 1
    return 1 - cosine_distance(vector1, vector2)
 
def build_similarity_matrix(sentences, stop_words):
    '''Creates an empty similarity matrix''' 
    similarity_matrix = np.zeros((len(sentences), len(sentences)))
    for idx1 in range(len(sentences)):
        for idx2 in range(len(sentences)):
            if idx1 == idx2: #ignore if both are same sentences
                continue 
            similarity_matrix[idx1][idx2] = sentence_similarity(sentences[idx1], sentences[idx2], stop_words)

    return similarity_matrix

def generate_summary(file_name, top_n=5):
    '''This is the main function'''
    #file_name = r'C:\Users\italo\OneDrive\Desktop\prova.txt'
    #top_n=2 #number of phrases
    stop_words = stopwords.words('english')
    summarize_text = []
    sentences =  read_article(file_name) # Step 1 - Read text anc split it
    sentence_similarity_martix = build_similarity_matrix(sentences, stop_words) # Step 2 - Generate Similary Martix across sentences
    sentence_similarity_graph = nx.from_numpy_array(sentence_similarity_martix) # Step 3 - Rank sentences in similarity matrix
    scores = nx.pagerank(sentence_similarity_graph) 
    ranked_sentence = sorted(((scores[i],s) for i,s in enumerate(sentences)), reverse=True) # Step 4 - Sort the rank and pick top sentences 
    print("Indexes of top ranked_sentence order are ", ranked_sentence)  
    print('\n')
    for i in range(top_n):
        #print(ranked_sentence[i])
        #print('\n')
        summarize_text.append(" ".join(ranked_sentence[0][1])) #indenta
    print("Summarize Text: \n", ". ".join(summarize_text)) # Outputs the summarize text

generate_summary( r'C:\Users\italo\OneDrive\Desktop\wiki.txt', 2)


# ## Motore di ricerca interattivo

# In[4]:


# fare delle query, ottenere dalla query l'indice e a partire da quello estrarre il summary
# colonne: 'Author Full Names', 'Article Title', 'Abstracts'


# In[5]:


#crea un nuovo df con meno features (ci serviranno per fare le query)
df = df.drop(columns = ['Unnamed: 0', 'Publication Type', 'Authors', 'Book Authors',
       'Book Editors', 'Book Group Authors',
       'Book Author Full Names', 'Group Authors',
       'Source Title', 'Book Series Title', 'Book Series Subtitle', 'Language',
       'Document Type', 'Conference Title', 'Conference Date',
       'Conference Location', 'Conference Sponsor', 'Conference Host',
       'Author Keywords', 'Keywords Plus', 'Addresses',
       'Reprint Addresses', 'Email Addresses', 'Researcher Ids', 'ORCIDs',
       'Funding Orgs', 'Funding Text', 'Cited References',
       'Cited Reference Count', 'Times Cited, WoS Core',
       'Times Cited, All Databases', '180 Day Usage Count',
       'Since 2013 Usage Count', 'Publisher', 'Publisher City',
       'Publisher Address', 'ISSN', 'eISSN', 'ISBN', 'Journal Abbreviation',
       'Journal ISO Abbreviation', 'Publication Date', 'Publication Year',
       'Volume', 'Issue', 'Part Number', 'Supplement', 'Special Issue',
       'Meeting Abstract', 'Start Page', 'End Page', 'Article Number', 'DOI',
       'Book DOI', 'Early Access Date', 'Number of Pages', 'WoS Categories',
       'Research Areas', 'IDS Number', 'UT (Unique WOS ID)', 'Pubmed Id',
       'Open Access Designations', 'Highly Cited Status', 'Hot Paper Status',
       'Date of Export', 'Unnamed: 67'])

df = df[df.Abstract != 'nan'] # removes where there aren't abstracts
#df.head(30:50)


# In[6]:


print(df.iloc[1]['Article Title'])


# In[7]:


df.head(5)


# In[10]:


risposta_1 = input('Vuoi procedere ad una ricerca per autore o per articolo? ')
if risposta_1 == 'Autore':
    df_Autore = df.drop(columns = 'Article Title')
    risposta_2 = input('Che autore stai cercando? Scrivi prima il cognome e poi il nome separati da una virgola ') 
    a = df_Autore.loc[df['Author Full Names'] == risposta_2]
    index = a.index
    #print(index)
    t = str(df.loc[index, 'Abstract'].tolist())
    #print(t[0])
    #t.replace(t[0], '')
    #print(t)
    f = open(r'C:\Users\italo\OneDrive\Desktop\ciaone.txt', 'w')
    f.write(t)
    f.close()
    generate_summary(r'C:\Users\italo\OneDrive\Desktop\ciaone.txt', 2) #Ponga, Mauricio
    f = open(r'C:\Users\italo\OneDrive\Desktop\ciaone.txt', 'r')
    print('\n')
    print('Original Abstract:')
    print(f.readline())
    
else:
    df_Articolo = df.drop(columns = 'Author Full Names')
    risposta_2 = input('Che articolo stai cercando? ') 
    b = df_Articolo.loc[df['Article Title'] == risposta_2]
    index = b.index
    #print(index)
    t = str(df.loc[index, 'Abstract'].tolist())
    #print(t)
    #print(t[0])
    #t.replace(t[0], '')
    #print(t)
    f = open(r'C:\Users\italo\OneDrive\Desktop\ciaonee.txt', 'w')
    f.write(t)
    f.close()
    generate_summary(r'C:\Users\italo\OneDrive\Desktop\ciaonee.txt', 2) 
    f = open(r'C:\Users\italo\OneDrive\Desktop\ciaone.txt', 'r')
    print('\n')
    print('Original Abstract:')
    print(f.readline())

