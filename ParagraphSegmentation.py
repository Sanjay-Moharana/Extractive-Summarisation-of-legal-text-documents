
# coding: utf-8

# In[27]:


import json
import ijson
import gc
import pandas


# In[28]:


# Reading the DF Corpus
# with open('TFCorpus.json', 'r') as fp:
#     readTF = json.load(fp)
# fp.close()
# gc.collect()



# fileName = '1953_24.txt'
# readTFFile = ijson.parse(open('TFCorpus.json', 'r'))
# readTF = ijson.items(readTFFile, fileName+'.item')

# print(readTF.items())
df = pandas.read_json("TFCorpus.json")


# In[ ]:


# Reading the IDF Corpus
with open('IDFCorpus.json', 'r') as fp:
    readIDF = json.load(fp)
fp.close()
gc.collect()


# In[ ]:


# Reading the IDF Corpus
with open('ParagraphWordCorpus.json', 'r') as fp:
    readIDF13 = json.load(fp)
fp.close()
gc.collect()


# In[ ]:


# Reading the IDF Corpus
with open('ParagraphCorpus.json', 'r') as fp:
    readIDF13 = json.load(fp)
fp.close()
gc.collect()


# In[ ]:


#function to compute cosine similarity of the two tf-idf vector
def computeCosineSimilarity(vector1, vector2):
    return np.dot(vector1, vector2)/(np.linalg.norm(vector1) * np.linalg.norm(vector2))


# In[ ]:


#computing cosine similarity
for i in range(len(tokenisedSentences)-1):
    j = i + 1
    while(j < len(tokenisedSentences)):
        graph.add_edge(i, j, weight=computeCosineSimilarity(scoreMatrix[:, i], scoreMatrix[:, j]))#(u, v) edge
        j += 1

