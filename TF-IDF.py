
# coding: utf-8

# In[ ]:


import re
import string
import nltk
from nltk.corpus import stopwords
import math
import json
import os
import gc


# In[ ]:


# reading the summary corpus
with open('SummaryCorpus.json', 'r') as fp:
    readSummary = json.load(fp)
fp.close()
gc.collect()


# In[ ]:


# reading the sentences corpus
with open('SentCorpus.json', 'r') as fp:
    readSent = json.load(fp)
fp.close()
gc.collect()


# In[ ]:


# reading the bag of word corpus
with open('BagOfWordCorpus.json', 'r') as fp:
    readWord = json.load(fp)
fp.close()
gc.collect()


# In[ ]:


# Reading the IDF Corpus
with open('IDFCorpus13.json', 'r') as fp:
    readIDF13 = json.load(fp)
fp.close()
gc.collect()


# In[ ]:


# Reading the IDF Corpus
with open('IDFCorpus18.json', 'r') as fp:
    readIDF18 = json.load(fp)
fp.close()
gc.collect()


# In[ ]:


# Reading the DF Corpus
with open('TFCorpus.json', 'r') as fp:
    readTF = json.load(fp)
fp.close()
gc.collect()


# In[ ]:


#reading the legal dictionary
with open('LegalDictionaryCorpus.json', 'r') as fp:
    legalDictionary = json.load(fp)
fp.close()
gc.collect()


# In[ ]:


fileName = []
listOfFiles = []
listOfDocument = []


# In[ ]:


#reading the document
fileName = input('Please insert the name of the document : ')


# In[ ]:


sentencesofDocument = readSent[fileName]
bagofWordsofDocument = readWord[fileName]
dictTF = readTF[fileName]

if(fileName.startswith('19') or fileName.startswith('2010') or fileName.startswith('2011') or fileName.startswith('2012') or fileName.startswith('2013')):
        dictIDF = readIDF13[fileName]
    else:
        dictIDF = readIDF18[fileName]


# In[ ]:


#Calculating the term frequency of each dictionary
tfDictionary = {}
tfDictionary = dictTF


# In[ ]:


idfDictionary = {}
idfDictionary = dictIDF


# In[ ]:


def computeTFIDF(tfDictionary, idfDictionary):
    tfidf = {}
    found = 0
    for term, value in tfDictionary.items():
        if value > 0 and term in legalDictionary['LegalDict']:
            found = 1

        if(found == 1):
             tfidf[term] = value * idfDictionary[term] * 2.0 # giving more weightage to legal terms
        else:
            tfidf[term] = value * idfDictionary[term] / 2.0  # tfidf normal calculation for non legal terms
        found = 0

    return tfidf


# In[ ]:


tfidfDictionary = {}
tfidfDictionary = computeTFIDF(tfDictionary, idfDictionary)


# In[ ]:


# Computing the length of the each sentence in terms of #words in the sentences
countTokensinSentence = []
for eachLine in sentencesofDocument[0]:
    tokenisedWords = nltk.word_tokenize(eachLine)
    if(tokenisedWords != []): # for counting the number of terms
        countTokensinSentence.append(tokenisedWords)


sentenceScore = []
dictSentenceScore = {}
sum = 0
i = 0
for sent in bagofWordsofDocument:
    for word in sent:
        sum += tfidfDictionary[word]
    sum = (sum/(len(countTokensinSentence[i])))
    sentenceScore.append(sum)
    dictSentenceScore[i] = sum
    i += 1
    sum = 0
    
import statistics
sd = statistics.stdev(sentenceScore)

ner = []
listdigits = []
named_entities = []
from date_extractor import extract_dates
for sent in sentencesofDocument[0]:
    tokenized_doc = nltk.word_tokenize(sent)
    # tag sentences and use nltk's Named Entity Chunker
    tagged_sentences = nltk.pos_tag(tokenized_doc)
    ne_chunked_sents = nltk.ne_chunk(tagged_sentences)
    # extract all named entities
    named_entities = []
    for tagged_tree in ne_chunked_sents:
        if hasattr(tagged_tree, 'label'):
            entity_name = ' '.join(c[0] for c in tagged_tree.leaves()) #
            entity_type = tagged_tree.label() # get NE category
            named_entities.append((entity_name, entity_type))
    ner.append(len(named_entities))

    digits = re.findall(r'\d+', sent)
    listdigits.append(len(digits))
    
sum = 0
l = 0
i = 0
for sent in bagofWordsofDocument:
    for word in sent:
        if word in legalDictionary['LegalDict']:
            l += 1

    d = listdigits[i]
    e = ner[i]
    sum += sentenceScore[i] + (sd*(1.5*l + 0.2*d + 0.3*e))/len(countTokensinSentence[i])
    sentenceScore.append(sum)
    dictSentenceScore[i] = sum
    i += 1
    sum = 0
    l = 0


sentenceIndex = sorted(dictSentenceScore, key=dictSentenceScore.get, reverse=True)

system_generated_summary = ""
termCount = 0
tempIndices = []
sent = []
totalWordsinDocument = sentencesofDocument[1]
for index in sentenceIndex:
    if (termCount <= totalWordsinDocument):
        tempIndices.append(index)
        termCount += len(sentencesofDocument[0][index])
    else:
        break


# In[ ]:


# GENERATING THE COLORS FOR HEAT MAP
from IPython.display import Markdown

upper = "0xff0000"
lower = "0x0000ff"
totalSent = len(tempIndices)
lowSent = totalSent//2
topSent = totalSent - lowSent

incFactortopRed = (256)//topSent
incFactortopGreen = (256)//topSent

if(topSent >= lowSent):
    loopfor = topSent
    
upperColors = []
lowerColors = []
upperColors.append('#ff0000')
lowerColors.append('#0000ff')
for i in range(1, loopfor):
    upper = upper.replace("0x", "")
    upper = list(upper)
    
    red = '0x' + upper[0] + upper[1]
    green = '0x' + upper[2] + upper[3]
    
    i = int(green, 16)
    i += incFactortopGreen
    k = int(red, 16)
    k -= incFactortopRed
    
    i = hex(i)
    k = hex(k)
    
    i = str(i).replace("0x", "")
    k = str(k).replace("0x", "")
    
    if(len(i)==1):
        i = '0'+ i
    if(len(k)==1):
        k = '0' + k
    
    upper = '0x' + k + i + '00'
    
    i = str(upper).replace('0x', '')

    j = list(str(i))
    j[4] = j[0]
    j[5] = j[1]
    j[0] = '0'
    j[1] = '0'
    j = "" + "".join(j)
    
    upperColors.append('#' + i)
    lowerColors.append('#' + j)

if(topSent < lowSent):
    del lowerColors[-1]

listColor = []
listColor = upperColors
for i in range(len(lowerColors)):
    listColor.append(lowerColors[-i-1])


# In[ ]:


dictColor = {}
j = 0
for i in tempIndices:
    dictColor[i] = listColor[j]
    j += 1

sentenceIndex = []
sentenceIndex = sorted(tempIndices)
    
# print highest 34% words of the total sentences
for index in sentenceIndex:
    display (Markdown('<span style="color: '+ dictColor[index] +'">'+str(dictSentenceScore[index])+' '+sentencesofDocument[0][index]+'</span>'))
    system_generated_summary += sentencesofDocument[0][index]
flag = 0
if not len(sentencesofDocument[0])-1 in sentenceIndex:
    for key, value in dictColor.items():
        if(dictSentenceScore[len(sentencesofDocument[0])-1] >= dictSentenceScore[key]):
            flag = 1
            break
    if(flag == 1):
        display (Markdown('<span style="color: '+dictColor[key]+'">'+str(dictSentenceScore[len(sentencesofDocument[0])-1])+' '+sentencesofDocument[0][len(sentencesofDocument[0])-1]+'</span>'))
        system_generated_summary += sentencesofDocument[0][len(sentencesofDocument[0])-1]
    
    else:
        display (Markdown('<span style="color: #0000ff">'+str(dictSentenceScore[len(sentencesofDocument[0])-1])+' '+sentencesofDocument[0][len(sentencesofDocument[0])-1]+'</span>'))
        system_generated_summary += sentencesofDocument[0][len(sentencesofDocument[0])-1]


# In[ ]:


from rouge import Rouge 

hypothesis = system_generated_summary

reference = readSummary[fileName]

rouge = Rouge()
scores = rouge.get_scores(hypothesis, reference)
print ('Rouge-1 : ', scores[0]['rouge-1'])
print ('Rouge-2 : ', scores[0]['rouge-2'])
print ('Rouge-L : ', scores[0]['rouge-l'])

