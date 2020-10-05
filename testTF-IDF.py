
# coding: utf-8

# In[1]:


import csv
import re
import string
import nltk
from nltk.corpus import stopwords
import math
import json
import os
import gc
from rouge import Rouge 
# In[2]:


# reading the summary corpus
with open('SummaryCorpus.json', 'r') as fp:
    readSummary = json.load(fp)
fp.close()
readSummary = dict(sorted(readSummary.items()))


# In[3]:


# reading the sentences corpus
with open('SentCorpus.json', 'r') as fp:
    readSent = json.load(fp)
fp.close()


# In[4]:


# reading the bag of word corpus
with open('BagOfWordCorpus.json', 'r') as fp:
    readWord = json.load(fp)
fp.close()


# In[5]:


# Reading the TF Corpus
with open('TFCorpus.json', 'r') as fp:
    readTF = json.load(fp)
fp.close()


# Reading the IDF Corpus
with open('IDFCorpus13.json', 'r') as fp:
    readIDF13 = json.load(fp)
fp.close()
gc.collect()

# Reading the IDF Corpus
with open('IDFCorpus18.json', 'r') as fp:
    readIDF18 = json.load(fp)
fp.close()
gc.collect()

# In[6]:


#reading the legal dictionary
with open('LegalDictionaryCorpus.json', 'r') as fp:
    legalDictionary = json.load(fp)
fp.close()


# In[7]:


fileName = []
listOfFiles = []
listOfDocument = []


# In[8]:


listofReadFile = []
numDoc = 0
for file in os.listdir("LegalCorpus"):
    if file.endswith(".txt"):
        listofReadFile.append(file)
        numDoc += 1
listofReadFile = sorted(listofReadFile)


# In[ ]:


csvData = []
csvData.append(['File Name', 'Rouge-1 F', 'Rouge-1 P', 'Rouge-1 R', 'Rouge-2 F', 'Rouge-2 P', 'Rouge-2 R', 'Rouge-L F', 'Rouge-L P', 'Rouge-L R'])
flagCount = 1
sumRouge1f = 0
sumRouge1p = 0
sumRouge1r = 0

sumRouge2f = 0
sumRouge2p = 0
sumRouge2r = 0

sumRougeLf = 0
sumRougeLp = 0
sumRougeLr = 0

for fileName in listofReadFile[:]:
    fileinfo = os.stat("LegalCorpus/"+fileName)

    if(fileinfo.st_size <= 1024):
        continue
    
    print (flagCount, '>', fileName, end = '  ')
    
    sentencesofDocument = readSent[fileName]
    bagofWordsofDocument = readWord[fileName]
    dictTF = readTF[fileName]
    if(fileName.startswith('19') or fileName.startswith('2010') or fileName.startswith('2011') or fileName.startswith('2012') or fileName.startswith('2013')):
        dictIDF = readIDF13[fileName]
    else:
        dictIDF = readIDF18[fileName]

    #Calculating the term frequency of each dictionary
    tfDictionary = {}
    tfDictionary = dictTF


    idfDictionary = {}
    idfDictionary = dictIDF
    
    def computeTFIDF(tfDictionary, idfDictionary):
        tfidf = {}
        found = 0
        for term, value in tfDictionary.items():
            if value > 0 and term in legalDictionary['LegalDict']:
                found = 1

            if(found == 1):
                tfidf[term] = value * idfDictionary[term]  # giving more weightage to legal term
            else:
                tfidf[term] = value * idfDictionary[term]  # tfidf normal calculation for non legal terms
            found = 0

        return tfidf

    tfidfDictionary = {}
    tfidfDictionary = computeTFIDF(tfDictionary, idfDictionary)

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

    sentenceIndex = []
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

    sentenceIndex = []
    sentenceIndex = sorted(tempIndices)

    # print highest 34% words of the total sentences
    for index in sentenceIndex:
        system_generated_summary += sentencesofDocument[0][index]

    if not len(sentencesofDocument[0])-1 in sentenceIndex:
        system_generated_summary += sentencesofDocument[0][len(sentencesofDocument[0])-1]

    # Computing the Rouge Scores
    hypothesis = system_generated_summary
    reference = readSummary[fileName]

    rouge = Rouge()
    scores = rouge.get_scores(hypothesis, reference)

    sumRouge1f += scores[0]['rouge-1']['f']
    sumRouge1p += scores[0]['rouge-1']['p']
    sumRouge1r += scores[0]['rouge-1']['r']
    
    sumRouge2f += scores[0]['rouge-2']['f']
    sumRouge2p += scores[0]['rouge-2']['p']
    sumRouge2r += scores[0]['rouge-2']['r']
    
    sumRougeLf += scores[0]['rouge-l']['f']
    sumRougeLp += scores[0]['rouge-l']['p']
    sumRougeLr += scores[0]['rouge-l']['r']
    
    csvData.append([fileName, scores[0]['rouge-1']['f'], scores[0]['rouge-1']['p'], scores[0]['rouge-1']['r'], scores[0]['rouge-2']['f'], scores[0]['rouge-2']['p'], scores[0]['rouge-2']['r'], scores[0]['rouge-l']['f'], scores[0]['rouge-l']['p'], scores[0]['rouge-l']['r']])

    if(flagCount % 10 == 0):
        print('\n')
    flagCount += 1
    numDoc = flagCount
    gc.collect()

# In[ ]:

print (numDoc)
numDoc = 7090
csvData.append(['Average', sumRouge1f/numDoc, sumRouge1p/numDoc, sumRouge1r/numDoc, sumRouge2f/numDoc, sumRouge2p/numDoc, sumRouge2r/numDoc, sumRougeLf/numDoc, sumRougeLp/numDoc, sumRougeLr/numDoc])
with open('New1scoresNoMul2NoDiv2Formula.csv', 'w') as csvFile:
    writer = csv.writer(csvFile)
    writer.writerows(csvData)

csvFile.close()
print ('Done')

