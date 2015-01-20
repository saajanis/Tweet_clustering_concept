from crossValidation import *

import requests
import json
import urllib
from collections import OrderedDict
import pickle

import os
import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import Perceptron
from sklearn import svm
from sklearn.cross_validation import KFold
from sklearn.pipeline import Pipeline

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
import nltk

stopset = set(stopwords.words('english'))

def removeStopWords(input):
    tokenizer = RegexpTokenizer(r'\w+')
    tokens=tokenizer.tokenize(input)
    tokens = [w for w in tokens if not w in stopset]
    returnString= ' '.join(tokens)
    returnString = ' '.join(OrderedDict((w,w) for w in returnString.split()).keys())
    return returnString


def strip_non_ascii(string):
    ''' Returns the string without non ASCII characters'''
    stripped = (c for c in string if 0 < ord(c) < 127)
    return ''.join(stripped)


def getConceptsForWord(word):
   
    
    word = word_tokenize(word)
    word = nltk.tag.pos_tag(word)
    print word
    
    if (word[0][1]=='NNP' or word[0][1]=='NN'):
        response = requests.get("https://www.googleapis.com/freebase/v1/search?query=" + word[0][0] + "&key=AIzaSyAP06H4HjCk69jQy4J78SKtlhcxup6Ae8k")
        jsonResponse = response.json()
        #print jsonResponse['result']
        resultString = ''
        
        counter=0
        for row in jsonResponse['result']:
            
            print row
            try: 
                resultString = resultString + row['notable']['id'] + ' '
                counter+=1
            except KeyError: pass
            
            if (counter>=4): #no. of ids
                break
        
        #resultString = ' '.join(OrderedDict((w,w) for w in resultString.lower().split()).keys())    
        #print resultString   
        return resultString

def replaceStringWithConcepts(input):
    
    input = strip_non_ascii(input)
    
    input = removeStopWords(input)
    returnString = ''
    splitString = input.split()
    for splitWord in splitString:
        concept = getConceptsForWord(splitWord)
        if not concept:
            returnString = returnString + ' ' + splitWord
        else:
            returnString = returnString +  ' ' + concept
    returnString = ' '.join(OrderedDict((w,w) for w in returnString.split()).keys())
    print "pass" + input
    print returnString
    return returnString


#print getConceptsForWord("iphone")

#print replaceStringWithConcepts("I'm gonna be home soon and i don't want to talk about this stuff anymore tonight")

''' Language
for edges in jsonResponse['edges']:
    relation = edges['rel']
    #if (relation == '/r/IsA'):
    if (relation == '/r/TranslationOf'):
        x= edges['surfaceText']
        print x.encode('ascii','ignore')
        print edges['score']
'''
    
#s = "klein calvin  design dress calvin klein"
#print ' '.join(OrderedDict((w,w) for w in s.split()).keys())





############Spam Detect //USE FREEBASE

##########################MAIN#############################

####################TRAIN PICKLE##########################

'''

inputFile = open("techTweets6000.txt", 'r+')

classifiedMessages = np.loadtxt("techTweets6000.txt", comments='\\<>=#', delimiter="\t", unpack=False, dtype ='string' )


i=0 
messageArray = pickle.load(open("TransformedTechTweetsArray2.p","rb"))
for message in classifiedMessages[:]:
    print message[-1].replace('#','')
    messageArray.append([str( message[0] ) , str( replaceStringWithConcepts(( urllib.unquote(message[-1]).decode('utf8') ).replace('#',' ').replace('\\n',' ').replace('\\','')) )])
    if (i%200==0):
        pickle.dump(messageArray,open("TransformedTechTweetsArray2.p","wb"))
    i+=1
print messageArray

'''

###########################################################################





#####################LOAD PICKLE AND CROSS VALIDATE###########################################

## Pickle Dumps Add / Load
#pickle.dump(messageArray,open("TransformedTechTweetsArray.p","wb"))
labelledMessageArray = pickle.load(open("TransformedTechTweetsArray.p","rb"))
tempArray = pickle.load(open("TransformedTechTweetsArray2.p","rb"))
labelledMessageArray = labelledMessageArray + tempArray
np.random.shuffle(labelledMessageArray)
#labelledMessageArray = [ [s[0], s[1].replace('/', ' ')] for s in labelledMessageArray]

print len(labelledMessageArray)

CrossValidation = CrossValidation() 

##################
averageTP = 0
averageFP = 0
averageTN = 0
averageFN = 0
averageAccuracy = 0
averagePrecision = 0
averageRecall = 0
averageF1Score = 0

ResultSet = CrossValidation.get9010Splits( labelledMessageArray)


for splits9010Instance in ResultSet:
    
    labelledMessageArrayTrain = np.asarray(splits9010Instance[0])
    labelledMessageArray10PercentTest = np.asarray(splits9010Instance[1])
    
    
    trainingLabels = np.asarray(labelledMessageArrayTrain[:, 0], dtype=int )
    trainingMessages =  np.asarray(labelledMessageArrayTrain[:, 1] )
    
    
    
    vectorizer = CountVectorizer(ngram_range=(1, 3))
    frequencies = vectorizer.fit_transform(trainingMessages )
    
    
    
    targets = trainingLabels
    
    '''NB Part
    classifier = MultinomialNB()
    classifier.fit(frequencies, targets)
    '''
    
    #''' Perceptron Part
    Perceptronclf = Perceptron()
    Perceptronclf.fit(frequencies, targets)
    #'''
    
    
    testingLabels = np.asanyarray(labelledMessageArray10PercentTest[:, 0], dtype=int)
    testingMessages =  np.asarray( labelledMessageArray10PercentTest[:, 1] )
    
    
    truth = testingLabels
    result = []
    
    
    for test_msg in testingMessages:
       test_result = vectorizer.transform([test_msg])
       #prediction = classifier.predict(test_result)
       prediction = Perceptronclf.predict(test_result)
       
       result.append(int( prediction[0]) )   
    # 
    result = np.asarray(result, dtype=int)
    
    TrueFalsePositives = CrossValidation.getTrueFalsePositives(result, truth)
    TrueFalseNegatives = CrossValidation.getTrueFalseNegatives(result, truth)
    TP = TrueFalsePositives[0]
    FP = TrueFalsePositives [1]
    TN = TrueFalseNegatives[0]
    FN = TrueFalseNegatives[1]
    
    
    print "Truth: " + str(truth)       
    print "Final result: " + str(result)
    print "True positives: " + str(TP)
    print "False positives: " + str(FP)
    print "True Negatives: " + str(TN)
    print "False Negatives: " + str(FN)
    print "Accuracy: " + str(CrossValidation.getAccuracy(TP, FP, TN, FN) * 100) + "%"
    print "Precision: " + str(CrossValidation.getPrecision(TP, FP, TN, FN) * 100) + "%"
    print "Recall: " + str(CrossValidation.getRecall(TP, FP, TN, FN) * 100) + "%"
    print "F1 Score: " + str (CrossValidation.getF1Score(CrossValidation.getPrecision(TP, FP, TN, FN), CrossValidation.getRecall(TP, FP, TN, FN))* 100) + "%"
    
    
    
     
     
    averageTP += TP
    averageFP += FP
    averageTN += TN
    averageFN += FN
    averageAccuracy += CrossValidation.getAccuracy(TP, FP, TN, FN)
    averagePrecision += CrossValidation.getPrecision(TP, FP, TN, FN)
    averageRecall += CrossValidation.getRecall(TP, FP, TN, FN)     
    averageF1Score += CrossValidation.getF1Score(CrossValidation.getPrecision(TP, FP, TN, FN), CrossValidation.getRecall(TP, FP, TN, FN))
    #break
    
print "\n\nAverage True Positives: " + str(averageTP/10)  
print "Average False positives: " + str(averageFP/10)  
print "Average True Negatives: " + str(averageTN/10)  
print "Average False Negatives: " + str(averageFN/10)  
print "Average Accuracy: " + str( (averageAccuracy/10) * 100) + "%"  
print "Average Precision: " + str( (averagePrecision/10) * 100) + "%"
print "Average Recall: " + str( (averageRecall/10)   * 100) + "%" 
print "Average F1 Score: " + str( (averageF1Score/10)   * 100) + "%"  

##################


'''
engadget
TheNextWeb
ZDNet
TechRepublic
tecnbiz     
'''