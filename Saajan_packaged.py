import requests
import urllib
from collections import OrderedDict
import pickle
import re


import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB


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
    #print word
    
    if (word[0][1]=='NNP' or word[0][1]=='NN'):
        response = requests.get("https://www.googleapis.com/freebase/v1/search?query=" + word[0][0] + "&key=AIzaSyAP06H4HjCk69jQy4J78SKtlhcxup6Ae8k")
        jsonResponse = response.json()
        #print jsonResponse['result']
        resultString = ''
        
        counter=0
        for row in jsonResponse['result']:
            
            #print row
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
    #print "pass" + input
    #print returnString
    return returnString


##########################MAIN#############################

####################TRAIN PICKLE##########################
'''
inputFile = open("techTweets.txt", 'r+')

classifiedMessages = np.loadtxt("techTweets.txt", comments='\\<>=#', delimiter="\t", unpack=False, dtype ='string' )


i=0 
messageArray = pickle.load(open("TransformedTweetsArray.p","rb"))
for message in classifiedMessages[:]:
    print message[-1].replace('#','')
    messageArray.append([str( message[0] ) , str( replaceStringWithConcepts(( urllib.unquote(message[-1]).decode('utf8') ).replace('#',' ').replace('\\n',' ').replace('\\','')) )])
    if (i%200==0):
        pickle.dump(messageArray,open("TransformedTweetsArray.p","wb"))
    i+=1
print messageArray
'''


###########################################################################



def getPredictions (testingTweetsWithIDs):

    #####################LOAD PICKLE AND CROSS VALIDATE###########################################
    
    ## Pickle Dumps Add / Load
    #pickle.dump(messageArray,open("TransformedTechTweetsArray.p","wb"))
    labelledMessageArray = pickle.load(open("TransformedTechTweetsArray.p","rb"))
    tempArray = pickle.load(open("TransformedTechTweetsArray2.p","rb"))
    labelledMessageArray = labelledMessageArray + tempArray
    
    np.random.shuffle(labelledMessageArray)
    #labelledMessageArray = [ [s[0], s[1].replace('/', ' ')] for s in labelledMessageArray]
    labelledMessageArray = np.asarray(labelledMessageArray)
    trainingLabels = np.asarray(labelledMessageArray[:, 0], dtype=int )
    trainingMessages =  np.asarray(labelledMessageArray[:, 1] )
        
        
    vectorizer = CountVectorizer(ngram_range=(1, 3))
    frequencies = vectorizer.fit_transform(trainingMessages )
    
    classifier = MultinomialNB()
    targets = trainingLabels
    #print "\n\n\n\n" + str(targets)
    classifier.fit(frequencies, targets)
    
    #################################### CLASSIFIER TRAINED
    
    testingTweetsWithIDs = testingTweetsWithIDs
    #testingTweetsWithIDs = np.loadtxt("lala.txt", comments='\\<>=#', delimiter="\t", unpack=False, dtype ='string' )
    
    
    testingTweetsWithIDs = np.asarray(testingTweetsWithIDs)
    #print testingTweetsWithIDs
    
    
    ###############
    
    results = []
    for row in testingTweetsWithIDs:
        tweet_id = row[0]
        tweet = row[1]
        tweet_stripped_url = re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', '', urllib.unquote_plus(tweet).decode('utf8'))
        tweet_concept = str( replaceStringWithConcepts(( urllib.unquote(tweet_stripped_url).decode('utf8') ).replace('#',' ').replace('\\n',' ').replace('\\','')) )
        
        test_result = vectorizer.transform([tweet_concept]) #TWEET
        prediction = classifier.predict(test_result)
        prediction_probability = classifier.predict_proba(test_result)
        print "Tweet id: " + tweet_id + " " + "done!"
        #print prediction
        #print prediction_probability
        results.append([tweet_id, prediction[0], prediction_probability[0][1]])   
    
    
    return results
    
