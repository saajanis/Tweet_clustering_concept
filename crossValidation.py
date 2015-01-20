import numpy as np
import random

class CrossValidation: 
    def get9010Splits(self, list):
        ResultSet = []
        
        #np.random.shuffle(list)
        
        Parts10 = np.asarray(np.array_split(list, 10))
        
        for x in range (Parts10.shape[0]):
            Splits9010 = []
            Split90 = []
            Split10 = []
            for i in range(Parts10.shape[0]):
                if (i==x):
                    continue
                for line in Parts10[i]:
                    Split90.append(line)
            
            for line in Parts10[x]:
                Split10.append(line)
            Splits9010.append(Split90)
            Splits9010.append(Split10)
            ResultSet.insert(x,Splits9010)
        
        return np.asarray(ResultSet)
        
    
    ##
    def removeAttributes10Percent(self, listOfRows):
        resultlistOfRows = []
        
        shuffledRowIndices = [i for i in range(listOfRows.shape[0])]
        np.random.shuffle(shuffledRowIndices)
        shuffledRowsParts = np.asarray(np.array_split(shuffledRowIndices, 10))
        
        for i in range (shuffledRowsParts.shape[0]-1):
            for index in shuffledRowsParts[i]:
                resultlistOfRows.append(listOfRows[index])
        #print len(resultlistOfRows)
        
        for example in listOfRows[shuffledRowsParts[shuffledRowsParts.shape[0]-1], :]:
            attrIndexToRemove = random.randint(0,3)
            example[attrIndexToRemove] = -11
            resultlistOfRows.append(example)
        
        np.random.shuffle(resultlistOfRows)
        return resultlistOfRows
    
    
    def getAccuracy(self, TP, FP, TN, FN):
        
        numerator = float(TP+TN)
        denominator = float(TP)+float(FP)+float(TN)+float(FN)
        
        return (numerator/denominator)
    
    def getTrueFalsePositives(self, test, truth):
        countTP = 0
        countFP = 0
        for i in range(len(test)):
            if (test[i]==1):
                if (truth[i]==1):
                    countTP+=1
                else:
                    countFP+=1
                    
        return [countTP, countFP] 
    
    def getTrueFalseNegatives(self, test, truth):
        countTN = 0
        countFN = 0
        for i in range(len(test)):
            if (test[i]==0):
                if (truth[i]==0):
                    countTN+=1
                else:
                    countFN+=1
                    
        return [countTN, countFN]                
    
    def getPrecision(self, TP, FP, TN, FN):
        
        numerator = float(TP)
        denominator = float(TP)+float(FP)
        
        return (numerator/denominator)
     
    def getRecall(self, TP, FP, TN, FN):
        
        numerator = float(TP)
        denominator = float(TP)+float(FN)
        
        return (numerator/(denominator+1))
    
    def getF1Score(self, precision, recall):
        
        numerator = float(precision*recall)
        denominator = float(precision)+float(recall)
        
        return (2*(numerator/denominator))
     
 
 
 
#################MAIN####################
# 
# with open("data_banknote_authentication_jaggered.txt", "r") as dataset:
#     testList = np.asarray([line.split(",") for line in dataset.readlines()], dtype=np.float64)
#  
#  
# ResultSet = get9010Splits( testList)
#  
# print np.asarray(ResultSet[1][1]).shape
#    





#print CV.getTrueFalsePositives(test, truth)
#print CV.getTrueFalseNegatives(test, truth)
