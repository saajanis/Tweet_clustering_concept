from Saajan_packaged import *

testingTweetsWithIDs = np.loadtxt("sample.txt", comments='\\<>=#', delimiter="\t", unpack=False, dtype ='string' )
testingTweetsWithIDs = np.asarray(testingTweetsWithIDs)
results = getPredictions (testingTweetsWithIDs)

for result in results:
    print result
    
