       
import urllib, json
import sys
import tweepy
from tweepy import OAuthHandler

def twitter_fetch(screen_name = "BBCNews",maxnumtweets=10):
    'Fetch tweets from @BBCNews'
    # API described at https://dev.twitter.com/docs/api/1.1/get/statuses/user_timeline

    consumer_token = 'Q1U7mTxPFWwKBQhkjtua3g' #substitute values from twitter website
    consumer_secret = 'GIzajCDvNMCPR9xZPuIgQJUVkJk5NriA5mmC2slgA'
    access_token = '390753770-sIkR0XBpXZGWrqBeN0GZMa4TvHHFjTbW1U1MAikK'
    access_secret = '12ciPkWppjB44L6SeHWNXBhFhUv2KF4RJAUV3AABAk'
    
 #!/usr/bin/env python
# encoding: utf-8
 
import tweepy #https://github.com/tweepy/tweepy
import csv
 
#Twitter API credentials
consumer_key = ""
consumer_secret = ""
access_key = ""
access_secret = ""
 
 
def get_all_tweets(screen_name):
    #Twitter only allows access to a users most recent 3240 tweets with this method
    
    #authorize twitter, initialize tweepy
    auth = tweepy.OAuthHandler("Q1U7mTxPFWwKBQhkjtua3g", "GIzajCDvNMCPR9xZPuIgQJUVkJk5NriA5mmC2slgA")
    auth.set_access_token("390753770-sIkR0XBpXZGWrqBeN0GZMa4TvHHFjTbW1U1MAikK", "12ciPkWppjB44L6SeHWNXBhFhUv2KF4RJAUV3AABAk")
    api = tweepy.API(auth)
    
    #initialize a list to hold all the tweepy Tweets
    alltweets = []    
    
    #make initial request for most recent tweets (200 is the maximum allowed count)
    new_tweets = api.user_timeline(screen_name = screen_name,count=200)
    
    #save most recent tweets
    alltweets.extend(new_tweets)
    
    #save the id of the oldest tweet less one
    oldest = alltweets[-1].id - 1
    i=0
    #keep grabbing tweets until there are no tweets left to grab
    while len(new_tweets) > 0:
        print "getting tweets before %s" % (oldest)
        
        #all subsiquent requests use the max_id param to prevent duplicates
        new_tweets = api.user_timeline(screen_name = screen_name,count=200,max_id=oldest)
        
        #save most recent tweets
        alltweets.extend(new_tweets)
        
        #update the id of the oldest tweet less one
        oldest = alltweets[-1].id - 1
        
        print "...%s tweets downloaded so far" % (len(alltweets))
        if (i>8):
            break
        else:
            i=i+1
    
    
    #transform the tweepy tweets into a 2D array that will populate the csv    
    outtweets = [[tweet.text.encode("utf-8")] for tweet in alltweets]
    print outtweets
    
    with open("techTweets_temp.txt", "a") as f:
        i=0
        for tweet in outtweets:
            i=i+1
            f.write ("1" + "\t" + str(urllib.quote_plus(tweet[0])) + "\n")
            #if (i>=200):
            # break
                
#     #write the csv    
#     with open('%s_tweets.csv' % screen_name, 'wb') as f:
#         writer = csv.writer(f)
#         writer.writerow(["id","created_at","text"])
#         writer.writerows(outtweets)
#     
#     pass
 
 
if __name__ == '__main__':
    #pass in the username of the account you want to download
    get_all_tweets("tecnbiz")
# user_timeline = twitter.get_user_timeline(screen_name="sadatshami",count=1)
# print user_timeline