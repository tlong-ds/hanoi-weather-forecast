import joblib
import numpy as np
import re
import string

import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords, twitter_samples
from sklearn.linear_model import LogisticRegression

nltk.download('twitter_samples')
nltk.download('stopwords')
all_positive_tweets = twitter_samples.strings('positive_tweets.json')
all_negative_tweets = twitter_samples.strings('negative_tweets.json')

class LogisticRegressionModel:
    def __init__(self):
        # split data into train and test set
        train_pos = all_positive_tweets[:4000]
        train_neg = all_negative_tweets[:4000]
        self.train_x = train_pos + train_neg 
        self.train_y = np.append(np.ones((len(train_pos), 1)), np.zeros((len(train_neg), 1)), axis=0)
        self.freqs = LogisticRegressionModel.build_freqs(self.train_x, self.train_y)
        try:
            self.model = joblib.load("sk_logreg.pkl")
        except:
            self.model = LogisticRegressionModel.train(self.train_x, self.train_y, self.freqs)
            joblib.dump(self.model, "sk_logreg.pkl")

    def predict(self, query):
        features = LogisticRegressionModel.extract_features(query, self.freqs)
        result = self.model.predict_proba(features[:, 1:])
        return result
    
    @staticmethod
    def train(train_x, train_y, freqs):
        train_x_vec = np.vstack([LogisticRegressionModel.extract_features(t,freqs) for t in train_x])

        model = LogisticRegression()
        model.fit(train_x_vec[:, 1:], train_y.ravel())

        return model

    @staticmethod
    def process_tweet(tweet):
        """
        Input:
            :tweet: a string
        Output:
            :tweets_clean: a list of words containing the processed tweet
        """
        stemmer = PorterStemmer()
        stopwords_english = stopwords.words('english')

        # remove stock market tickers like $GE
        tweet = re.sub(r'\$\w*', '', tweet)
        # remove old style retweet text "RT"
        tweet = re.sub(r'^RT[\s]+', '', tweet)
        # remove hyperlinks
        tweet = re.sub(r'https?://[^\s\n\r]+', '', tweet)
        # remove hashtags
        # only removing the hash # sign from the word
        tweet = re.sub(r'#', '', tweet)

        # tokenize tweets
        tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True) #the tokenizer will downcase everything except for emoticons
        tweet_tokens = tokenizer.tokenize(tweet)

        tweets_clean = []
        for word in tweet_tokens:
            if (word not in stopwords_english and   # remove stopwords
                    word not in string.punctuation): # remove punctuation
                stem_word = stemmer.stem(word)
                tweets_clean.append(stem_word)

        return tweets_clean

    @staticmethod
    def build_freqs(tweets, ys):
        """ Build frequencies
        Input:
        tweets: a list of tweets
        ys: an mx1 array with the sentiment label of each tweet (either 0 or 1)
        Output:
        freqs: a dictionary mapping each (word, sentiment) pair to its frequency
        """
        yslist = np.squeeze(ys).tolist()
        # start with an empty dict and populate it by looping over all tweets
        freqs = {}
        for y, tweet in zip(yslist, tweets):
            for word in LogisticRegressionModel.process_tweet(tweet):
                pair = (word, y)
                if pair in freqs:
                    freqs[pair] += 1
                else:
                    freqs[pair] = 1

        return freqs

    @staticmethod
    def extract_features(tweet, freqs, process_tweet=process_tweet):
        '''
        Input: 
            tweet: a list of words for one tweet
            freqs: a dictionary corresponding to the frequencies of each tuple (word, label)
        Output: 
            x: a feature vector of dimension (1,3)
        '''
        # process_tweet tokenizes, stems, and removes stopwords
        word_l = process_tweet(tweet)
        
        # 3 elements in the form of a 1 x 3 vector
        x = np.zeros((1, 3)) 
        
        #bias term is set to 1
        x[0,0] = 1    
        # loop through each word in the list of words
        for word in word_l:
            
            # increment the word count for the positive label 1
            if (word, 1) in freqs.keys():
                x[0,1] += freqs[(word, 1)]
            
            # increment the word count for the negative label 0
            if (word, 0) in freqs.keys():
                x[0,2] += freqs[(word, 0)]
            
        assert(x.shape == (1, 3))
        return x
    

if __name__ == "__main__":
    # Example usage
    lr_instance = LogisticRegressionModel()
    test_tweet = "I am happy happy happy!"
    prediction = lr_instance.predict(test_tweet)
    print(f"Tweet: {test_tweet}")
    print(f"Prediction (probabilities for [neg, pos]): {prediction}")
    print(f"Predicted sentiment: {'Positive' if prediction[0][1] >= 0.5 else 'Negative'}")