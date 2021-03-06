"""
This file is used to process reviews and turn them into word vectors for classification. 
Each vector represents a sentiment phrase. 
"""
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS as stop_words
import sklearn
import re


class WordProcessor:

    def __init__(self,max_features):
        self.tfidfconverter = TfidfVectorizer(max_features=max_features, min_df=5, max_df=0.7, stop_words=stop_words, use_idf=True, ngram_range=(1,3))

    def process(self, reviews):
        """
        Processes a review for classification. 
        Breaks each review into phrases and filters special characters.
        Returns a list of vector representations of the review. 
        """
        final_reviews = []
        for review in reviews:
            review = review.lower()
            filtered_review = self.filter_special_characters(review)
            words = filtered_review.split()
            processed_review = []
            for word in words:
                if self.is_valid_word(word):
                    processed_review.append(word)

            final_reveiew = ' '.join(processed_review)
            final_reviews.append(final_reveiew)

        return final_reviews

        
    def is_valid_word(self, word):
        """ Check if word begins with an alphabet"""
        return (re.search(r'^[a-zA-Z][a-z0-9A-Z\._]*$', word) is not None)

    def filter_special_characters(self, review):
        """
        Filter special characters from the review. 
        Don't remove periods.
        """
        re_reviews = re.sub('[^A-Za-z0-9.!?:;\'\"(), ]+', '', review)
        h_re_reviews = re.sub('[-]+', ' ', re_reviews)
        return h_re_reviews

    def vectorize_train(self, reviews):
        """ Vectorizes reviews after fitting tfid. """
        return self.tfidfconverter.fit_transform(reviews)

    def vectorize(self, reviews):
        """ Vectorizes reviews using an already fitted tdif. """
        return self.tfidfconverter.transform(reviews)
