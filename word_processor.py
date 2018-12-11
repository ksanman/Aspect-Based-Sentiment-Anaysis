"""
This file is used to process reviews and turn them into word vectors for classification. 
Each vector represents a sentiment phrase. 
"""
from nltk import sent_tokenize
from pprint import pprint
from nltk.tokenize.punkt import PunktSentenceTokenizer, PunktTrainer
from nltk.corpus import gutenberg

class WordProcessor:

    def __init__(self):
        pass

    def process(self, review):
        """
        Processes a review for classification. 
        Breaks each review into phrases and filters special characters.
        Returns a list of vector representations of the review. 
        """
        sentances = self.break_into_sentances(review)
        phrases = self.break_into_phrases(sentances)
        filtered_review = self.filter_special_characters(review)
        unhyphenated_review = self.filter_hyphen_words(filtered_review)

        #Vectorize and return

    def filter_special_characters(self, review):
        """
        Filter special characters from the review. 
        Don't remove periods.
        """
        stripped_review = review.strip('\'"?!,():;')
        return stripped_review

    def filter_hyphen_words(self, review):
        """
        Fix hyphenated words.
        """
        pass

    def break_into_sentances(self, review):
        """
        Break the review into sentances. 
        """
        sentances = sent_tokenize(review)
        return sentances

    def break_into_phrases(self, sentances):
        """
        Break each sentance into phrases that can be classified. 
        """
        pass

    def train_tokenizer(self, reviews=None):
        
        text = ""
        if not reviews:
            print dir(gutenberg)
            print gutenberg.fileids()
        
            
            for file_id in gutenberg.fileids():
                text += gutenberg.raw(file_id)
        else:
            for review in reviews:
                text += review
        text = text.decode('UTF-8')
        print len(text)      
        trainer = PunktTrainer()
        trainer.INCLUDE_ALL_COLLOCS = True
        trainer.train(text)
        
        tokenizer = PunktSentenceTokenizer(trainer.get_params())
        
        # Test the tokenizer on a piece of text
        sentences = "Mr. James told me Dr. Brown is not available today. I will try tomorrow."
        
        print tokenizer.tokenize(sentences)
        # ['Mr. James told me Dr.', 'Brown is not available today.', 'I will try tomorrow.']
        
        # View the learned abbreviations
        print tokenizer._params.abbrev_types
        # set([...])
        
        # Here's how to debug every split decision
        for decision in tokenizer.debug_decisions(sentences):
            pprint(decision)
            print '=' * 30

    