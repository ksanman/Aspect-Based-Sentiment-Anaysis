

class DataAccess:
    """
    Class contains operations used to manage data io. 
    """

    def __init__(self):
        pass

    def get_training_reviews(self, partial_set, file_name='../review_data/train.ft.txt'):
        """
        Gets a set of reviews for training. 
        Returns a list of reviews. Each review is labeled for a positive or negative sentiment. 
        The partial_set parameter is an integer value used to indicate a partial_set of reviews
        to train. 
        """
        training_reviews = {'negative':[], 'positive':[]}
        iteration = 0
        with open(file_name, 'r') as f:
            for line in f:
                if '__label__1' in line:
                    training_reviews['negative'].append(line.replace('__label__1',''))
                if '__label__2' in line:
                    training_reviews['positive'].append(line.replace('__label__2',''))

                if partial_set & iteration >= (partial_set - 1):
                    break

                iteration += 1
        
        return training_reviews

    def get_test_reviews(self, partial_set, file_name='../review_data/test.ft.txt'):
        """
        Gets a set of reviews for testing. 
        Returns a list of reviews. Each review is labeled for a positive or negative sentiment. 
        The partial_set parameter is an integer value used to indicate a partial_set of reviews
        to train. 
        """
        testing_reviews = {'negative':[], 'positive':[]}
        iteration = 0
        with open(file_name, 'r') as f:
            for line in f:
                if '__label__1' in line:
                    testing_reviews['negative'].append(line.replace('__label__1',''))
                if '__label__2' in line:
                    testing_reviews['positive'].append(line.replace('__label__2',''))

                if partial_set & iteration >= (partial_set - 1):
                    break

                iteration += 1
        
        return testing_reviews