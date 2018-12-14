

class DataAccess:
    """
    Class contains operations used to manage data io. 
    """

    def __init__(self):
        pass

    def get_data(self, number_of_reviews=1000, directory='../review_data/'):
        """ Gets an equal number of positive and negative reviews. """
        i = 0
        pos_reviews = []
        with open(directory + 'train_pos.txt', 'r') as p:
            for l in p:
                if i >= number_of_reviews:
                    break
                pos_reviews.append(l.replace('__label__2',''))
                i += 1
        i = 0
        neg_reviews = []
        with open(directory + 'train_neg.txt', 'r') as n:
            for l in n:
                if i >= number_of_reviews:
                    break
                neg_reviews.append(l.replace('__label__1', ''))     
                i += 1

        return {'positive': pos_reviews, 'negative':neg_reviews}
	
    def split_file(self, master_filepath, new_directory):
        """ Use this function to split the review data into positive and negative files. 
        This is so equal number of reviews can be pulled later. The master_filepath is the location
        of train.ft.txt (unzipped). The new directory is the path to the directory each
        file will be saved in. """

        negative_file = open(new_directory + 'train_neg.txt', 'w+')
        positive_file = open(new_directory + 'train_pos.txt', 'w+')
        with open(master_filepath, 'r') as f:
            for line in f:
                if '__label__2' in line:
                    positive_file.write(line.replace('__label__2', ''))
                else:
                    negative_file.write(line.replace('__label__1',''))

        negative_file.close()
        positive_file.close()


    def get_training_reviews(self, partial_set=None, file_name='../review_data/train.ft.txt'):
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
                if partial_set != None and  iteration >= (partial_set - 1):
                    break
                if '__label__1' in line:
                    training_reviews['negative'].append(line.replace('__label__1',''))
                if '__label__2' in line:
                    training_reviews['positive'].append(line.replace('__label__2',''))

                if partial_set != None and  iteration >= (partial_set - 1):
                    break

                iteration += 1
        
        return training_reviews

    def get_test_reviews(self, partial_set=None, file_name='../review_data/test.ft.txt'):
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

                if partial_set != None and iteration >= (partial_set - 1):
                    break

                iteration += 1
        
        return testing_reviews
