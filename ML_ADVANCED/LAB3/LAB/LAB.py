#%% md
# # Lab 3 Discriminative and generative models
#%% md
# 
#%% md
# ### 0. Imports
#%%
import os
from  collections import Counter
import numpy as np
#%% md
# ## 1. Divide the data in two groups: training and test examples.
#%%
# Directories for training and testing data
train_dir = './data/train-mails'
test_dir = './data/test-mails'
#%% md
# ## 2. Parse both the training and test examples to generate both the spam and ham data sets.
#%%
def parse_emails(directory):
    """Parse emails from a directory and categorize as spam or ham based on filename prefix
    
    Args:
        directory (str): Directory containing email files
        
    Returns:
        tuple: (ham_emails, spam_emails) lists of file paths
    """
    ham_emails = []
    spam_emails = []
    
    # Get all files in the directory
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        
        # Check if it's spam or ham based on filename prefix
        # In the ling-spam dataset, filenames starting with 'spmsg' are spam
        if filename.startswith('spmsg'):
            spam_emails.append(filepath)
        else:
            ham_emails.append(filepath)
    
    return ham_emails, spam_emails

#%%
# Parse training emails
train_ham_emails, train_spam_emails = parse_emails(train_dir)
print(f"Training data: {len(train_ham_emails)} ham emails, {len(train_spam_emails)} spam emails")

# Parse test emails
test_ham_emails, test_spam_emails = parse_emails(test_dir)
print(f"Test data: {len(test_ham_emails)} ham emails, {len(test_spam_emails)} spam emails")

#%% md
# ## 3. Generate a dictionary from the training data.
#%%
# Function to create a dictionary from email text data
def make_Dictionary(train_dir):
    emails = [os.path.join(train_dir, f) for f in os.listdir(train_dir)]
    all_words = []
    for mail in emails:
        with open(mail, encoding='latin1') as m:
            for i, line in enumerate(m):
                if i == 2:  # Usually, the 3rd line contains useful content
                    words = line.split()
                    all_words += words

    dictionary = Counter(all_words)

    # Removing non-alphabetic words and single-character words
    for item in list(dictionary):
        if not item.isalpha() or len(item) == 1:
            del dictionary[item]

    
    return dictionary
#%%
train_dictionary = make_Dictionary(train_dir)
#%%
# print the number of keys in the dictionary
print(len(train_dictionary))
#%% md
# ## 4. Extract features from both the training data and test data.
#%%
def extract_features(mail_dir, dictionary):
    files = [os.path.join(mail_dir, fi) for fi in os.listdir(mail_dir)]
    features_matrix = np.zeros((len(files), 3000))

    for docID, fil in enumerate(files):
        with open(fil, encoding='latin1') as fi:
            for i, line in enumerate(fi):
                if i == 2:  # Usually the subject/content line
                    words = line.split()
                    for word in words:
                        for wordID, d in enumerate(dictionary):
                            if d[0] == word:
                                features_matrix[docID, wordID] = words.count(word)

    return features_matrix

#%%
train_features = extract_features(train_dir, train_dictionary)
print(train_features)
#%%
test_dictionary = make_Dictionary(test_dir)
test_features = extract_features(test_dir, test_dictionary)
print(test_features)
#%% md
# ## 5. Implement the Naïve Bayes from scratch, and fit it to the training data.
#%%
class NaiveBayes:
    def __init__(self, train_features, train_labels, alpha=1.0):
        self.classes = ['ham', 'spam']
        self.alpha = alpha
        self.class_prior = {}
        self.class_conditional_prob = {}
        self.vocab_size = 3000
        self.train_features = train_features
    
    def fit(self, X, y):
        self.class_prior = {}
        self.class_conditional_prob = {}
        self.vocab_size = train_features.shape[1]
        self.train_features = train_features
        self.train_labels = train_labels
    
    def predict(self, X):
        predictions = []
        for x in X:
            posteriors = []
            for c in self.classes:
                prior = np.log(self.class_prior[c])
    
    def plot_confusion_matrix(self, y_true, y_pred):
        import matplotlib.pyplot as plt

        
        # Calculate confusion matrix manually
        cm = np.zeros((2,2))
        for i in range(len(y_true)):
            if y_true[i] == y_pred[i] == 'ham':
                cm[0,0] += 1 
            elif y_true[i] == y_pred[i] == 'spam':
                cm[1,1] += 1
            elif y_true[i] == 'ham' and y_pred[i] == 'spam':
                cm[0,1] += 1
            else:
                cm[1,0] += 1
                
        plt.figure(figsize=(10, 7))
        
#%% md
# ## 6. Make predictions for the test data.
#%%

#%% md
# ## 7. Measure the spam-filtering performance for each approach through the confusion matrix,
# accuracy, precision, recall, and F1 score.
#%%

#%% md
# ## 8. Plot a graph with true positive rate on the vertical axis and with false positive rate on the
# horizontal axis
#%%

#%% md
# ## 9. Discuss the results
#%% md
# agfjsqcq    mojb