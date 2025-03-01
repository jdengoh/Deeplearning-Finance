#------ SVM model ----------
import joblib # type: ignore
import numpy as np # type: ignore
import pandas as pd # type: ignore
import spacy # type: ignore
from sklearn.metrics import accuracy_score # type: ignore
from sklearn.model_selection import train_test_split, KFold, cross_val_score # type: ignore
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix # type: ignore
import matplotlib.pyplot as plt # type: ignore
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score # type: ignore
from sklearn.externals import joblib # type: ignore
from copy import deepcopy

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, TensorDataset

def get_embeddings(spacy_model,news_headline):
    nlp = spacy.load(spacy_model)
    if spacy_model == 'en_core_web_sm':
        embedding_dims = 96
    elif spacy_model == 'en_core_web_lg':
        embedding_dims = 300
    all_vectors = []
    for s in news_headline:
        # Get the token vectors for the sentence
        tokens = [token.vector for token in nlp(s) if token.has_vector]
        
        # If tokens exist with vectors, compute the mean of the vectors
        if tokens:
            avg_vector = np.mean(tokens, axis=0)
        else:
            # If no valid token vectors, create a zero vector of the correct dimension
            avg_vector = np.zeros(embedding_dims)
        
        all_vectors.append(avg_vector)
    
    # Convert the list of vectors to a NumPy array
    all_vectors = np.array(all_vectors)
    
    print(all_vectors.shape)
    
    return all_vectors

def get_svm_model(weightsPath,news):

        embeddings = get_embeddings('en_core_web_lg',news)
        # Load the model
        model = SVC(probability=True) 
        model = joblib.load(weightsPath)
        predictions = model.predict(embeddings)
        print("Predictions:", predictions)
#------------- BiLSTM --------
class BiLSTMModel(nn.Module):
        def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, input_length):
            super().__init__()
            # Embedding layer
            self.embedding = nn.Embedding(vocab_size, embedding_dim)
            
            # BiLSTM layer with bidirectional=True
            self.bilstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, dropout=0.2, bidirectional=True)
            
            # Fully connected layer (hidden_dim * 2 due to bidirectional LSTM)
            self.fc = nn.Linear(hidden_dim * 2, output_dim)  # *2 because of bidirectional
            self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            # Get the embedding of the input
            x = self.embedding(x)
            
            # Pass through the BiLSTM
            lstm_out, (hn, cn) = self.bilstm(x)
            
            # Combine hidden states from both directions (forward and backward)
            out = torch.cat((hn[-2], hn[-1]), dim=1)  # Concatenate forward and backward hidden states
            
            # Pass through the fully connected layer
            out = self.fc(out)
            
            # Apply sigmoid activation for binary classification
            out = self.sigmoid(out)
            
            return out
def main():
    #test
    news_headlines = [
    "Global markets are fluctuating as new economic policies take shape.",
    "Technology stocks are soaring to new heights, experts say.",
    "The stock market is experiencing its biggest slump in years.",
    "Experts predict a shift in global energy consumption patterns."
    ]
    
    # Path to the saved SVM model
    model_path = '..//svm_model.pkl' # Change this to the correct path

    # Get the model and embeddings
    get_svm_model(model_path,news_headlines)

# Run the main function
if __name__ == "__main__":
    main()