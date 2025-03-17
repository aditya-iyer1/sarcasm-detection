import torch
import torch.nn as nn
import json
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

class SarcasmLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, text):
        embedded = self.embedding(text)
        output, (hidden, cell) = self.lstm(embedded)
        hidden = hidden[-1]
        output = self.fc(hidden)
        return output

def load_data(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def evaluate_model(model, test_loader, device):
    model.eval()
    predictions = []
    actual = []
    
    with torch.no_grad():
        for batch in test_loader:
            texts = batch['text'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(texts)
            _, predicted = torch.max(outputs, 1)
            
            predictions.extend(predicted.cpu().numpy())
            actual.extend(labels.cpu().numpy())
    
    accuracy = accuracy_score(actual, predictions)
    report = classification_report(actual, predictions)
    
    return accuracy, report

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load the saved model
    model_path = '../sarcasm_lstm_GloVe2.pth'
    model_state = torch.load(model_path, map_location=device)
    
    # Initialize model with same parameters as training
    vocab_size = model_state['vocab_size']
    embedding_dim = model_state['embedding_dim']
    hidden_dim = model_state['hidden_dim']
    output_dim = model_state['output_dim']
    
    model = SarcasmLSTM(vocab_size, embedding_dim, hidden_dim, output_dim)
    model.load_state_dict(model_state['model_state_dict'])
    model.to(device)
    
    # Load and preprocess test data
    test_data = load_data('../data/Sarcasm_Headlines_Dataset_v2.json')
    
    # Evaluate model
    accuracy, report = evaluate_model(model, test_loader, device)
    
    print(f'Test Accuracy: {accuracy:.4f}')
    print('\nClassification Report:')
    print(report)

if __name__ == '__main__':
    main()