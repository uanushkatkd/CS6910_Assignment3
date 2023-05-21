import torch
# Print the name of the CUDA device, if available
print(torch.device('cuda:0'))
# Print the version of the torch library
print(torch.__version__)

# Create a variable to store the device to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Print the device that will be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# imports
import argparse
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import math
import torch
import torchvision
import torch.nn.functional as F  # Parameterless functions, like (some) activation functions
import torchvision.datasets as datasets  # Standard datasets
import torchvision.transforms as transforms  # Transformations we can perform on our dataset for augmentation
from torch import optim  # For optimizers like SGD, Adam, etc.
from torch import nn  # All neural network modules
from torch.utils.data import (
    DataLoader, random_split
)  # Gives easier dataset managment by creating mini batches etc.
from tqdm import tqdm  # For nice progress bar!

from torchvision.datasets import ImageFolder
import os
import random
import matplotlib.pyplot as plt
import numpy as np
import pathlib

"""
  This function sets the random seed for all major Python libraries.

  Args:
    seed (int): The random seed to use.

  """
def seed_everything(seed=1):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything()

# Data Preprocessing
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
'''
Class Vovabulary is used to create vocabulary from the training dataset.
'''
class Vocabulary:
    """
    Args:
      file_path (string): The path to the CSV file containing the training data.
      src_lang (string): The name of the source language.
      trg_lang (string): The name of the target language.

    Raises:
      ValueError: If the file_path does not exist.

    """
    def __init__(self, file_path, src_lang, trg_lang):
        # Read the CSV file into a Pandas DataFrame.
        self.translations = pd.read_csv(file_path, header=None, names=[src_lang, trg_lang])
        # It will drop any rows with missing values
        self.translations.dropna()
        self.src_lang = src_lang
        self.trg_lang = trg_lang
        # Create a dictionary that maps each character in the source language to an integer index.
        self.trg_vocab = {char: i+3 for i, char in enumerate(sorted(list(set(''.join(self.translations[trg_lang].tolist())))))}
        # Create a dictionary that maps each character in the target language to an integer index.
        self.src_vocab = {char: i+3 for i, char in enumerate(sorted(list(set(''.join(self.translations[src_lang].tolist())))))}
        
        # Add special tokens to the vocabularies.
        self.trg_vocab['<'] = 0
        self.src_vocab['<'] = 0

        self.trg_vocab['<unk>'] = 2
        self.src_vocab['<pad>'] = 1
        self.trg_vocab['<pad>'] = 1
        
        self.src_vocab['<unk>'] = 2
        
        # Extract the unique characters in the source and target languages
        src_chars = sorted(set(''.join(self.translations[src_lang])))
        trg_chars = sorted(set(''.join(self.translations[trg_lang])))

        # Assign an index to each character in the source and target languages
        self.t_char_to_idx = {char: idx+3 for idx, char in enumerate(trg_chars)}
        self.t_char_to_idx['<unk>']=2
        self.t_idx_to_char = {idx: char for char, idx in self.t_char_to_idx.items()}
        
        self.s_char_to_idx = {char: idx+3 for idx, char in enumerate(src_chars)}
        self.s_char_to_idx['<unk>']=2
        self.s_idx_to_char = {idx: char for char, idx in self.s_char_to_idx.items()}
        
      


    def get(self):
         # This function returns the source and target vocabularies, as well as the dictionaries that map characters to integer indexes and vice versa.
        return self.src_vocab,self.trg_vocab,self.t_char_to_idx,self.t_idx_to_char,self.s_char_to_idx,self.s_idx_to_char
        


class TransliterationDataset(Dataset):
    """
    Args:
      file_path (string): The path to the CSV file containing the training data.
      src_lang (string): The name of the source language.
      trg_lang (string): The name of the target language.
      src_vocab (Vocabulary): The vocabulary for the source language.
      trg_vocab (Vocabulary): The vocabulary for the target language.

    Raises:
      ValueError: If the file_path does not exist.

    """
    def __init__(self, file_path, src_lang, trg_lang,src_vocab,trg_vocab,t_char_to_idx):
        self.translations = pd.read_csv(file_path, header=None, names=[src_lang, trg_lang])
        self.translations.dropna()
    
        self.src_lang = src_lang
        self.t_char_to_idx = t_char_to_idx
        self.trg_lang = trg_lang
        self.src_vocab = src_vocab
        self.trg_vocab = trg_vocab
        self.max_src_len = max([len(word) for word in self.translations[src_lang].tolist()])+1
        #print("max src len",self.max_src_len)
        self.max_trg_len = max([len(word) for word in self.translations[trg_lang].tolist()])+1
        #print("max trg len",self.max_trg_len)
    def __len__(self):
        return len(self.translations)

    def target_to_one_hot(self, target_word, char_to_idx):
        num_trg_chars = len(char_to_idx)
        max_target_len = self.max_trg_len
        # Create a tensor of zeros for the one-hot encoding
        one_hot = torch.zeros((max_target_len, num_trg_chars))
        # Encode each character in the target word as a one-hot vector
        for i, char in enumerate(target_word):
            #print(i,char)
            char_idx = char_to_idx[char if char in  char_to_idx else '<unk>']
            #print(char_idx)
            one_hot[i][char_idx] = 1
        return one_hot

    def __getitem__(self, idx):
        src_word = self.translations.iloc[idx][self.src_lang]
        trg_word = self.translations.iloc[idx][self.trg_lang]
        #print(src_word)
        # Initialize the start-of-word token
        sow=0
        
        # Convert source and target words to lists of vocabulary indices
        src = [self.src_vocab.get(char, self.src_vocab['<unk>']) for char in src_word]
        trg = [self.trg_vocab.get(char, self.src_vocab['<unk>']) for char in trg_word]
        # Insert the start-of-word token at the beginning
        src.insert(0, sow)
        trg.insert(0, sow)

        src_len = len(src)
        trg_len = len(trg)

        # Pad the source and target sequences with the <pad> token
        src_pad = [self.src_vocab['<pad>']] * (self.max_src_len - src_len)
        trg_pad = [self.trg_vocab['<pad>']] * (self.max_trg_len - trg_len)

        # Extend the source and target sequences with padding
        src.extend(src_pad)
        trg.extend(trg_pad)

        # Convert source and target sequences to tensors
        src = torch.LongTensor(src)
        trg = torch.LongTensor(trg)
        #trg_one_hot = self.target_to_one_hot(trg_word, self.trg_vocab)
        #src_one_hot = self.target_to_one_hot(src_word, self.src_vocab)

        # This will return encoded source word ,target word and their length
        return src, trg, src_len, trg_len

def load_data(bs):
    '''
    This function loads data into batches provided the batch size as an argument.
    '''
    # Define the paths for the train, validation, and test CSV files
    train_path  ="/content/aksharantar_sampled/hin/hin_train.csv"
    val_path  ="/content/aksharantar_sampled/hin/hin_valid.csv"
    test_path  ="/content/aksharantar_sampled/hin/hin_test.csv"

    # Create a vocabulary object and retrieve the source and target vocabularies,
    # character-to-index and index-to-character mappings
    vocab = Vocabulary(train_path, 'src', 'trg')
    src_vocab,trg_vocab,t_char_to_idx,t_idx_to_char,s_char_to_idx,s_idx_to_char=vocab.get()
    #print(len(src_vocab))
    #print(len(trg_vocab))
    #print("char to idc outside",char_to_idx)


    # Create train, validation, and test datasets using TransliterationDataset
    # with the appropriate source and target vocabularies and mappings
    train_dataset = TransliterationDataset(train_path, 'src', 'trg',src_vocab,trg_vocab,t_char_to_idx)
    val_dataset = TransliterationDataset(val_path, 'src', 'trg',src_vocab,trg_vocab,t_char_to_idx)
    test_dataset = TransliterationDataset(test_path, 'src', 'trg',src_vocab,trg_vocab,t_char_to_idx)
    
    # Create train, validation, and test data loaders
    train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=bs, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=bs, shuffle=False)
    
    return train_loader,test_loader,val_loader,t_idx_to_char,s_idx_to_char


    
# Model

class Encoder(nn.Module):
    def __init__(self, input_dim, embedded_size,hidden_dim, num_layers,bidirectional, cell_type,dp):
        super(Encoder, self).__init__()
        # Initialize the attributes of the Encoder class
        self.input_dim = input_dim
        self.embedded_size=embedded_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.cell_type = cell_type
        self.bidirectional=bidirectional
        self.dropout = nn.Dropout(dp)
        
        # Linear layers for transforming the hidden state and cell state in case of bidirectional models
        self.fc_h = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc_c = nn.Linear(hidden_dim * 2, hidden_dim)
        
        # Determine the direction based on whether the model is bidirectional or not
 
        if self.bidirectional:
            self.dir=2
        else:
            self.dir=1  
        
        # Embedding layer to convert input sequences into dense vectors
        self.embedding = nn.Embedding(input_dim,embedded_size)
        
        # Select the appropriate RNN cell type based on the provided cell_type argument
        if cell_type == 'rnn':
              self.rnn = nn.RNN(embedded_size, hidden_dim, num_layers,bidirectional=bidirectional)
        elif cell_type == 'lstm':
              self.rnn = nn.LSTM(embedded_size, hidden_dim, num_layers,bidirectional=bidirectional)
        elif cell_type == 'gru':
              self.rnn = nn.GRU(embedded_size, hidden_dim, num_layers,bidirectional=bidirectional)
        else:
            raise ValueError("Invalid cell type. Choose 'rnn', 'lstm', or 'gru'.")
        
        

    def forward(self, src):
        # Apply dropout to the embedded input sequence
        embedded = self.dropout(self.embedding(src))
        if self.bidirectional:
            if self.cell_type == 'lstm':
                # Run the embedded sequence through the LSTM model
                output, (hidden, cell) = self.rnn(embedded)

                # Concatenate and transform the hidden and cell states from forward and backward directions
                hidden = self.fc_h(torch.cat((hidden[0:1], hidden[1:2]), dim=2))
                cell = self.fc_c(torch.cat((cell[0:1], cell[1:2]), dim=2))

                # Return the output sequence and the concatenated hidden and cell states
                return output, (hidden, cell)

            else:
                # Run the embedded sequence through the RNN or GRU model
                output, hidden = self.rnn(embedded)

                # Concatenate and transform the hidden states from forward and backward directions
                hidden = self.fc_h(torch.cat((hidden[0:1], hidden[1:2]), dim=2))
                
                # Return the output sequence and the concatenated hidden states
                return output,hidden
        
        else:
            # if bidirectional is set to False
            if self.cell_type == 'lstm':
                output, (hidden, cell) = self.rnn(embedded)
                return output, (hidden, cell)
            else:
                output, hidden = self.rnn(embedded)
                return output,hidden


        
        
class Decoder(nn.Module):
    def __init__(self, output_dim,embedded_size, hidden_dim, num_layers,bidirectional,cell_type,dp):
        super(Decoder, self).__init__()
        
        self.output_dim = output_dim
        self.embedded_size=embedded_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.cell_type = cell_type
        self.bidirectional=bidirectional
        self.dropout = nn.Dropout(dp)
        if self.bidirectional:
            self.dir=2
        else:
            self.dir=1  

        self.embedding = nn.Embedding(output_dim,embedded_size)
        
        if cell_type == 'rnn':
            self.rnn = nn.RNN((hidden_dim*self.dir)+embedded_size, hidden_dim, num_layers)
        elif cell_type == 'lstm':
            self.rnn = nn.LSTM((hidden_dim*self.dir)+embedded_size, hidden_dim, num_layers)
        elif cell_type == 'gru':
            self.rnn = nn.GRU((hidden_dim*self.dir)+embedded_size, hidden_dim, num_layers)
        else:
            raise ValueError("Invalid cell type. Choose 'rnn', 'lstm', or 'gru'.")

        self.energy = nn.Linear((hidden_dim *(self.dir+1) ), 1)
        self.dropout = nn.Dropout(dp)
        self.softmax = nn.Softmax(dim=0)
        self.relu = nn.ReLU()
        self.fc_out = nn.Linear(hidden_dim, output_dim)

        
    def forward(self, input,encoder_states,hidden):
        input = input.unsqueeze(0) # Add a time step dimension to the input
        #print("decoder input shape inside",input.shape)
        embedded = self.dropout(self.embedding(input))
                
        sequence_length = encoder_states.shape[0]
        #hl = encoder_output.shape[2]
        
        #bs = hidden.shape[1]
        #print("sequence length",sequence_length)
        h_reshaped = hidden[0].repeat(sequence_length,1,1)# Repeat the hidden state to match the sequence length
        #h_reshaped = sequence length x batch size x hidden size

        # Calculate attention energy       
        energy = self.relu(self.energy(torch.cat((h_reshaped,encoder_states), dim=2)))
        # energy: (seq_length, N, 1)
        #print("energy",energy.shape)
        
        attention = self.softmax(energy)# Calculate attention energy

        attention =attention.permute(1,2,0)
        encoder_states =encoder_states.permute(1,0,2)
        
        context_vector = torch.bmm(attention, encoder_states).permute(1,0,2)# Compute the context vector
        
        # Concatenate context vector and embedded input
        rnn_input = torch.cat((context_vector, embedded), dim=2)
        # rnn_input: (1, N, hidden_size*num_dir + embedding_size)

        
        output, hidden = self.rnn(rnn_input, hidden)
        
        output = self.fc_out(output)# Apply linear transformation for output prediction
        output = output.squeeze(0)
        
        return output, hidden

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder,cell_type,bidirectional):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.cell_type=cell_type
        self.bidirectional=bidirectional
        
    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size = trg.shape[1]# Get the batch size from the target sequence
        max_len = trg.shape[0]  # Get the maximum length of the target sequence
        trg_vocab_size = self.decoder.output_dim  # Get the vocabulary size of the decoder
        #print(batch_size)
        #print(max_len)
        #print(trg_vocab_size)

        # Tensor to store decoder outputs
        outputs = torch.zeros(max_len, batch_size, trg_vocab_size).to(device)
        
        # Pass the source sequence through the encoder
        encoder_states, encoder_hidden = self.encoder(src)
        #print("encoder hidden shape",encoder_hidden.shape)
        
        # Set the first decoder input as the first target token
        decoder_input = trg[0]
        #print("decoder input shape",decoder_input.shape)
        
        for t in range(1,max_len ):
             # Pass the decoder input and encoder states through the decoder
            decoder_output, decoder_hidden = self.decoder(decoder_input,encoder_states,encoder_hidden)

            outputs[t] = decoder_output

            # Determine the next decoder input based on teacher forcing ratio
            # Either use the true target token or the predicted token from the previous time step
 
            max_pr=decoder_output.argmax(1)
            decoder_input=trg[t] if random.random()<teacher_forcing_ratio else max_pr

        return outputs

def indices_to_string(trg, t_idx_to_char):
    """Converts a batch of indices to strings using the given index-to-char mapping
    Args:
    trg(Tensor):encoder words of size batch_size x sequence length
    t_idx_to_char(Dict.): index to char mapping
    
    """
    strings = []
    bs=trg.shape[0]
    sq=trg.shape[1]
    for i in range(bs):
        chars = []
      #print(i)
      # Convert the sequence of indices to a sequence of characters using the index-to-char mapping
        for j in range(sq):
            if trg[i,j].item() in t_idx_to_char:
                char = t_idx_to_char[trg[i,j].item()]
                chars.append(char)
      # Join the characters into a string
        string = ''.join(chars)
      #print(string)
        # Append the string to the list of strings
        strings.append(string)
    return strings

def calculate_word_level_accuracy(model,t_idx_to_char,data_loader, criterion):
    '''
    This function will calculate word level accuracy after each epoch.
    Args:
        model: The trained model
        t_idx_to_char: Mapping from target indices to characters
        data_loader: Data loader for the validation/test dataset
        criterion: Loss criterion used for training the model


    '''
    model.eval()
    num_correct = 0
    num_total = 0
    epoch_loss = 0
    
    with torch.no_grad():
        for batch_idx, (src, trg, src_len, trg_len) in enumerate(data_loader):
            # Convert target indices to string for comparison
            string_trg=indices_to_string(trg,t_idx_to_char)
            # Move tensors to the device
            src = src.permute(1, 0)
            trg = trg.permute(1, 0)
            src = src.to(device)
            trg = trg.to(device)
            # Perform forward pass through the model
            output = model(src, trg, 0)
            # turn off teacher forcing
            output = output[1:].reshape(-1, output.shape[2])
            #print("op after ",output.shape) # exclude the start-of-sequence token

            trg = trg[1:].reshape(-1) # exclude the start-of-sequence token
            #print("trg after reshape",trg.shape)
            
            # Calculate the loss
            output = output.to(device)
            loss = criterion(output, trg)
            epoch_loss += loss.item()
            
            batch_size = trg_len.shape[0]
            #print("bs", batch_size)
            seq_length = int(trg.numel() / batch_size)
            

            # Convert the output to predicted characters
            predicted_indices = torch.argmax(output, dim=1)
            predicted_indices = predicted_indices.reshape(seq_length,-1)
            predicted_indices = predicted_indices.permute(1, 0)
            # Convert predicted indices to strings
            string_pred=indices_to_string(predicted_indices,t_idx_to_char)
            #print(string_pred)
            #print(string_trg)
            
            for i in range(batch_size):
                num_total+=1
                # Compare the predicted string with the target string
                if string_pred[i][:len(string_trg[i])] == string_trg[i]:
                    num_correct+=1

    print("Total",num_total)
    print("Correct",num_correct*2)
    # Calculate word-level accuracy and average loss
    return ((num_correct*2) /num_total) * 100, (epoch_loss/(len(data_loader)))

def calculate_word_level_accuracy1(model,t_idx_to_char,s_idx_to_char,data_loader, criterion):
    
    '''
    This function is just extension of above function and
    will calculate word level accuracy as well as store the correct and 
    incorrect words into list .
    We'll call this function after training only once for test data.
    Args:
        model: The trained model
        t_idx_to_char: Mapping from target indices to characters
        s_idx_to_char: Mapping from source indices to characters
        data_loader: Data loader for the validation/test dataset
        criterion: Loss criterion used for training the model


    '''
    model.eval()
    num_correct = 0
    num_total = 0
    epoch_loss = 0
    c_trg=[]
    c_src=[]
    c_pred=[]
    
    i_trg=[]
    i_src=[]
    i_pred=[]
    

    with torch.no_grad():
        for batch_idx, (src, trg, src_len, trg_len) in enumerate(data_loader):
            # Convert target indices to string for comparison
            string_trg=indices_to_string(trg,t_idx_to_char)
            string_src=indices_to_string(src,s_idx_to_char)
            
            # Move tensors to the device
            src = src.permute(1, 0)
            trg = trg.permute(1, 0)
            src = src.to(device)
            trg = trg.to(device)
            # Perform forward pass through the model
            output = model(src, trg, 0)
            # turn off teacher forcing
            output = output[1:].reshape(-1, output.shape[2])
            #print("op after ",output.shape) # exclude the start-of-sequence token

            trg = trg[1:].reshape(-1) # exclude the start-of-sequence token
            #print("trg after reshape",trg.shape)
            
            # Calculate the loss
            output = output.to(device)
            loss = criterion(output, trg)
            epoch_loss += loss.item()
            
            batch_size = trg_len.shape[0]
            #print("bs", batch_size)
            seq_length = int(trg.numel() / batch_size)
            

            # Convert the output to predicted characters
            predicted_indices = torch.argmax(output, dim=1)
            predicted_indices = predicted_indices.reshape(seq_length,-1)
            predicted_indices = predicted_indices.permute(1, 0)
            # Convert predicted indices to strings
            string_pred=indices_to_string(predicted_indices,t_idx_to_char)
            #print(string_pred)
            #print(string_trg)
            
            for i in range(batch_size):
                num_total+=1
                # Compare the predicted string with the target string
                if string_pred[i][:len(string_trg[i])] == string_trg[i]:
                    c_trg.append(string_trg[i])
                    c_src.append(string_src[i])
                    c_pred.append(string_pred[i][:len(string_trg[i])])
                    num_correct+=1
                else :
                    i_trg.append(string_trg[i])
                    i_src.append(string_src[i])
                    i_pred.append(string_pred[i][:len(string_trg[i])])
                  


    print("Total",num_total)
    print("Correct",num_correct*2)
    # Calculate word-level accuracy and average loss
    return ((num_correct*2) /num_total) * 100, (epoch_loss/(len(data_loader))),c_trg,c_src,c_pred,i_trg,i_src,i_pred

# Define hyperparameters

parser = argparse.ArgumentParser()
parser.add_argument('-wp' , '--wandb_project', help='Project name used to track experiments in Weights & Biases dashboard' , type=str, default='CS6910_Assignment_2__Q2')
parser.add_argument('-we', '--wandb_entity' , help='Wandb Entity used to track experiments in the Weights & Biases dashboard.' , type=str, default='cs22s015')
parser.add_argument('-ct', '--cell_type', help="Choices:['rnn','gru','lstm']", type=str, default='gru')
parser.add_argument('-b', '--batch_size', help="Batch size used to train neural network.", type=int, default=128)
parser.add_argument('-o', '--optimizer', help = 'choices: [ "adam", "nadam"]', type=str, default = 'adam')
parser.add_argument('-lr', '--learning_rate', help = 'Learning rate used to optimize model parameters', type=float, default=0.0002)
parser.add_argument('-em', '--embedding_size', help='size of embedding', type=int, default=512)
parser.add_argument('-hs', '--hidden_size', help='choices:[64,128,256,512]',type=int, default=512)
parser.add_argument('-dp', '--dropout', help='choices:[0,0.2,0.3]',type=float, default=0.1)
parser.add_argument('-nl', '--num_layers', help='Number of layers in network ',type=int, default=3)
parser.add_argument('-bidir', '--bidirectional', help='Choices:["True","False"]',type=bool, default=False)
parser.add_argument('-tf', '--teacher_forcing', help='choices:[0,0.2,0.3,0.5,0.7]',type=float, default=0.7)


args = parser.parse_args()


# if __name__=='__main__':


# Define hyperparameters
INPUT_DIM = 29
OUTPUT_DIM = 67
EPOCHS = 25


es=args.embedding_size
hs = args.hidden_size
nl = args.num_layers
ct = args.cell_type
bs = args.batch_size
lr= args.learning_rate
tf = args.teacher_forcing
dp=args.dropout
bidir=args.bidirectional
opt=args.optimizer


# Load data and create data loaders
train_loader,test_loader,val_loader,t_idx_to_char,s_idx_to_char=load_data(bs)
#print(len(test_loader))
#print(len(train_loader))
#print(len(val_loader))
# Instantiate the Encoder and Decoder models
encoder = Encoder(INPUT_DIM,es,hs,nl,bidir,ct,dp).to(device)
decoder = Decoder(OUTPUT_DIM,es,hs,nl,bidir,ct,dp).to(device)

# Instantiate the Seq2Seq model with the Encoder and Decoder models
model = Seq2Seq(encoder, decoder,ct,bidir).to(device)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
if opt == "adam":
          optimizer = optim.Adam(model.parameters(),lr=lr)
elif opt == "nadam":
          optimizer= optim.NAdam(model.parameters(),lr=lr)
  
#optimizer=optimizer(model,opt,LEARNING_RATE)




# Train the model
for epoch in range(EPOCHS):
    epoch_loss = 0
    model.train()

    for batch_idx, (src, trg, src_len, trg_len) in enumerate(train_loader):
        #print(batch_idx)
        src = src.permute(1, 0)  # swapping the dimensions of src tensor
        trg = trg.permute(1, 0)  # swapping the dimensions of trg tensor

        src = src.to(device)
        trg = trg.to(device)
        
        optimizer.zero_grad()
        
        output = model(src, trg,tf)
        
        # Ignore the first element of the output, which is initialized as all zeros
        # since we use it to store the output for the start-of-sequence token
        #print(output.shape[2])
        
        output = output[1:].reshape(-1, output.shape[2])
        #print(output.shape)
        #print(trg.shape)
        trg = trg[1:].reshape(-1)
        
        loss = criterion(output, trg)
        loss.backward()
        
        optimizer.step()
        
        epoch_loss += (loss.item())
        
        if batch_idx % 1000 == 0:
            print(f"Epoch: {epoch}, Batch: {batch_idx}, Training...")

    # Calculate word-level accuracy after every epoch
    val_acc,val_loss = calculate_word_level_accuracy(model,t_idx_to_char,val_loader,criterion)
    
    print(f"Epoch: {epoch}, Loss: {epoch_loss / (len(train_loader))}, Val Acc: {val_acc}, Val loss: {val_loss}")
  #wandb.log({'epoch': epoch, 'loss': loss.item(), 'test_acc': test_acc,'train_acc': train_acc,'val_acc': val_acc})
    

val_acc,val_loss,c_trg,c_src,c_pred,i_trg,i_src,i_pred = calculate_word_level_accuracy1(model,t_idx_to_char,s_idx_to_char,test_loader,criterion)

# Save the best model
'''
best_model_path = 'best_model_AttnSeq2Seq.pth'
torch.save(model.state_dict(), best_model_path)
print(f"Best model saved to {best_model_path}")

'''
# Here we got the lists of correct source,target,predicted words
# And incorrect source,target,predicted words
# now we're writing it into csv.
#and later plotting it using Wandb
'''
import csv
def save_to_csv(src_list, trg_list, pred_list, file_name):
    rows = zip(src_list, trg_list, pred_list)

    with open(file_name, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['English', 'Target', 'Predicted'])
        writer.writerows(rows)

save_to_csv(c_src,c_trg,c_pred,'correct_predictions.csv')
save_to_csv(i_src,i_trg,i_pred,'incorrect_predictions.csv')
'''






'''
from signal import signal,SIGPIPE, SIG_DFL
signal(SIGPIPE,SIG_DFL)
!pip install wandb -qU
import wandb
!wandb login 

# Load the CSV file
ca_dataframe = pd.read_csv("/kaggle/working/correct_predictions.csv")
a_table = wandb.Table(dataframe=ca_dataframe)

# Add the table to an Artifact to increase the row 
# limit to 200000 and make it easier to reuse
ca_table_artifact = wandb.Artifact(
    "correct_predictions_Attn", 
    type="dataset"
    )        
ca_table_artifact.add(a_table, "Correct_predictions_Attn")

# Log the raw csv file within an artifact to preserve our data
ca_table_artifact.add_file("/kaggle/working/correct_predictions.csv")

# Display as a table


run = wandb.init(project='CS6910_Assignment3')

# Log the table to visualize with a run...
run.log({"Attn_correct_predictions_table": a_table})

# and Log as an Artifact to increase the available row limit!
run.log_artifact(ca_table_artifact)

# Load the CSV file
i_dataframe = pd.read_csv("/kaggle/working/incorrect_predictions.csv")
i_table = wandb.Table(dataframe=i_dataframe)

# Add the table to an Artifact to increase the row 
# limit to 200000 and make it easier to reuse
i_table_artifact = wandb.Artifact(
    "incorrect_predictions_Attn", 
    type="dataset"
    )        
i_table_artifact.add(i_table, "Incorrect_predictions_Attn")

# Log the raw csv file within an artifact to preserve our data
i_table_artifact.add_file("/kaggle/working/incorrect_predictions.csv")

# Display as a table


run = wandb.init(project='CS6910_Assignment3')

# Log the table to visualize with a run...
run.log({"Attn_incorrect_predictions_table": i_table})

# and Log as an Artifact to increase the available row limit!
run.log_artifact(i_table_artifact)


# wandb sweeps

sweep_config= {
    "name" : "CS6910_Assignment3",
    "method" : "bayes",
    'metric': {
        'name': 'val_acc',
        'goal': 'maximize'
    },
    'parameters' : {
        'cell_type' : { 'values' : ['lstm','gru','rnn'] },
        'dropout' : { 'values' : [0,0.1,0.2,0.5]},
        'embedding_size' : {'values' : [64,128,256,512]},
        'num_layers' : {'values' : [1]},
        'batch_size' : {'values' : [32,64,128]},
        'hidden_size' : {'values' : [128,256,512]},
        'bidirectional' : {'values' : [True ,False]},
        'learning_rate':{
            "values": [0.001,0.002,0.0001,0.0002]
        },
        'optim':{
            "values": ['adam','nadam']
        },
        'teacher_forcing':{"values":[0.2,0.5,0.7]}
    }
}



def train():
    wandb.init()

    c= wandb.config
    name = "cell_type_"+str(c.cell_type)+"_num_layers_"+str(c.num_layers)+"_dp_"+str(c.dropout)+"_bidir_"+str(c.bidirectional)+"_lr_"+str(c.learning_rate)+"_bs_"+str(c.batch_size)
    wandb.run.name=name
  
    # Retrieve the hyperparameters from the config
    ct=c.cell_type
    dp = c.dropout
    em=c.embedding_size
    nlayer=c.num_layers
    bs = c.batch_size
    hs=c.hidden_size
    bidir = c.bidirectional
    lr = c.learning_rate
    opt= c.optim
    epochs = 25
    tf=c.teacher_forcing
    trg_pad_idx=0

  

    INPUT_DIM = 29
    OUTPUT_DIM = 67

  
  # Load the dataset
    train_loader,val_loader,test_loader,t_idx_to_char,s_idx_to_char=load_data(bs)
   
  #print("data loaded ====================================================")

  # Instantiate the Encoder and Decoder models
    encoder = Encoder(INPUT_DIM,em,hs,nlayer,bidir,ct,dp).to(device)
    decoder = Decoder(OUTPUT_DIM,em,hs,nlayer,bidir,ct,dp).to(device)

  # Instantiate the Seq2Seq model with the Encoder and Decoder models
    model = Seq2Seq(encoder,decoder,ct,bidir).to(device)
  #print("model ini==============================================================")
 
  # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()      
    if opt == "adam":
          optimizer = optim.Adam(model.parameters(),lr=lr)
    elif opt == "nadam":
          optimizer= optim.NAdam(model.parameters(),lr=lr)
  
  # Train Network
    for epoch in range(epochs):
        epoch_loss = 0
        model.train()

        for batch_idx, (src, trg, src_len, trg_len) in enumerate(train_loader):
            src = src.permute(1, 0)  # swapping the dimensions of src tensor
            trg = trg.permute(1, 0)  # swapping the dimensions of trg tensor

            src = src.to(device)
            trg = trg.to(device)
            #print("done")
            optimizer.zero_grad()
            #print("doe")
            output = model(src,trg,tf)
            #print("doe")

            # Ignore the first element of the output, which is initialized as all zeros
            # since we use it to store the output for the start-of-sequence token
            #print(output.shape[2])

            output = output[1:].reshape(-1, output.shape[2])
            #print(output.shape)
            #print(trg.shape)
            trg = trg[1:].reshape(-1)

            loss = criterion(output, trg)
            loss.backward()

            optimizer.step()

            epoch_loss += loss.item()

            if batch_idx % 1000 == 0:
                print(f"Epoch: {epoch}, Batch: {batch_idx} , Training..")
        
        # Calculate word-level accuracy after every epoch
        #train_acc ,train_loss= calculate_word_level_accuracy(model, train_loader,criterion)
        val_acc,val_loss = calculate_word_level_accuracy(model,t_idx_to_char, val_loader, criterion)
        test_acc,test_loss = calculate_word_level_accuracy(model,t_idx_to_char, test_loader, criterion)
     
    #print(f"Epoch: {epoch}, Loss: {epoch_loss / len(train_loader)}, Train Acc: {train_acc}, Val Acc: {val_acc}")

            
    # Log the metrics to WandB
        wandb.log({'epoch': epochs, 'train_loss': loss.item(), 'test_acc': test_acc,'val_acc': val_acc,'test_loss': test_loss,'val_loss': val_loss})
    # Save the best model
    wandb.run.save()
    wandb.run.finish()
    return

# final train
# Initialize the WandB sweep
# sweep_id = wandb.sweep(sweep_config, project='CS6910_Assignment3')
# print(sweep_id, type(sweep_id))
wandb.agent('72j6yzep', function=train,count=40, project='CS6910_Assignment3')

'''



