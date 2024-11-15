import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import os
import torch.nn as nn
import argparse

from utilities import Utilities
from tokenizer import SimpleTokenizer
from dataset import SpeechesClassificationDataset, LanguageModelingDataset
from transformer import EncoderAndClassifier, Decoder, EncoderAndClassifierMod


seed = 42

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

""" Hyperparameters to use for training to roughly match 
the numbers mentioned in the assignment description """
batch_size = 16  # Number of independent sequences  we will process in parallel
block_size = 32  # Maximum context length for predictions
learning_rate = 1e-3  # Learning rate for the optimizer
n_embd = 64  # Embedding dimension
n_head = 2  # Number of attention heads
n_layer = 4  # Number of transformer layers


eval_interval = 100  # How often to evaluate train and test perplexity during training
max_iters = 500 # For language modeling, we can process all the batches for the entire dataset, but that takes a while, so we'll limit it to 500 iterations. For batch size of 16 and block size of  32, this is roughly, this is  500 * 16 * 32 = 256000 tokens, SOTA LMs are trained on trillions of tokens, so this is a very small dataset.
eval_iters = 200  # Number of iterations to evaluate perplexity on the test set


## classifier training hyperparameters. It is a simple 1 hidden layer feedforward network, with input 
## size of 64, hidden size of 50 and output size of 3.

n_input = 64  # Input size for the classifier, should match the embedding size of the transformer
n_hidden = 100  # Hidden size for the classifier
n_output = 3  # Output size for the classifier, we have 3 classes
epochs_CLS = 15 # epochs for classifier training

def load_texts(directory):
    """
    This function loads all texts from the specified directory, ignoring any files with "test" in their name. The text is used for "training" the tokenizer. Since our tokenizer is simple, we don't need to do any training, but we still need to ignore the test data. 
    """

    texts = []
    files = os.listdir(directory)
    for filename in files: 
        if "test" in filename:  ## don't "read test files"
            continue
        with open(os.path.join(directory, filename), 'r', encoding='utf-8') as file:
            texts.append(file.read())
    return texts



def collate_batch(batch):
    """ Collate a batch of data into a single tensor with padding."""
    data, labels = zip(*batch)  # Separate the data and labels
    # Pad sequences to the fixed length
    padded_sequences = pad_sequence(data, batch_first=True, padding_value=0)
    padded_sequences = padded_sequences[:, :block_size]  # Truncate if longer
    # Add padding if shorter
    padded_sequences = torch.nn.functional.pad(padded_sequences, (0, max(0, block_size - padded_sequences.shape[1])), "constant", 0)
    labels = torch.stack(labels)  
    return padded_sequences, labels

def compute_classifier_accuracy(classifier, data_loader):
    """ Compute the accuracy of the classifier on the data in data_loader."""
    classifier.eval()
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for X, Y in data_loader:
            X, Y = X.to(device), Y.to(device)
            outputs = classifier(X)
            _, predicted = torch.max(outputs.data, 1)
            total_correct += (predicted == Y).sum().item()
            total_samples += Y.size(0)
        accuracy = (100 * total_correct / total_samples)
        classifier.train()
        return accuracy


def compute_perplexity(decoderLMmodel, data_loader, eval_iters=100):
    """ Compute the perplexity of the decoderLMmodel on the data in data_loader.
    Make sure to use the cross entropy loss for the decoderLMmodel.
    """
    decoderLMmodel.eval()
    losses= []
    total_loss = 0
    for X, Y in data_loader:
        X, Y = X.to(device), Y.to(device)
        loss,_ = decoderLMmodel(X, Y) # your model should be computing the cross entropy loss
        losses.append(loss.item())
        total_loss += loss.item()
        if len(losses) >= eval_iters: break


    losses = torch.tensor(losses)
    mean_loss = losses.mean()
    perplexity = torch.exp(mean_loss).item()  # Calculate perplexity as exp(mean loss)

    decoderLMmodel.train()
    return perplexity


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-part1", "--part1", action="store_true", help="If you want to run part 1")
    parser.add_argument("-part2", "--part2", action="store_true", help="If you want to run part 2")
    parser.add_argument("-part3", "--part3", action="store_true", help="If you want to run part 3")
    part1 = parser.parse_args().part1
    part2 = parser.parse_args().part2
    part3 = parser.parse_args().part3


    print("Loading data and creating tokenizer ...")
    texts = load_texts('speechesdataset')
    tokenizer = SimpleTokenizer(' '.join(texts)) # create a tokenizer from the data
    print("Vocabulary size is", tokenizer.vocab_size)

    train_CLS_dataset = SpeechesClassificationDataset(tokenizer, "speechesdataset/train_CLS.tsv")
    train_CLS_loader = DataLoader(train_CLS_dataset, batch_size=batch_size,collate_fn=collate_batch,shuffle=True)

  
    inputfile = "speechesdataset/train_LM.txt"
    with open(inputfile, 'r', encoding='utf-8') as f:
        lmtrainText = f.read()
    train_LM_dataset = LanguageModelingDataset(tokenizer, lmtrainText,  block_size)
    train_LM_loader = DataLoader(train_LM_dataset, batch_size=batch_size, shuffle=True)
    inputfile = "speechesdataset/test_LM_hbush.txt"
    with open(inputfile, 'r', encoding='utf-8') as f:
        lmtest_hbush_Text = f.read()
    test_LM_hbush_dataset = LanguageModelingDataset(tokenizer, lmtest_hbush_Text, block_size)
    test_LM_hbush_loader = DataLoader(test_LM_hbush_dataset, batch_size=batch_size, shuffle=True)
    inputfile = "speechesdataset/test_LM_obama.txt"
    with open(inputfile, 'r', encoding='utf-8') as f:
        lmtest_obama_Text = f.read()
    test_LM_obama_dataset = LanguageModelingDataset(tokenizer, lmtest_obama_Text, block_size)
    test_LM_obama_loader = DataLoader(test_LM_obama_dataset, batch_size=batch_size, shuffle=True)
    inputfile = "speechesdataset/test_LM_wbush.txt"
    with open(inputfile, 'r', encoding='utf-8') as f:
        lmtest_wbush_Text = f.read()
    test_LM_wbush_dataset = LanguageModelingDataset(tokenizer, lmtest_wbush_Text, block_size)
    test_LM_wbush_loader = DataLoader(test_LM_wbush_dataset, batch_size=batch_size, shuffle=True)
    inputfile = "speechesdataset/train_LM.txt"
    with open(inputfile, 'r', encoding='utf-8') as f:
        lmtrainText2 = f.read()
    train_LM_dataset2 = LanguageModelingDataset(tokenizer, lmtrainText2, block_size)
    train_LM_loader2 = DataLoader(train_LM_dataset2, batch_size=batch_size, shuffle=True)

    #PART 1
    if(part1):
        print('PART 1')
        encoderandclassifier = EncoderAndClassifier(n_layer, n_embd, n_head, tokenizer.vocab_size, block_size,
                                                    n_input, n_hidden, n_output)
        print(sum(p.numel() for p in encoderandclassifier.parameters()))
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(encoderandclassifier.parameters(), lr=learning_rate)
        test_CLS_dataset = SpeechesClassificationDataset(tokenizer, "speechesdataset/test_CLS.tsv")
        # for the classification  task, you will train for a fixed number of epochs like this:
        for epoch in range(epochs_CLS):
            size = len(train_CLS_loader.dataset)
            num_batches = len(train_CLS_loader)
            encoderandclassifier.train()
            train_loss, correct = 0,0
            for xb, yb in (train_CLS_loader):
                xb, yb = xb.to(device), yb.to(device)
                # CLS training code here
                #xb = xb.float()
                pred = encoderandclassifier(xb)
                #_, predicted = torch.max(pred.data, 1)
                loss = loss_fn(pred.float(), yb)
                train_loss += loss.item()
                correct += (pred.argmax(1) == yb).type(torch.float).sum().item()

                #Backpropagation
                optimizer.zero_grad()
                assert loss.requires_grad, "Loss does not require gradients"
                loss.backward()
                optimizer.step()


            average_train_loss = train_loss / num_batches
            accuracy = correct / size
            print(f'Epoch #{epoch}:\t train accuracy: {accuracy:.3f}\t, train loss: {average_train_loss:.3f}')


        test_CLS_loader = DataLoader(test_CLS_dataset, batch_size=batch_size, collate_fn=collate_batch, shuffle=True)
        final_accuracy = compute_classifier_accuracy(encoderandclassifier, test_CLS_loader)
        print(f'The final test accuracy is {final_accuracy}')

        ut = Utilities(tokenizer, encoderandclassifier.encoder)
        ut.sanity_check("And it's not just my belief.", block_size)
        #ut.sanity_check("Yet compassion is the work of a nation, not just a government.", block_size)


    #PART 2
    if (part2):
        print('PART 2')
        decoder = Decoder(n_layer, n_embd, n_head, tokenizer.vocab_size, block_size)
        print(sum(p.numel() for p in decoder.parameters()))
        optimizer = torch.optim.Adam(decoder.parameters(), lr=learning_rate)
        # for the language modeling task, you will iterate over the training data for a fixed number of iterations like this:
        decoder.train()
        for i, (xb, yb) in enumerate(train_LM_loader):
            if i > max_iters:
                break
            xb, yb = xb.to(device), yb.to(device)
            loss,_ = decoder(xb, yb)
            optimizer.zero_grad()
            #assert loss.requires_grad, "Loss does not require gradients"
            loss.backward()
            optimizer.step()

            if i%eval_interval == 0:
                perp1 = compute_perplexity(decoder, test_LM_hbush_loader, eval_iters=eval_interval)
                perp2 = compute_perplexity(decoder, test_LM_obama_loader, eval_iters=eval_interval)
                perp3 = compute_perplexity(decoder, test_LM_wbush_loader, eval_iters=eval_interval)
                perp4 = compute_perplexity(decoder, train_LM_loader2, eval_iters=eval_interval)

                print(f'Iteration #{i}:\tPerplexities: HBush: {perp1:.3f}, Obama: {perp2:.3f}, WBush: {perp3:.3f}, Train: {perp4:.3f}')


        ut2 = Utilities(tokenizer, decoder)
        ut2.sanity_check("And it's not just my belief.", block_size)
        #ut2.sanity_check("Yet compassion is the work of a nation, not just a government.", block_size)


    #PART 3
    if(part3):
        print('PART 3')
        #NEW LEARNING RATE TO TRAIN FOR PERFORMANCE IMPROVEMENT
        learning_rate2 = 3e-4
        encoderandclassifiermod = EncoderAndClassifierMod(n_layer, n_embd, n_head, tokenizer.vocab_size, block_size,
                                                    n_input, n_hidden, n_output)
        print(sum(p.numel() for p in encoderandclassifiermod.parameters()))
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(encoderandclassifiermod.parameters(), lr=learning_rate2)
        # for the classification  task, you will train for a fixed number of epochs like this:
        for epoch in range(epochs_CLS):
            size = len(train_CLS_loader.dataset)
            num_batches = len(train_CLS_loader)
            encoderandclassifiermod.train()
            train_loss, correct = 0, 0
            for xb, yb in (train_CLS_loader):
                xb, yb = xb.to(device), yb.to(device)
                # CLS training code here
                # xb = xb.float()
                pred = encoderandclassifiermod(xb)
                # _, predicted = torch.max(pred.data, 1)
                loss = loss_fn(pred.float(), yb)
                train_loss += loss.item()
                correct += (pred.argmax(1) == yb).type(torch.float).sum().item()

                # Backpropagation
                optimizer.zero_grad()
                assert loss.requires_grad, "Loss does not require gradients"
                loss.backward()
                optimizer.step()

            average_train_loss = train_loss / num_batches
            accuracy = correct / size
            print(f'Epoch #{epoch}:\t train accuracy: {accuracy:.3f}\t, train loss: {average_train_loss:.3f}')

        test_CLS_dataset = SpeechesClassificationDataset(tokenizer, "speechesdataset/test_CLS.tsv")
        test_CLS_loader = DataLoader(test_CLS_dataset, batch_size=batch_size, collate_fn=collate_batch, shuffle=True)
        final_accuracy = compute_classifier_accuracy(encoderandclassifiermod, test_CLS_loader)
        print(f'The final test accuracy is {final_accuracy}')

        #ut = Utilities(tokenizer, encoderandclassifiermod.encoder)
        #ut.sanity_check("And it's not just my belief.", block_size)

    



if __name__ == "__main__":
    main()
