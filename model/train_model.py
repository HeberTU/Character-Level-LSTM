import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from data_prep.encode_data import get_batches, one_hot_encode

def train(model, data, train_on_gpu=False, epochs = 10, batch_size = 10, seq_length = 50, lr = 0.001, clip = 5, val_frac = 0.1, print_status = 10):
    ''' Train a LSTM model using Adam optimizer and CreossEntropyLoss
        Arguments
        -----------

        model: CharLSTM
        data: text data to train the network
        train_on_gpu: flag to indicate whether we will train the net using GPU
        epochs: Total number of epochs to train the model
        batch_size: Number of mini-sequences per mini-batch
        seq_length: Number character steps per mini-batches
        lr: learning rate to be use in gradient descent algorithm
        clip: gradient clipping (in case gradient explodes)
        val_frac: Fraction of data to hold out for validation
        print_status: Number of steps before printing the training a validation loss
    '''

    model.train()
    # Criterion and optimization definition
    opt = torch.optim.Adam(model.parameters(), lr =lr)
    criterion = nn.CrossEntropyLoss()

    # Create Training and Validation data
    val_idx = int(len(data) * (1-val_frac))
    data = data[:val_idx]
    val_data = data[val_idx:]

    if (train_on_gpu):
        model.cuda()

    counter = 0
    n_chars = len(model.chars)
    for e in range(epochs):
        # Initialize the hidden state with all zeros
        h = model.init_hidden_state(batch_size,train_on_gpu)

        for x, y in get_batches(data, batch_size, seq_length):

            counter += 1
            x = one_hot_encode(x, n_chars)
            inputs = torch.from_numpy(x)
            targets = torch.from_numpy(y)

            if (train_on_gpu):
                inputs = inputs.cuda()
                targets = targets.cuda()

            # Detach the previous hidden state to implement the truncated version of BPTT
            h = tuple([each.data for each in h])

            model.zero_grad()

            output, h = model(inputs, h)

            loss = criterion(output, targets.view(batch_size*seq_length).long())
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), clip)
            opt.step()
            print("counter...{}".format(counter))
            if counter % print_status == 0:
                print("Entering validation loop")
                val_h = model.init_hidden_state(batch_size,train_on_gpu)
                val_losses = []
                model.eval()

                for x, y in get_batches(val_data, batch_size, seq_length):

                    x = one_hot_encode(x, n_chars)
                    x, y = torch.from_numpy(x), torch.from_numpy(y)


                    val_h = tuple([each.data for each in val_h])
                    inputs, targets = x, y
                    if(train_on_gpu):
                        inputs, targets = inputs.cuda(), targets.cuda()

                    output, val_h = model(inputs, val_h)
                    val_loss = criterion(output, targets.view(batch_size*seq_length).long())
                    val_losses.append(val_loss.item())


                model.train()

                print(
                "Epoch: {}/{}...".format(e+1, epochs),
                "Step: {}...".format(counter),
                "Loss: {:.4f}...".format(loss.item()),
                "Val Loss: {:.4f}...".format(np.mean(val_losses)))
