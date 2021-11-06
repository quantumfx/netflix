
# netflix_ae.py - Fang Xi Lin, 2021

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow.keras.models as km
import tensorflow.keras.layers as kl
import tensorflow.keras.backend as K
import tensorflow as tf

import pandas as pd
import argparse

def load_data(data_path = 'ratings.sparse.small.csv', shuffle = True):
    """
    This functon loads Netflix ratings, normalize them to 0 to 1, convert
    them to tensors, then split them into a training (80 %) and test (20%)
    set.

    Although not necessary, the tensor conversion makes training a bit
    faster.

    Inputs:
        data_path: str, relative path to data file.
        shuffle: bool, whether to shuffle the data before splitting.

    Returns:
        float32 tensors: `data_train, data_test`.

        **data_train**: float32 tensor of normalized ratings
            with shape (num_users*0.8, 4499).

        **data_test, labels_test**: float32 tensor of normalized ratings
            with shape (num_users*0.2, 4499).
    """

    print('Loading Netflix data')
    df = pd.read_csv(data_path, header=0)

    # Convert to float and normalize
    df = df.astype('float32')
    df = df/5

    N = df.shape[0]

    if shuffle:
        print('Shuffling data')
        df = df.sample(frac=1)

    # Convert to tensor
    df = tf.constant(df)

    # Split into test and training
    data_train = df[N//5:]

    data_test = df[:N//5]

    return data_train, data_test

def mmse(input, output):
    """
    Masked Mean Square Error. This function computes the 'masked'
    mean squared error, calculated from only non-zero entries in the input.
    """
    nonzero_entries = K.not_equal(input, K.zeros_like(input))

    # factor of 5 for more intuitive numbers in loss function,
    # since we're normalizing the data
    reconstruction_loss = K.mean(K.square(5*tf.boolean_mask(input - output,
                            mask=nonzero_entries)))
    return reconstruction_loss

def rmse(input, output):
    """
    Root Masked Mean Square Error. Takes the square root of the MMSE
    to give a more intuitive difference as metric.
    """
    return K.sqrt(mmse(input, output))

def encoder_model(latent_dim = 128, input_shape = (4499), dropout_rate = 0.65):
    """
    Encoder Model. Consists of multiple fully connected layers
    with ELU followed by a dropout to ensure no feature dominates
    in training.
    """

    # Input
    encoder_input = kl.Input(shape = input_shape)

    # Deep layers
    x = kl.Dense(latent_dim, activation='elu')(encoder_input)
    x = kl.Dense(latent_dim, activation='elu')(x)
    x = kl.Dense(latent_dim, activation='elu')(x)
    x = kl.Dense(latent_dim, activation='elu')(x)
    x = kl.Dense(latent_dim, activation='elu')(x)

    # Final dropout
    encoded = kl.Dropout(dropout_rate)(x)

    model = km.Model(inputs = encoder_input, outputs = encoded)
    return model

def decoder_model(latent_dim = 128, output_shape = (4499)):
    """
    Decoder Model. Consists of a single dense hidden layer, followed by a
    dense sigmoid layer so the final output is between 0 and 1.
    """

    # The image inputted into the encoder.
    decoder_input = kl.Input(shape = (latent_dim))

    x = kl.Dense(latent_dim, activation='elu')(decoder_input)

    decoded = kl.Dense(output_shape, activation='sigmoid')(x)

    model = km.Model(inputs = decoder_input, outputs = decoded)
    return model

def autoencoder(latent_dim = 128, ranking_shape = (4499)):
    """
    Autoencoder Model. Stitches the encoder and decoder together. Note that
    it's asymmetric: the encoder is much deeper than the decoder.
    """

    input = kl.Input(shape = ranking_shape)

    encoder = encoder_model(latent_dim = latent_dim, input_shape = ranking_shape)
    decoder = decoder_model(latent_dim = latent_dim, output_shape = ranking_shape)

    output = decoder(encoder(input))

    model = km.Model(inputs = input, outputs = output)
    return model

def main(data_path = 'ratings.sparse.small.csv', batch_size = 64, epochs=50,
    verbose = True):
    # Load data
    data_train, data_test = load_data(data_path = data_path)

    # The network is an autoencoder, with Masked Mean Square Error as loss,
    # and Root Masked Mean Square Error as the metric
    print('Buinding network')
    model = autoencoder(latent_dim = 32)
    model.compile(loss = mmse, optimizer = 'adam', metrics = [rmse])

    print('Training network')
    N = data_train.shape[0]
    real_data_loss = []
    fake_data_loss = []

    # if interrupted, just evaluate the network
    # Each epoch is split between training on the data itself, and training on
    # the prediction of autoencoder(data), to enforce the fact that
    # autoencoders should be a fixed point of the data, ie.
    # autoencoder(data) = autoencoder(autoencoder(data))
    try:
        for ep in range(epochs):
                # first, build indices for shuffling data for the epoch
                idx_all = tf.random.shuffle(tf.range(data_train.shape[0]))
                for i in range(N//batch_size):
                    # train on real data every 4th batch
                    if (i%4 == 0):
                        # select on the relevant indices
                        idx = idx_all[i*batch_size:(i+1)*batch_size]
                        x = tf.gather(data_train, idx)

                        # train the network
                        l = model.train_on_batch(x, x)
                        real_data_loss.append(l)
                    # train on autoencoder(data) 3 out of 4 batches
                    else:
                        # select on the relevant indices
                        idx = idx_all[i*batch_size:(i+1)*batch_size]
                        x = tf.gather(data_train,idx)

                        # Train the network
                        # Calling model directly, as suggested in the
                        # documentation of model.predict (otherwise it
                        # is too slow)
                        xmodel = model(x, training=True)
                        l = model.train_on_batch(xmodel, xmodel)
                        fake_data_loss.append(l)
                if verbose:
                    print('Epoch {} of {}, loss: {} - rmse: {}'.format(ep+1, epochs,real_data_loss[-1][0],real_data_loss[-1][1]))
    except:
        pass

    print('Evalutating network')
    train_score = model.evaluate(data_train, data_train)
    test_score = model.evaluate(data_test, data_test)

    print('Training score is {}'.format(train_score))
    print('Test score is {}'.format(test_score))
    print('RMSE difference is {} %'.format(100 * (test_score[1] - train_score[1]) ) )

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Neural network for downsized\
                    Netflix dataset for collaborative filtering.')
    parser.add_argument('--data-path', help='Relative path to Netflix\
                    data file', default='ratings.sparse.small.csv')
    parser.add_argument('-b', '--batch-size', type=int, help='Batch size',
                    default=64)
    # early stopping at 50 seems to be the best testing score
    parser.add_argument('-e', '--epochs', type=int, help='Number to\
                    epochs to train', default=50)
    parser.add_argument('-v', '--verbose', action="store_true", help='Show \
                    loss per epoch', default=True)
    args = parser.parse_args()

    main(data_path = args.data_path, batch_size = args.batch_size,
        epochs = args.epochs, verbose = args.verbose)
