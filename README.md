# Data Representations

preprocess.ipynb: Uses the message representation from Music Transformer. Each recording.npy file in preprocessed_data contains a numpy array of numpy arrays, one for each channel. Each inner numpy array is a message sequence for the respective channel. Each instruments.npy file in preprocessed_data is a 1D numpy array giving the MIDI program number for each channel (telling us the instrument).

preprocess_unified.ipynb: Uses the message representation from Music Transformer. Each recording.npy file in train_unified and test_unifed contains an Lx2 numpy array, containing pairings of messages and channel numbers. Each instruments.npy file in train_unified and test_unified contains a 1D numpy array giving the MIDI program number for each channel.

preprocess_sync.ipynb: Adapts the music representation from DeepJ where we represent time in discrete time steps, with 24 steps per second. At each time step, an instrument can play any number of notes and hold any number of notes that it was already playing at the previous time step. Each recording.npy file in preprocessed_data_sync contains an NxLxD numpy array (N is the number of channels, L is the number of time steps, and D is the dimension of actions (2 times the number of MIDI notes)).

Each of these has a corresponding data_to_midi file that produces a MIDI file from a recording.npy and instruments.npy file.

# Notebooks

lstm_baseline.ipynb: Concatenates all piano messages from all files into one long sequence. Trains an LSTM to generate the next message, given the previous message, randomly sampling sequences of length 200. There's a potential problem in that a sample could potentially contain the end of one piece concatenated with the start of another piece.

gru_baseline.ipynb: Same thing as lstm_baseline, uses a GRU.

transformer_baseline.ipynb: Same thing as lstm_baseline and gru_baseline, but uses a Transformer (encoder only).

The following two models (assigner and composer) are meant to be used in tandem. They both use LSTMs, though we could potentially replace them with Transformers.

assigner.ipynb: Uses the unified representation. Given a sequence of messages, predicts the channel associated with each message, given the instrument identity for each channel.

composer.ipynb: Uses the unified representation. Given a sequence of messages and a set of instruments, predicts the next message.

The following two are similar, but one uses an LSTM and the other uses a transformer.

train_unified.ipynb: Uses a Transformer. Takes a history of messages, concatenated with instrument embeddings, and predicts the next message and associated instrument. Overfit test fails.

train_unified_lstm.ipynb: Uses an LSTM. Same overall idea as train_unified, but the model itself is slightly different. Overfit test passes.

# Out-of-date/Incomplete Notebooks
train.ipynb: Uses the initial multi-instrument architecture we discussed, where the ensemble history is passed to the encoder, and the current instrument's history is passed to the decoder, and the decoder predicts the next message for the current instrument. Mask out any messages with an absolute timestamp ahead of the generation time (horribly slow). Passed the overfit test for two instruments.

train2.ipynb: Same as train.ipynb, but we mask out messages with an index ahead of the generation index. This isn't really conceptually correct. Passed the overfit test for two instruments.

train_sync.ipynb: Same idea as train and train2, but with the sync representation (so we don't have to worry about absolute message times being different for each instrument, even though the message index is the same). Overfit test fails (the network just generates all zeros).

trainxl: Not functional, but it was meant to be similar to train2, just using TransformerXL.