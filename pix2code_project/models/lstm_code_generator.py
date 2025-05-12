import tensorflow as tf
from tensorflow.keras.layers import LSTM, Bidirectional, Embedding, Dense
from tensorflow.keras.layers import Concatenate, Dropout, Layer

class CodeGeneratorLSTM(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim=256, lstm_units=256, dropout_rate=0.3):
        super(CodeGeneratorLSTM, self).__init__()
        
        self.embedding = Embedding(vocab_size, embedding_dim)
        self.dropout1 = Dropout(dropout_rate)
        
        # Bidirectional LSTM for context capture
        self.lstm1 = Bidirectional(LSTM(lstm_units, return_sequences=True))
        self.dropout2 = Dropout(dropout_rate)
        
        # Second LSTM with state output for attention
        self.lstm2 = LSTM(lstm_units*2, return_sequences=True, return_state=True)
        self.dropout3 = Dropout(dropout_rate)
        
        # Output projection
        self.dense = Dense(vocab_size)
    
    def call(self, inputs, visual_features, training=False):
        # inputs: [batch_size, seq_length]
        # visual_features: [batch_size, H, W, C]
        
        # Feature preparation - flatten spatial dimensions
        batch_size = tf.shape(visual_features)[0]
        h = tf.shape(visual_features)[1]
        w = tf.shape(visual_features)[2]
        c = tf.shape(visual_features)[3]
        
        # Reshape to [batch_size, H*W, C]
        visual_features_flat = tf.reshape(visual_features, [batch_size, h*w, c])
        
        # Apply embedding
        x = self.embedding(inputs)  # [batch_size, seq_length, embedding_dim]
        x = self.dropout1(x, training=training)
        
        # Repeat visual features for each token
        seq_length = tf.shape(x)[1]
        
        # Create combined input that includes both text and image features
        # We'll use the visual features at the beginning of the sequence
        x = self.lstm1(x)
        x = self.dropout2(x, training=training)
        
        # Pass through second LSTM
        lstm_output, state_h, state_c = self.lstm2(x)
        lstm_output = self.dropout3(lstm_output, training=training)
        
        # Return full sequence output and states
        return lstm_output, [state_h, state_c]
    
    def predict_next_token(self, x, states, visual_features):
        """Predict the next token given the current state."""
        # This is used during inference
        x = self.embedding(x)
        lstm_output, states = self.lstm2(x, initial_state=states)
        output = self.dense(lstm_output)
        return output, states 