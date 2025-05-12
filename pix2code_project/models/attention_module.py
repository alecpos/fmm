import tensorflow as tf
from tensorflow.keras.layers import Dense, Multiply, Add, Layer, Activation, Permute, Reshape, Concatenate

class BahdanauAttention(Layer):
    """Bahdanau-style attention mechanism."""
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = Dense(units)
        self.W2 = Dense(units)
        self.V = Dense(1)
    
    def call(self, query, values):
        # query: hidden state tensor of shape [batch_size, units]
        # values: feature tensor of shape [batch_size, seq_len, features]
        
        # Expand query dimensions to match feature tensor for broadcasting
        query_expanded = tf.expand_dims(query, 1)  # [batch_size, 1, units]
        
        # Score each feature with the query
        score = self.V(tf.nn.tanh(
            self.W1(query_expanded) + self.W2(values)
        ))  # [batch_size, seq_len, 1]
        
        # Calculate attention weights
        attention_weights = tf.nn.softmax(score, axis=1)  # [batch_size, seq_len, 1]
        
        # Apply attention weights to values
        context = attention_weights * values  # [batch_size, seq_len, features]
        context = tf.reduce_sum(context, axis=1)  # [batch_size, features]
        
        return context, attention_weights

class DualAttention(tf.keras.layers.Layer):
    def __init__(self, units):
        super(DualAttention, self).__init__()
        self.units = units
        self.attention1 = Dense(units)
        self.attention2 = Dense(units)
        self.visual_proj = Dense(units)  # Add projection for visual features
        self.concat = Concatenate()
        self.dense = Dense(units)
        
    def call(self, query, visual_features, lstm_output):
        # Reshape visual features if needed
        if len(visual_features.shape) == 4:  # [batch, height, width, channels]
            batch_size = tf.shape(visual_features)[0]
            h = tf.shape(visual_features)[1]
            w = tf.shape(visual_features)[2]
            c = tf.shape(visual_features)[3]
            visual_features = tf.reshape(visual_features, [batch_size, h*w, c])
        
        # Project query to match feature dimensions
        query_proj = self.attention1(query)  # [batch, units]
        query_expanded = tf.expand_dims(query_proj, 1)  # [batch, 1, units]
        
        # Project visual features to match query dimension
        visual_proj = self.visual_proj(visual_features)  # [batch, h*w, units]
        
        # Project LSTM output to match query dimension
        lstm_proj = self.attention2(lstm_output)  # [batch, seq_len, units]
        
        # Visual attention
        attention_weights1 = tf.nn.softmax(
            tf.matmul(query_expanded, visual_proj, transpose_b=True), axis=-1
        )  # [batch, 1, h*w]
        context1 = tf.matmul(attention_weights1, visual_proj)  # [batch, 1, units]
        
        # Syntactic attention
        attention_weights2 = tf.nn.softmax(
            tf.matmul(query_expanded, lstm_proj, transpose_b=True), axis=-1
        )  # [batch, 1, seq_len]
        context2 = tf.matmul(attention_weights2, lstm_proj)  # [batch, 1, units]
        
        # Combine contexts
        context1 = tf.squeeze(context1, axis=1)  # [batch, units]
        context2 = tf.squeeze(context2, axis=1)  # [batch, units]
        combined_context = self.concat([context1, context2])
        
        # Project to final dimension
        output = self.dense(combined_context)
        
        return output, {
            'visual_weights': attention_weights1,
            'syntactic_weights': attention_weights2
        } 