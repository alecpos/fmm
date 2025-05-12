import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Reshape, Concatenate
from tensorflow.keras.models import Model

from .cnn_feature_extractor import create_cnn_feature_extractor
from .lstm_code_generator import CodeGeneratorLSTM
from .attention_module import DualAttention

def create_pix2code_model(config):
    """Create the full pix2code model with dual attention."""
    
    # Define inputs
    img_input = Input(shape=config['img_shape'], name='image_input')
    code_input = Input(shape=(config['max_code_length'],), dtype=tf.int32, name='code_input')
    
    # CNN Feature Extractor
    cnn_model = create_cnn_feature_extractor(config['img_shape'])
    visual_features = cnn_model(img_input)
    
    # LSTM Code Generator
    lstm_model = CodeGeneratorLSTM(config['vocab_size'], config['embedding_dim'], config['lstm_units'])
    lstm_output, lstm_states = lstm_model(code_input, visual_features)
    
    # Project LSTM state to match attention units
    state_projection = Dense(config['lstm_units'])(lstm_states[0])
    
    # Dual Attention
    attention = DualAttention(units=config['lstm_units'])
    context, attention_weights = attention(state_projection, visual_features, lstm_output)
    
    # Combine context with LSTM output for final prediction
    final_context = Concatenate()([context, state_projection])
    
    # Reshape final_context to match sequence length
    final_context = tf.tile(
        tf.expand_dims(final_context, axis=1),
        [1, config['max_code_length'], 1]
    )
    
    # Final dense layer for next token prediction
    decoder_output = Dense(config['vocab_size'], activation='softmax', name='output_layer')(final_context)

    # Create model
    model = Model(inputs=[img_input, code_input], outputs=decoder_output)
    
    # Store components for testing
    model.cnn_model = cnn_model
    model.lstm_model = lstm_model
    model.attention = attention
    
    # Add method to get attention weights
    def get_attention_weights(inputs):
        img, code = inputs
        visual_features = cnn_model(img)
        lstm_output, lstm_states = lstm_model(code, visual_features)
        state_projection = Dense(config['lstm_units'])(lstm_states[0])
        _, attention_weights = attention(state_projection, visual_features, lstm_output)
        return attention_weights
    
    model.get_attention_weights = get_attention_weights
    
    return model 