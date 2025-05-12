# Implementation Guide: CNN-LSTM Architecture with Dual Attention for Mockup-to-Code Conversion

## 1. Environment Setup

First, set up your development environment with the necessary dependencies:

```bash
# Create and activate conda environment
conda create -n pix2code python=3.10
conda activate pix2code

# Install core dependencies
conda install -c pytorch pytorch=2.3.1 torchvision
pip install tensorflow==2.14.0
pip install tensorflow-addons==0.23.0
pip install keras-cv==0.8.2
pip install matplotlib pandas scikit-learn scikit-image
pip install opencv-python pillow nltk
```

Create a project structure:

```
pix2code_project/
├── data/
│   ├── preprocessing/
│   ├── dataset_utils.py
│   └── data_loader.py
├── models/
│   ├── cnn_feature_extractor.py
│   ├── lstm_code_generator.py
│   ├── attention_module.py
│   └── pix2code_model.py
├── training/
│   ├── loss_functions.py
│   ├── metrics.py
│   └── trainer.py
├── utils/
│   ├── visualization.py
│   └── evaluation.py
├── configs/
│   └── config.yaml
├── notebooks/
│   └── exploration.ipynb
├── results/
├── main.py
└── README.md
```

## 2. Data Preparation

Start by implementing the data loading and preprocessing scripts:

```python
# data/dataset_utils.py
import tensorflow as tf
import numpy as np
import os
import json
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

class CodeTokenizer:
    def __init__(self, vocab_size=5000):
        self.tokenizer = Tokenizer(num_words=vocab_size, 
                                  filters='',
                                  oov_token='<UNK>')
        self.vocab_size = vocab_size
        self.is_fit = False
        
    def fit_on_texts(self, texts):
        self.tokenizer.fit_on_texts(texts)
        self.is_fit = True
        
    def texts_to_sequences(self, texts):
        if not self.is_fit:
            raise ValueError("Tokenizer must be fit before encoding texts")
        return self.tokenizer.texts_to_sequences(texts)
    
    def sequences_to_texts(self, sequences):
        if not self.is_fit:
            raise ValueError("Tokenizer must be fit before decoding sequences")
        return self.tokenizer.sequences_to_texts(sequences)
    
    def save(self, path):
        with open(path, 'w') as f:
            json.dump({
                'word_index': self.tokenizer.word_index,
                'index_word': self.tokenizer.index_word,
                'vocab_size': self.vocab_size
            }, f)
    
    def load(self, path):
        with open(path, 'r') as f:
            data = json.load(f)
            self.tokenizer.word_index = data['word_index']
            self.tokenizer.index_word = data['index_word']
            self.vocab_size = data['vocab_size']
            self.is_fit = True
```

Now create the data loader:

```python
# data/data_loader.py
import tensorflow as tf
import numpy as np
import os
from .dataset_utils import CodeTokenizer

def preprocess_image(img_path, target_size=(256, 256)):
    """Preprocess a single image for the CNN."""
    img = tf.io.read_file(img_path)
    img = tf.image.decode_png(img, channels=3)
    img = tf.image.resize(img, target_size)
    img = tf.keras.applications.resnet.preprocess_input(img)
    return img

def preprocess_code(code_path, tokenizer, max_length=512):
    """Preprocess a single code file for the LSTM."""
    with open(code_path, 'r') as f:
        code = f.read()
    
    tokens = tokenizer.texts_to_sequences([code])[0]
    padded = pad_sequences([tokens], maxlen=max_length, padding='post')[0]
    return padded

class Pix2CodeDataset:
    def __init__(self, data_dir, img_size=(256, 256), max_code_length=512, 
                 batch_size=32, tokenizer=None):
        self.data_dir = data_dir
        self.img_size = img_size
        self.max_code_length = max_code_length
        self.batch_size = batch_size
        self.tokenizer = tokenizer or CodeTokenizer()
        
        # Directories
        self.img_dir = os.path.join(data_dir, 'images')
        self.code_dir = os.path.join(data_dir, 'codes')
        
        # Get all filenames
        self.img_files = sorted([f for f in os.listdir(self.img_dir) 
                              if f.endswith(('.png', '.jpg', '.jpeg'))])
        self.code_files = sorted([f for f in os.listdir(self.code_dir) 
                               if f.endswith(('.html', '.xml', '.code'))])
        
        # Sanity check
        assert len(self.img_files) == len(self.code_files), "Number of images and code files must match"
        
    def prepare_tokenizer(self):
        """Fit tokenizer on all code files."""
        all_codes = []
        for code_file in self.code_files:
            with open(os.path.join(self.code_dir, code_file), 'r') as f:
                all_codes.append(f.read())
        
        self.tokenizer.fit_on_texts(all_codes)
        return self.tokenizer
    
    def create_tf_dataset(self, shuffle=True):
        """Create a TensorFlow dataset for training/validation."""
        img_paths = [os.path.join(self.img_dir, f) for f in self.img_files]
        code_paths = [os.path.join(self.code_dir, f) for f in self.code_files]
        
        # Create dataset of paths
        dataset = tf.data.Dataset.from_tensor_slices((img_paths, code_paths))
        
        # Shuffle if needed
        if shuffle:
            dataset = dataset.shuffle(buffer_size=len(img_paths))
        
        # Preprocess data
        dataset = dataset.map(
            lambda img_path, code_path: (
                preprocess_image(img_path, self.img_size),
                preprocess_code(code_path, self.tokenizer, self.max_code_length)
            ),
            num_parallel_calls=tf.data.AUTOTUNE
        )
        
        # Batch and prefetch
        dataset = dataset.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
        
        return dataset
```

## 3. Implementing the Core Model Components

### 3.1 CNN Feature Extractor

Start with the CNN backbone for visual feature extraction:

```python
# models/cnn_feature_extractor.py
import tensorflow as tf
from tensorflow.keras.applications import ResNet50, VGG16
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation
from tensorflow.keras.layers import GlobalAveragePooling2D, Reshape, Add, Multiply
from tensorflow.keras import Model, Input

class ASPPModule(tf.keras.layers.Layer):
    """Atrous Spatial Pyramid Pooling Module for multi-scale context."""
    def __init__(self, filters=256):
        super(ASPPModule, self).__init__()
        self.conv1 = Conv2D(filters, 1, padding='same')
        self.conv3_1 = Conv2D(filters, 3, padding='same', dilation_rate=1)
        self.conv3_6 = Conv2D(filters, 3, padding='same', dilation_rate=6)
        self.conv3_12 = Conv2D(filters, 3, padding='same', dilation_rate=12)
        self.global_pool = GlobalAveragePooling2D()
        self.reshape = Reshape((1, 1, -1))
        self.conv_pool = Conv2D(filters, 1, padding='same')
        self.output_conv = Conv2D(filters, 1, padding='same')
        self.bn = BatchNormalization()
        self.activation = Activation('relu')
    
    def call(self, inputs):
        # Apply different atrous rates
        feat_1x1 = self.activation(self.bn(self.conv1(inputs)))
        feat_3x3_1 = self.activation(self.bn(self.conv3_1(inputs)))
        feat_3x3_6 = self.activation(self.bn(self.conv3_6(inputs)))
        feat_3x3_12 = self.activation(self.bn(self.conv3_12(inputs)))
        
        # Global pooling path
        feat_pool = self.global_pool(inputs)
        feat_pool = self.reshape(feat_pool)
        feat_pool = self.conv_pool(feat_pool)
        feat_pool = tf.image.resize(feat_pool, tf.shape(inputs)[1:3])
        
        # Concatenate and apply final convolution
        feats = tf.concat([feat_1x1, feat_3x3_1, feat_3x3_6, feat_3x3_12, feat_pool], axis=-1)
        feats = self.output_conv(feats)
        feats = self.bn(feats)
        return self.activation(feats)

class CoordinateAttention(tf.keras.layers.Layer):
    """Coordinate attention for spatial relationship preservation."""
    def __init__(self, reduction_ratio=32):
        super(CoordinateAttention, self).__init__()
        self.reduction_ratio = reduction_ratio
        
    def build(self, input_shape):
        channels = input_shape[-1]
        self.reduced_channels = max(8, channels // self.reduction_ratio)
        
        # Pooling layers
        self.pool_h = tf.keras.layers.Lambda(
            lambda x: tf.reduce_mean(x, axis=2, keepdims=True))
        self.pool_w = tf.keras.layers.Lambda(
            lambda x: tf.reduce_mean(x, axis=1, keepdims=True))
        
        # Shared MLP
        self.mlp_shared = tf.keras.Sequential([
            Conv2D(self.reduced_channels, 1, padding='same'),
            BatchNormalization(),
            Activation('relu')
        ])
        
        # Height attention
        self.mlp_h = Conv2D(channels, 1, padding='same')
        
        # Width attention
        self.mlp_w = Conv2D(channels, 1, padding='same')
    
    def call(self, inputs):
        h, w, c = inputs.shape[1:4]
        
        # Height path
        h_feat = self.pool_h(inputs)          # [B, H, 1, C]
        h_feat = self.mlp_shared(h_feat)      # [B, H, 1, C/r]
        h_feat = self.mlp_h(h_feat)           # [B, H, 1, C]
        
        # Width path
        w_feat = tf.transpose(inputs, [0, 2, 1, 3])  # [B, W, H, C]
        w_feat = self.pool_w(w_feat)                  # [B, W, 1, C]
        w_feat = self.mlp_shared(w_feat)              # [B, W, 1, C/r]
        w_feat = self.mlp_w(w_feat)                   # [B, W, 1, C]
        w_feat = tf.transpose(w_feat, [0, 2, 1, 3])   # [B, 1, W, C]
        
        # Generate attention maps
        attention = tf.sigmoid(h_feat + w_feat)  # [B, H, W, C]
        
        return inputs * attention

def create_cnn_feature_extractor(input_shape=(256, 256, 3)):
    """Create the CNN feature extractor with hybrid architecture."""
    inputs = Input(shape=input_shape)
    
    # ResNet-50 base
    resnet_base = ResNet50(
        weights='imagenet', 
        include_top=False, 
        input_shape=input_shape
    )
    
    # Freeze early layers for transfer learning
    for layer in resnet_base.layers[:100]:
        layer.trainable = False
    
    # Get feature maps
    features = resnet_base(inputs)
    
    # Apply ASPP for multi-scale context
    aspp_features = ASPPModule(filters=512)(features)
    
    # Apply Coordinate Attention
    coord_att_features = CoordinateAttention()(aspp_features)
    
    # Final output
    output_features = Conv2D(512, 1, padding='same')(coord_att_features)
    
    # Create model
    model = Model(inputs=inputs, outputs=output_features)
    
    return model
```

### 3.2 LSTM Code Generator

Now implement the LSTM-based code generator:

```python
# models/lstm_code_generator.py
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
```

### 3.3 Attention Module

Implement the dual attention mechanism:

```python
# models/attention_module.py
import tensorflow as tf
from tensorflow.keras.layers import Dense, Multiply, Add, Layer, Activation, Permute, Reshape

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

class DualAttention(Layer):
    """Co-attention framework with visual and syntactic gates."""
    def __init__(self, attention_units=256):
        super(DualAttention, self).__init__()
        self.visual_attention = BahdanauAttention(attention_units)
        self.syntactic_attention = BahdanauAttention(attention_units)
    
    def call(self, lstm_output, visual_features):
        # lstm_output: [batch_size, seq_length, lstm_units]
        # visual_features: [batch_size, h*w, visual_channels]
        
        batch_size = tf.shape(lstm_output)[0]
        
        # Get the last LSTM output as query for visual attention
        query_visual = lstm_output[:, -1, :]  # [batch_size, lstm_units]
        
        # Visual Attention Gate - Attend to visual features based on language context
        visual_context, visual_weights = self.visual_attention(
            query_visual, visual_features
        )  # [batch_size, visual_channels], [batch_size, h*w, 1]
        
        # Syntactic Attention Gate - Attend to language features based on visual context
        # We use the visual context as the query for the language attention
        syntactic_context, syntactic_weights = self.syntactic_attention(
            visual_context, lstm_output
        )  # [batch_size, lstm_units], [batch_size, seq_length, 1]
        
        # Fusion of contexts - Add for residual connection
        fused_context = Add()([visual_context, syntactic_context])
        
        return fused_context, {
            'visual_weights': visual_weights,
            'syntactic_weights': syntactic_weights
        }
```

### 3.4 Complete Pix2Code Model

Now integrate all components into the full model:

```python
# models/pix2code_model.py
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Reshape, Concatenate
from tensorflow.keras.models import Model

from .cnn_feature_extractor import create_cnn_feature_extractor
from .lstm_code_generator import CodeGeneratorLSTM
from .attention_module import DualAttention

def create_pix2code_model(vocab_size, img_shape=(256, 256, 3), 
                         embedding_dim=256, lstm_units=256):
    """Create the full pix2code model with dual attention."""
    
    # Define inputs
    img_input = Input(shape=img_shape, name='image_input')
    code_input = Input(shape=(None,), dtype=tf.int32, name='code_input')
    
    # CNN Feature Extractor
    cnn_model = create_cnn_feature_extractor(img_shape)
    visual_features = cnn_model(img_input)
    
    # Reshape visual features for attention
    batch_size = tf.shape(visual_features)[0]
    h = tf.shape(visual_features)[1]
    w = tf.shape(visual_features)[2]
    c = tf.shape(visual_features)[3]
    visual_features_flat = tf.reshape(visual_features, [batch_size, h*w, c])
    
    # LSTM Code Generator
    lstm_model = CodeGeneratorLSTM(vocab_size, embedding_dim, lstm_units)
    lstm_output, lstm_states = lstm_model(code_input, visual_features)
    
    # Dual Attention
    attention = DualAttention(attention_units=lstm_units)
    context, attention_weights = attention(lstm_output, visual_features_flat)
    
    # Combine context with LSTM output for final prediction
    final_context = Concatenate()([context, lstm_states[0]])
    
    # Final dense layer for next token prediction
    output = Dense(vocab_size)(final_context)
    
    # Build model
    model = Model(inputs=[img_input, code_input], outputs=output)
    
    return model
```

## 4. Training Pipeline

### 4.1 Loss Functions and Metrics

```python
# training/loss_functions.py
import tensorflow as tf

def sparse_categorical_crossentropy_loss(y_true, y_pred):
    """Standard sparse categorical crossentropy loss."""
    return tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction=tf.keras.losses.Reduction.NONE
    )(y_true, y_pred)

def masked_loss(y_true, y_pred):
    """Mask the loss to ignore padding tokens."""
    mask = tf.math.logical_not(tf.math.equal(y_true, 0))
    loss = sparse_categorical_crossentropy_loss(y_true, y_pred)
    
    mask = tf.cast(mask, dtype=loss.dtype)
    loss *= mask
    
    return tf.reduce_sum(loss) / tf.reduce_sum(mask)
```

```python
# training/metrics.py
import tensorflow as tf
import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

class BLEU4Metric(tf.keras.metrics.Metric):
    """BLEU-4 metric for evaluating generated code quality."""
    def __init__(self, name='bleu4', **kwargs):
        super(BLEU4Metric, self).__init__(name=name, **kwargs)
        self.total_score = self.add_weight(name='total_score', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')
        self.smoothing = SmoothingFunction().method1
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        """Update the metric state."""
        # Convert predictions from logits to token IDs
        pred_tokens = tf.argmax(y_pred, axis=-1)
        
        # Convert to numpy for NLTK
        y_true_np = y_true.numpy()
        pred_tokens_np = pred_tokens.numpy()
        
        batch_size = y_true_np.shape[0]
        scores = []
        
        # Calculate BLEU for each sequence in the batch
        for i in range(batch_size):
            # Remove padding (zeros)
            reference = [token for token in y_true_np[i] if token > 0]
            hypothesis = [token for token in pred_tokens_np[i] if token > 0]
            
            # Skip empty sequences
            if len(reference) == 0 or len(hypothesis) == 0:
                continue
                
            # Calculate BLEU-4 score
            try:
                score = sentence_bleu([reference], hypothesis, 
                                    weights=(0.25, 0.25, 0.25, 0.25),
                                    smoothing_function=self.smoothing)
                scores.append(score)
            except:
                continue
        
        # Update metric state
        if scores:
            avg_score = np.mean(scores)
            self.total_score.assign_add(avg_score)
            self.count.assign_add(1.0)
    
    def result(self):
        """Return the metric result."""
        return self.total_score / self.count if self.count > 0 else 0.0
    
    def reset_state(self):
        """Reset the metric state."""
        self.total_score.assign(0.0)
        self.count.assign(0.0)

class ExactMatchMetric(tf.keras.metrics.Metric):
    """Exact match metric for evaluating perfect code matches."""
    def __init__(self, name='exact_match', **kwargs):
        super(ExactMatchMetric, self).__init__(name=name, **kwargs)
        self.correct = self.add_weight(name='correct', initializer='zeros')
        self.total = self.add_weight(name='total', initializer='zeros')
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        """Update the metric state."""
        # Convert predictions from logits to token IDs
        pred_tokens = tf.argmax(y_pred, axis=-1)
        
        # Check exact matches
        matches = tf.cast(tf.reduce_all(tf.equal(y_true, pred_tokens), axis=1), tf.float32)
        
        # Update metric state
        self.correct.assign_add(tf.reduce_sum(matches))
        self.total.assign_add(tf.cast(tf.shape(y_true)[0], tf.float32))
    
    def result(self):
        """Return the metric result."""
        return self.correct / self.total if self.total > 0 else 0.0
    
    def reset_state(self):
        """Reset the metric state."""
        self.correct.assign(0.0)
        self.total.assign(0.0)
```

### 4.2 Trainer Implementation

```python
# training/trainer.py
import tensorflow as tf
import time
import os
from .loss_functions import masked_loss
from .metrics import BLEU4Metric, ExactMatchMetric

class Pix2CodeTrainer:
    def __init__(self, model, tokenizer, checkpoint_dir='./checkpoints'):
        self.model = model
        self.tokenizer = tokenizer
        self.checkpoint_dir = checkpoint_dir
        
        # Create checkpoint directory if it doesn't exist
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        
        # Setup optimizer with learning rate schedule
        self.learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=3e-4,
            decay_steps=10000,
            decay_rate=0.96
        )
        
        self.optimizer = tf.keras.optimizers.AdamW(
            learning_rate=self.learning_rate,
            weight_decay=0.05
        )
        
        # Metrics
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.val_loss = tf.keras.metrics.Mean(name='val_loss')
        self.bleu4 = BLEU4Metric()
        self.exact_match = ExactMatchMetric()
        
        # Checkpoint manager
        self.ckpt = tf.train.Checkpoint(model=self.model, optimizer=self.optimizer)
        self.ckpt_manager = tf.train.CheckpointManager(
            self.ckpt, self.checkpoint_dir, max_to_keep=5
        )
        
        # Load checkpoint if available
        if self.ckpt_manager.latest_checkpoint:
            self.ckpt.restore(self.ckpt_manager.latest_checkpoint)
            print(f"Restored from checkpoint: {self.ckpt_manager.latest_checkpoint}")
    
    @tf.function
    def train_step(self, img_batch, target_batch):
        """Execute a single training step."""
        # Create input and target sequences for teacher forcing
        # Input sequence is target with last token removed
        # Target sequence is target with first token removed
        inp_seq = target_batch[:, :-1]  # Input sequence
        tar_seq = target_batch[:, 1:]   # Target sequence
        
        with tf.GradientTape() as tape:
            # Forward pass
            predictions = self.model([img_batch, inp_seq], training=True)
            
            # Calculate loss with masking for padding
            loss = masked_loss(tar_seq, predictions)
        
        # Compute gradients and apply
        gradients = tape.gradient(loss, self.model.trainable_variables)
        
        # Clip gradients to avoid explosion
        gradients, _ = tf.clip_by_global_norm(gradients, clip_norm=1.0)
        
        # Apply gradients
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        
        # Update metrics
        self.train_loss.update_state(loss)
        self.bleu4.update_state(tar_seq, predictions)
        self.exact_match.update_state(tar_seq, predictions)
        
        return loss
    
    @tf.function
    def val_step(self, img_batch, target_batch):
        """Execute a single validation step."""
        # Create input and target sequences for teacher forcing
        inp_seq = target_batch[:, :-1]  # Input sequence
        tar_seq = target_batch[:, 1:]   # Target sequence
        
        # Forward pass
        predictions = self.model([img_batch, inp_seq], training=False)
        
        # Calculate loss with masking for padding
        loss = masked_loss(tar_seq, predictions)
        
        # Update metrics
        self.val_loss.update_state(loss)
        
        return loss
    
    def train(self, train_dataset, val_dataset, epochs=150, save_freq=5):
        """Train the model for specified number of epochs."""
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            start_time = time.time()
            
            # Reset metrics
            self.train_loss.reset_states()
            self.val_loss.reset_states()
            self.bleu4.reset_states()
            self.exact_match.reset_states()
            
            # Training loop
            for step, (img_batch, target_batch) in enumerate(train_dataset):
                loss = self.train_step(img_batch, target_batch)
                
                if step % 100 == 0:
                    print(f"Epoch {epoch+1} Step {step} Loss {loss.numpy():.4f}")
            
            # Validation loop
            for img_batch, target_batch in val_dataset:
                val_loss = self.val_step(img_batch, target_batch)
            
            # Print epoch results
            train_loss = self.train_loss.result()
            val_loss = self.val_loss.result()
            bleu4 = self.bleu4.result()
            exact_match = self.exact_match.result()
            
            print(f"Epoch {epoch+1}/{epochs}, Time: {time.time()-start_time:.2f}s")
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f}")
            print(f"BLEU-4: {bleu4:.4f}")
            print(f"Exact Match: {exact_match:.4f}")
            
            # Save checkpoint if better model found
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                ckpt_path = self.ckpt_manager.save()
                print(f"Saved checkpoint for epoch {epoch+1}: {ckpt_path}")
            
            # Save periodically
            if (epoch + 1) % save_freq == 0:
                ckpt_path = self.ckpt_manager.save()
                print(f"Periodic save for epoch {epoch+1}: {ckpt_path}") 
```

## 5. Main Script

Tie everything together with a main script:

```python
# main.py
import tensorflow as tf
import argparse
import os
import yaml

from data.data_loader import Pix2CodeDataset, CodeTokenizer
from models.pix2code_model import create_pix2code_model
from training.trainer import Pix2CodeTrainer

def main(args):
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set memory growth for GPUs
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    
    # Create tokenizer
    tokenizer = CodeTokenizer(vocab_size=config['model']['vocab_size'])
    
    # Create dataset
    train_dataset = Pix2CodeDataset(
        data_dir=os.path.join(args.data_dir, 'training'),
        img_size=tuple(config['data']['img_size']),
        max_code_length=config['data']['max_code_length'],
        batch_size=config['training']['batch_size'],
        tokenizer=tokenizer
    )
    
    val_dataset = Pix2CodeDataset(
        data_dir=os.path.join(args.data_dir, 'validation'),
        img_size=tuple(config['data']['img_size']),
        max_code_length=config['data']['max_code_length'],
        batch_size=config['training']['batch_size'],
        tokenizer=tokenizer
    )
    
    # Prepare tokenizer if not already prepared
    if args.train:
        print("Preparing tokenizer...")
        tokenizer = train_dataset.prepare_tokenizer()
        tokenizer.save(os.path.join(args.output_dir, 'tokenizer.json'))
    else:
        tokenizer.load(os.path.join(args.output_dir, 'tokenizer.json'))
    
    # Create TF datasets
    train_tf_dataset = train_dataset.create_tf_dataset(shuffle=True)
    val_tf_dataset = val_dataset.create_tf_dataset(shuffle=False)
    
    # Create model
    model = create_pix2code_model(
        vocab_size=tokenizer.vocab_size,
        img_shape=tuple(config['data']['img_size']) + (3,),
        embedding_dim=config['model']['embedding_dim'],
        lstm_units=config['model']['lstm_units']
    )
    
    # Create trainer
    trainer = Pix2CodeTrainer(
        model=model,
        tokenizer=tokenizer,
        checkpoint_dir=os.path.join(args.output_dir, 'checkpoints')
    )
    
    # Print model summary
    model.summary()
    
    # Train model if requested
    if args.train:
        print("Starting training...")
        trainer.train(
            train_dataset=train_tf_dataset,
            val_dataset=val_tf_dataset,
            epochs=config['training']['epochs'],
            save_freq=config['training']['save_freq']
        )
    
    # Save model
    if args.save_model:
        print("Saving model...")
        model.save(os.path.join(args.output_dir, 'model'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Pix2Code Training Script')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to dataset directory')
    parser.add_argument('--output_dir', type=str, default='./results',
                        help='Path to output directory')
    parser.add_argument('--train', action='store_true',
                        help='Train the model')
    parser.add_argument('--save_model', action='store_true',
                        help='Save the model after training')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    main(args) 

## 6. Configuration

Create a configuration file:

```yaml
# configs/config.yaml
data:
  img_size: [256, 256]
  max_code_length: 512

model:
  vocab_size: 5000
  embedding_dim: 256
  lstm_units: 256
  attention_units: 256

training:
  batch_size: 32
  epochs: 150
  learning_rate:
    initial: 0.0003
    decay_steps: 10000
    decay_rate: 0.96
  regularization:
    weight_decay: 0.05
    dropout: 0.3
  save_freq: 5
```

## 7. Next Steps

After implementing the core model components, here's what you should focus on next:

1. **Data Collection/Preparation**:
   - Gather mockup images and corresponding code from existing datasets or create your own
   - Split data into training, validation, and test sets

2. **Model Training**:
   - Start with a small subset of data to verify the implementation
   - Monitor training metrics (loss, BLEU-4, exact match)
   - Experiment with hyperparameters if needed

3. **Evaluation and Testing**:
   - Evaluate the model on a held-out test set
   - Compare with benchmarks mentioned in the task description
   - Analyze visual fidelity using metrics like SSIM, Element IoU, and Color ΔE

4. **Implementation Refinement**:
   - Address any issues that arise during training
   - Optimize performance (memory usage, training speed)
   - Implement additional features from the task description

5. **Documentation**:
   - Document the architecture, training process, and performance
   - Create visualization tools for attention maps and generation results

This implementation guide provides a solid foundation for the CNN-LSTM architecture with dual attention mechanisms for mockup-to-code conversion. Follow these steps to start building and training your model, and refer to the original research papers for additional details if needed. 

## 8. Testing and Evaluation

### 8.1 Unit Tests
We've implemented comprehensive unit tests for all model components:
- CNN feature extractor
- LSTM code generator
- Attention mechanisms
- Full model pipeline

### 8.2 End-to-End Testing
The test pipeline includes:
- Mockup image generation
- Sample code creation
- End-to-end model testing
- Attention visualization

### 8.3 Test Data Generation
The test suite includes utilities for:
- Generating synthetic mockup images
- Creating corresponding HTML code
- Tokenizing and preprocessing data
- Visualizing attention maps

### 8.4 Running Tests
To run the tests:
```bash
# Run all tests
python -m unittest discover -s tests

# Run specific test file
python tests/test_pipeline.py

# Run with coverage report
coverage run -m unittest discover -s tests
coverage report
```

### 8.5 Test Data Structure
```
tests/
├── test_data/
│   ├── test_mockup.png
│   └── test_code.html
├── test_components.py
├── test_pipeline.py
└── test_utils.py
```

### 8.6 Visualization Tools
The test suite includes visualization tools for:
- Attention maps
- Generated code
- Model predictions
- Training metrics

## 9. Next Steps

1. **Data Collection**:
   - Gather more diverse mockup images
   - Create a larger test dataset
   - Implement data augmentation

2. **Model Training**:
   - Train on the full dataset
   - Implement early stopping
   - Add learning rate scheduling

3. **Evaluation**:
   - Implement more metrics
   - Compare with baselines
   - Analyze failure cases

4. **Deployment**:
   - Create API endpoints
   - Implement batch processing
   - Add error handling

5. **Documentation**:
   - Add API documentation
   - Create usage examples
   - Document best practices 