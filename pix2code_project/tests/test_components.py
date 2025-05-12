import tensorflow as tf
import numpy as np
import os
import sys
import unittest

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataset_utils import CodeTokenizer
from data.data_loader import Pix2CodeDataset, preprocess_image, preprocess_code
from models.cnn_feature_extractor import create_cnn_feature_extractor, ASPPModule, CoordinateAttention
from models.lstm_code_generator import CodeGeneratorLSTM
from models.attention_module import BahdanauAttention, DualAttention
from models.pix2code_model import create_pix2code_model
from training.loss_functions import masked_loss
from training.metrics import BLEU4Metric, ExactMatchMetric

class TestComponents(unittest.TestCase):
    def setUp(self):
        self.batch_size = 2
        self.max_code_length = 50
        self.vocab_size = 1000
        self.image_size = (224, 224)
        self.feature_dim = 512  # Update this to match your model's feature dimension
        
        # Create test data
        self.test_image = tf.random.normal((self.batch_size, *self.image_size, 3))
        self.test_code = tf.random.uniform(
            (self.batch_size, self.max_code_length),
            minval=0,
            maxval=self.vocab_size,
            dtype=tf.int32
        )

    def test_cnn_feature_extractor(self):
        """Test CNN feature extractor components."""
        # Test ASPP Module
        aspp = ASPPModule(filters=256)
        test_input = tf.random.normal((1, 32, 32, 512))
        output = aspp(test_input)
        self.assertEqual(output.shape, (1, 32, 32, 256))
        
        # Test Coordinate Attention
        coord_att = CoordinateAttention()
        output = coord_att(test_input)
        self.assertEqual(output.shape, test_input.shape)
        
        # Test full CNN feature extractor
        cnn_model = create_cnn_feature_extractor(input_shape=(*self.image_size, 3))
        features = cnn_model(self.test_image)
        self.assertEqual(features.shape, (self.batch_size, 7, 7, 512))
    
    def test_lstm_code_generator(self):
        """Test LSTM code generator."""
        lstm_model = CodeGeneratorLSTM(
            vocab_size=self.vocab_size,
            embedding_dim=256,
            lstm_units=512
        )
        
        # Test forward pass
        output, states = lstm_model(
            self.test_code,
            tf.random.normal((self.batch_size, 7, 7, 512))
        )
        
        self.assertEqual(output.shape, (self.batch_size, self.max_code_length, 1024))
        self.assertEqual(len(states), 2)  # Should return hidden and cell states
    
    def test_attention_mechanism(self):
        """Test attention components."""
        # Create test inputs with matching dimensions
        batch_size = 2
        seq_len = 64
        units = 256
        features = 256  # Changed from 512 to match units
        
        query = tf.random.normal((batch_size, units))
        features1 = tf.random.normal((batch_size, seq_len, features))
        features2 = tf.random.normal((batch_size, seq_len, features))
        
        # Create and test attention mechanism
        dual_att = DualAttention(units=units)
        context, attention_weights = dual_att(query, features1, features2)
        
        # Check output shapes
        self.assertEqual(context.shape, (batch_size, units))
        self.assertIn('visual_weights', attention_weights)
        self.assertIn('syntactic_weights', attention_weights)
        self.assertEqual(attention_weights['visual_weights'].shape, (batch_size, 1, seq_len))
        self.assertEqual(attention_weights['syntactic_weights'].shape, (batch_size, 1, seq_len))
    
    def test_full_model(self):
        config = {
            'img_shape': (*self.image_size, 3),
            'max_code_length': self.max_code_length,
            'vocab_size': self.vocab_size,
            'embedding_dim': 256,
            'lstm_units': 512,
            'dropout_rate': 0.3
        }
        
        model = create_pix2code_model(config)
        output = model([self.test_image, self.test_code])
        
        self.assertEqual(output.shape, (self.batch_size, self.max_code_length, self.vocab_size))
    
    def test_loss_and_metrics(self):
        """Test loss functions and metrics."""
        # Test masked loss
        y_true = tf.constant([[1, 2, 0], [1, 0, 0]])  # 0 is padding
        y_pred = tf.random.normal((2, 3, self.vocab_size))
        loss = masked_loss(y_true, y_pred)
        self.assertIsInstance(loss, tf.Tensor)
        
        # Test BLEU-4 metric
        bleu_metric = BLEU4Metric()
        bleu_metric.update_state(y_true, y_pred)
        bleu_score = bleu_metric.result()
        self.assertIsInstance(bleu_score, tf.Tensor)
        
        # Test Exact Match metric
        exact_match = ExactMatchMetric()
        exact_match.update_state(y_true, y_pred)
        match_score = exact_match.result()
        self.assertIsInstance(match_score, tf.Tensor)

    def test_attention_edge_cases(self):
        """Test attention mechanism with edge cases."""
        # Test with empty sequence
        batch_size = 2
        units = 256
        empty_seq = tf.zeros((batch_size, 0, units))
        query = tf.random.normal((batch_size, units))
        
        dual_att = DualAttention(units=units)
        context, weights = dual_att(query, empty_seq, empty_seq)
        self.assertEqual(context.shape, (batch_size, units))
        
        # Test with very long sequence
        long_seq = tf.random.normal((batch_size, 1000, units))
        context, weights = dual_att(query, long_seq, long_seq)
        self.assertEqual(context.shape, (batch_size, units))
        
        # Test with single element sequence
        single_seq = tf.random.normal((batch_size, 1, units))
        context, weights = dual_att(query, single_seq, single_seq)
        self.assertEqual(context.shape, (batch_size, units))

    def test_model_edge_cases(self):
        """Test model with edge cases."""
        config = {
            'img_shape': (*self.image_size, 3),
            'max_code_length': self.max_code_length,
            'vocab_size': self.vocab_size,
            'embedding_dim': 256,
            'lstm_units': 512,
            'dropout_rate': 0.3
        }
        
        model = create_pix2code_model(config)
        
        # Test with zero image
        zero_img = tf.zeros((1, *self.image_size, 3))
        zero_code = tf.zeros((1, self.max_code_length), dtype=tf.int32)
        output = model([zero_img, zero_code])
        self.assertEqual(output.shape, (1, self.max_code_length, self.vocab_size))
        
        # Test with maximum values
        max_img = tf.ones((1, *self.image_size, 3)) * 255.0
        max_code = tf.ones((1, self.max_code_length), dtype=tf.int32) * (self.vocab_size - 1)
        output = model([max_img, max_code])
        self.assertEqual(output.shape, (1, self.max_code_length, self.vocab_size))

if __name__ == '__main__':
    unittest.main() 