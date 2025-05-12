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
        super().__init__(name=name, **kwargs)
        self.exact_matches = self.add_weight(name='exact_matches', initializer='zeros')
        self.total_samples = self.add_weight(name='total_samples', initializer='zeros')
    
    def update_state(self, y_true, y_pred):
        """Update the metric state."""
        # Convert both tensors to int32
        y_true = tf.cast(y_true, tf.int32)
        pred_tokens = tf.cast(tf.argmax(y_pred, axis=-1), tf.int32)
        
        matches = tf.cast(tf.reduce_all(tf.equal(y_true, pred_tokens), axis=1), tf.float32)
        self.exact_matches.assign_add(tf.reduce_sum(matches))
        self.total_samples.assign_add(tf.cast(tf.shape(y_true)[0], tf.float32))
    
    def result(self):
        """Return the metric result."""
        return self.exact_matches / self.total_samples
    
    def reset_state(self):
        """Reset the metric state."""
        self.exact_matches.assign(0.)
        self.total_samples.assign(0.) 