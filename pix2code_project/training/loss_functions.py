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