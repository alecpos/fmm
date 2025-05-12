import tensorflow as tf
from tensorflow.keras.applications import ResNet50
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