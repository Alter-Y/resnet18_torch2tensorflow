import tensorflow as tf
import torch
import torch.nn as nn
from tensorflow import keras

class TFBN(keras.layers.Layer):
    # TensorFlow BatchNormalization wrapper
    def __init__(self, w=None, name=None):
        super().__init__()
        self.n = [f'{name}.bias',
                  f'{name}.weight',
                  f'{name}.running_mean',
                  f'{name}.running_var']
        self.bn = keras.layers.BatchNormalization(
            beta_initializer=keras.initializers.Constant(w[self.n[0]].numpy()),
            gamma_initializer=keras.initializers.Constant(w[self.n[1]].numpy()),
            moving_mean_initializer=keras.initializers.Constant(w[self.n[2]].numpy()),
            moving_variance_initializer=keras.initializers.Constant(w[self.n[3]].numpy()),
            epsilon=1e-5)

    def call(self, inputs):
        return self.bn(inputs)

# class TFfc(keras.layers.Layer):
#     def __self__(self, num_classes=None, w=None):
#         super().__init__()
#         self.fc = keras.layers.Dense(num_classes,
#                                      use_bias=True,
#                                      kernel_initializer=keras.initializers.Constant(w.weight.numpy()),
#                                      bias_initializer=keras.initializers.Constant(w.bias.numpy()))
#     def call(self, inputs):
#         return self.fc(inputs)

def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else (x // 2 for x in k)  # auto-pad
    return p

class TFPad(keras.layers.Layer):

    def __init__(self, pad):
        super().__init__()
        self.pad = tf.constant([[0, 0], [pad, pad], [pad, pad], [0, 0]])

    def call(self, inputs):
        return tf.pad(inputs, self.pad, mode='constant', constant_values=0)

class TFConv1(keras.layers.Layer):
    def __init__(self, in_channels, out_channels, ksize=7, stride=2, padding=None, w=None):
        super().__init__()
        conv = keras.layers.Conv2D(
            out_channels,
            ksize,
            strides=stride,
            padding='SAME' if stride == 1 else 'VALID',
            use_bias=False,
            kernel_initializer=keras.initializers.Constant(w['conv1.conv.weight'].permute(2, 3, 1, 0).numpy()),
            bias_initializer='zeros')
        self.conv = conv if stride == 1 else keras.Sequential([TFPad(autopad(ksize, padding)), conv])
        self.bn = TFBN(w, name='conv1.bn')
        self.act = lambda x: keras.activations.relu(x)
        maxpool = keras.layers.MaxPooling2D(3, strides=stride, padding='SAME' if stride == 1 else 'VALID')
        self.maxpool = maxpool if stride == 1 else keras.Sequential([TFPad(autopad(3, padding)), maxpool])

    def call(self, inputs):
        return self.maxpool(self.act(self.bn(self.conv(inputs))))

class TFDownsampling(keras.layers.Layer):
    def __init__(self, in_channels, out_channels, ksize=1, stride=2, padding=None, w=None, name=None):
        super().__init__()
        self.n = f'{name}.residual.conv.weight'
        conv = keras.layers.Conv2D(
            out_channels,
            ksize,
            strides=stride,
            padding='SAME' if stride == 1 else 'VALID',
            use_bias=False,
            kernel_initializer=keras.initializers.Constant(w[self.n].permute(2, 3, 1, 0).numpy()),
            bias_initializer='zeros')
        self.conv = conv if stride == 1 else keras.Sequential([TFPad(autopad(ksize, padding)), conv])
        self.bn = TFBN(w, name= f'{name}.residual.bn')

    def call(self, inputs):
        return self.bn(self.conv(inputs))

class TFBasicblock(keras.layers.Layer):
    def __init__(self, in_channels, out_channels, ksize=3, stride=1, downsampling=False, padding=None, w=None, name=None):
        super().__init__()
        self.n = [f'{name}.conv1.weight',
                  f'{name}.conv2.weight']
        conv1 = keras.layers.Conv2D(
            out_channels,
            ksize,
            strides = stride,
            padding = 'SAME' if stride == 1 else 'VALID',
            use_bias=False,
            kernel_initializer=keras.initializers.Constant(w[self.n[0]].permute(2, 3, 1, 0).numpy()),
            bias_initializer='zeros')
        self.conv1 = conv1 if stride == 1 else keras.Sequential([TFPad(autopad(ksize, padding)), conv1])
        self.bn1 = TFBN(w, name=f'{name}.bn1')
        self.act = lambda x: keras.activations.relu(x)
        conv2 = keras.layers.Conv2D(
            out_channels,
            ksize,
            strides = 1,
            padding = 'SAME' if stride == 1 else 'VALID',
            use_bias=False,
            kernel_initializer=keras.initializers.Constant(w[self.n[1]].permute(2, 3, 1, 0).numpy()),
            bias_initializer='zeros')
        self.conv2 = conv2 if stride == 1 else keras.Sequential([TFPad(autopad(ksize, padding)), conv2])
        self.bn2 = TFBN(w, name=f'{name}.bn2')
        self.residual = TFDownsampling(in_channels=in_channels,
                                       out_channels=out_channels,
                                       stride=stride,
                                       w=w,
                                       name=name) if downsampling else tf.identity

    def call(self, inputs):
        y = self.residual(inputs)
        x_1 = self.act(self.bn1(self.conv1(inputs)))
        x_2 = self.bn2(self.conv2(x_1))

        out = x_2 + y
        out = self.act(out)

        return out

def make_layer(in_channels, out_channels, block, stride, w, name):
    blocks = []
    blocks.append(TFBasicblock(in_channels,
                               out_channels,
                               stride=stride,
                               downsampling=True,
                               w=w,
                               name=name[0]))
    for _ in range(1, block):
        blocks.append(TFBasicblock(out_channels, out_channels, w=w, name=name[1]))

    return keras.Sequential(blocks)

def TFResnet(block=2, num_classes=10, w=None):
    layers = []
    layers.append(keras.Sequential(TFConv1(3, 64, w=w)))
    layers.append(make_layer(64, 64, block=block, stride=2, w=w, name=['layer1.0', 'layer1.1']))
    layers.append(make_layer(64, 128, block=block, stride=2, w=w, name=['layer2.0', 'layer2.1']))
    layers.append(make_layer(128, 256, block=block, stride=2, w=w, name=['layer3.0', 'layer3.1']))
    layers.append(make_layer(256, 512, block=block, stride=2, w=w, name=['layer4.0', 'layer4.1']))
    layers.append(keras.Sequential(keras.layers.GlobalAveragePooling2D()))
    # layers.append(keras.Sequential(keras.layers.Flatten()))
    layers.append(keras.Sequential(keras.layers.Dense(
        num_classes,
        use_bias=True,
        kernel_initializer=keras.initializers.Constant(w['fc.weight'].permute(1,0).numpy()),
        bias_initializer=keras.initializers.Constant(w['fc.bias'].numpy()))))

    return keras.Sequential(layers)

def predict(inputs, model):
    x = inputs
    for m in model.layers:
        x = m(x)

    return x



if __name__ == '__main__':

    inputs = tf.keras.Input(shape=(32,32,3))
    pt = '.\\model\\resnet18.pt'
    m = torch.load(pt, map_location=torch.device('cpu'))
    model = TFResnet(w=m)
    out = predict(inputs, model)
    keras_model = tf.keras.Model(inputs=inputs, outputs=out)
    keras_model.trainable = False
    keras_model.summary()

    keras_model.save('.\\model\\resnet18_save_model', save_format='tf')