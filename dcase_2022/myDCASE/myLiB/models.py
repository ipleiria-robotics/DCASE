import tensorflow as tf


def modelSelector(model_type,input_Shape): #[Baseline,BaselineMod,BC_resnet,BC_resnet_DCASE2021,BC_resnet_DCASE2021_noSSN,BC_resnet_DCASE2021_resnorm,BC_resnet_DCASE2021_1resnorm_noSSN,BC_resnet_DCASE2021_resnorm_MOD8]
    if (model_type == 'Baseline'):
        model = Baseline(input_Shape)
    elif (model_type == 'BaselineMod'):
        model = BaselineMod(input_Shape)
    elif (model_type == 'BM1'):
        model = BM1(input_Shape)
    elif (model_type == 'BM2'):
        model = BM2(input_Shape)
    elif (model_type == 'inception_run2_test1'):
        model = inception_run2_test1(input_Shape)
    elif (model_type == 'IRMod'):
        model = IRMod(input_Shape)
    elif (model_type == 'BC_resnet'):
        model = BC_resnet(input_Shape)
    elif (model_type == 'BC_resnet_DCASE2021'):
        model = BC_resnet_DCASE2021(input_Shape)
    elif (model_type == 'BC_resnet_DCASE2021_MOD8'):
        model = BC_resnet_DCASE2021_MOD8(input_Shape)
    elif (model_type == 'MobileNetV3Small'):
        model = MobileNetV3Small(input_Shape)
    elif (model_type == 'BM2_resnet'):
        model = BM2_resnet(input_Shape)
    return model
def saveModelH5(model,filedir):
    model.save(filedir)
def loadModelH5(filedir):
    model=tf.keras.models.load_model(filedir)
    return model
########################################### CLASS ######################################################################
@tf.keras.utils.register_keras_serializable()
class TransitionBlock(tf.keras.layers.Layer):
  """TransitionBlock.

  It is based on paper:
    Broadcasted Residual Learning for Efficient Keyword Spotting
    https://arxiv.org/pdf/2106.04140.pdf

  Attributes:
    filters: number of filters/channels in conv layer
    dilation: dilation of conv layer
    stride: stride of conv layer
    padding: padding of conv layer (can be same or causal only)
    dropout: dropout rate
    use_one_step: this parameter will be used for streaming only
    sub_groups: number of groups for SubSpectralNormalization
    **kwargs: additional layer arguments
  """

  def __init__(self,
               filters=8,
               dilation=1,
               stride=1,
               padding='same',
               dropout=0.5,
               use_one_step=True,
               sub_groups=5,
               **kwargs):
    super(TransitionBlock, self).__init__(**kwargs)
    self.filters = filters
    self.dilation = dilation
    self.stride = stride
    self.padding = padding
    self.dropout = dropout
    self.use_one_step = use_one_step
    self.sub_groups = sub_groups

    self.frequency_dw_conv = tf.keras.layers.DepthwiseConv2D(
        kernel_size=(1, 3),
        strides=(1, 1),
        dilation_rate=self.dilation,
        padding='same',
        use_bias=False)
    if self.padding == 'same':
      self.temporal_dw_conv = tf.keras.layers.DepthwiseConv2D(
          kernel_size=(3, 1),
          strides=(1, 1),
          dilation_rate=self.dilation,
          padding='same',
          use_bias=False)
    self.batch_norm1 = tf.keras.layers.BatchNormalization()
    self.batch_norm2 = tf.keras.layers.BatchNormalization()
    self.conv1x1_1 = tf.keras.layers.Conv2D(
        filters=self.filters,
        kernel_size=1,
        strides=self.stride,
        padding='valid',
        use_bias=False)
    self.conv1x1_2 = tf.keras.layers.Conv2D(
        filters=self.filters,
        kernel_size=1,
        strides=1,
        padding='valid',
        use_bias=False)
    self.spatial_drop = tf.keras.layers.SpatialDropout2D(rate=self.dropout)
    self.spectral_norm = SubSpectralNormalization(
        self.sub_groups)

  def call(self, inputs):

    # expected input: [N, Time, Frequency, Channels]
    if inputs.shape.rank != 4:
      raise ValueError('input_shape.rank:%d must be 4' % inputs.shape.rank)

    net = inputs
    net = self.conv1x1_1(net)
    net = self.batch_norm1(net)
    net = tf.keras.activations.relu(net)
    net = self.frequency_dw_conv(net)
    net = self.spectral_norm(net)

    residual = net
    net = tf.keras.backend.mean(net, axis=2, keepdims=True)
    net = self.temporal_dw_conv(net)
    net = self.batch_norm2(net)
    net = tf.keras.activations.swish(net)
    net = self.conv1x1_2(net)
    net = self.spatial_drop(net)

    net = net + residual
    net = tf.keras.activations.relu(net)
    return net

  def get_config(self):
    config = {
        'filters': self.filters,
        'dilation': self.dilation,
        'stride': self.stride,
        'padding': self.padding,
        'dropout': self.dropout,
        'use_one_step': self.use_one_step,
        'sub_groups': self.sub_groups,
        }
    base_config = super(TransitionBlock, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  def get_input_state(self):
    return self.temporal_dw_conv.get_input_state()

  def get_output_state(self):
    return self.temporal_dw_conv.get_output_state()
@tf.keras.utils.register_keras_serializable()
class SubSpectralNormalization(tf.keras.layers.Layer):
  """Sub spectral normalization layer.

  It is based on paper:
  "SUBSPECTRAL NORMALIZATION FOR NEURAL AUDIO DATA PROCESSING"
  https://arxiv.org/pdf/2103.13620.pdf
  """

  def __init__(self, sub_groups, **kwargs):
    super(SubSpectralNormalization, self).__init__(**kwargs)
    self.sub_groups = sub_groups

    self.batch_norm = tf.keras.layers.BatchNormalization()

  def call(self, inputs):
    # expected input: [N, Time, Frequency, Channels]
    if inputs.shape.rank != 4:
      raise ValueError('input_shape.rank:%d must be 4' % inputs.shape.rank)

    input_shape = inputs.shape.as_list()
    if input_shape[2] % self.sub_groups:
      raise ValueError('input_shape[2]: %d must be divisible by '
                       'self.sub_groups %d ' %
                       (input_shape[2], self.sub_groups))

    net = inputs
    if self.sub_groups == 1:
      net = self.batch_norm(net)
    else:
      target_shape = [
          input_shape[1], input_shape[2] // self.sub_groups,
          input_shape[3] * self.sub_groups
      ]
      net = tf.keras.layers.Reshape(target_shape)(net)
      net = self.batch_norm(net)
      net = tf.keras.layers.Reshape(input_shape[1:])(net)
    return net

  def get_config(self):
    config = {'sub_groups': self.sub_groups}
    base_config = super(SubSpectralNormalization, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))
@tf.keras.utils.register_keras_serializable()
class NormalBlock(tf.keras.layers.Layer):
  """NormalBlock.

  It is based on paper:
    Broadcasted Residual Learning for Efficient Keyword Spotting
    https://arxiv.org/pdf/2106.04140.pdf

  Attributes:
    filters: number of filters/channels in conv layer
    dilation: dilation of conv layer
    stride: stride of conv layer
    padding: padding of conv layer (can be same or causal only)
    dropout: dropout rate
    use_one_step: this parameter will be used for streaming only
    sub_groups: number of groups for SubSpectralNormalization
    **kwargs: additional layer arguments
  """

  def __init__(
      self,
      filters,
      dilation=1,
      stride=1,
      padding='same',
      dropout=0.5,
      use_one_step=True,
      sub_groups=5,
      **kwargs):
    super(NormalBlock, self).__init__(**kwargs)
    self.filters = filters
    self.dilation = dilation
    self.stride = stride
    self.padding = padding
    self.dropout = dropout
    self.use_one_step = use_one_step
    self.sub_groups = sub_groups

    self.frequency_dw_conv = tf.keras.layers.DepthwiseConv2D(
        kernel_size=(1, 3),
        strides=self.stride,
        dilation_rate=self.dilation,
        padding=self.padding,
        use_bias=False)
    if self.padding == 'same':
      self.temporal_dw_conv = tf.keras.layers.DepthwiseConv2D(
          kernel_size=(3, 1),
          strides=self.stride,
          dilation_rate=self.dilation,
          padding='same',
          use_bias=False)
    self.batch_norm = tf.keras.layers.BatchNormalization()
    self.conv1x1 = tf.keras.layers.Conv2D(
        filters=self.filters,
        kernel_size=1,
        strides=1,
        padding=self.padding,
        use_bias=False)
    self.spatial_drop = tf.keras.layers.SpatialDropout2D(rate=self.dropout)
    self.spectral_norm = SubSpectralNormalization(
        self.sub_groups)

  def call(self, inputs):

    # expected input: [N, Time, Frequency, Channels]
    if inputs.shape.rank != 4:
      raise ValueError('input_shape.rank:%d must be 4' % inputs.shape.rank)

    identity = inputs
    net = inputs
    net = self.frequency_dw_conv(net)
    net = self.spectral_norm(net)

    residual = net
    net = tf.keras.backend.mean(net, axis=2, keepdims=True)
    net = self.temporal_dw_conv(net)
    net = self.batch_norm(net)
    net = tf.keras.activations.swish(net)
    net = self.conv1x1(net)
    net = self.spatial_drop(net)

    net = net + identity + residual
    net = tf.keras.activations.relu(net)
    return net

  def get_config(self):
    config = {
        'filters': self.filters,
        'dilation': self.dilation,
        'stride': self.stride,
        'padding': self.padding,
        'dropout': self.dropout,
        'use_one_step': self.use_one_step,
        'sub_groups': self.sub_groups,
        }
    base_config = super(NormalBlock, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  def get_input_state(self):
    return self.temporal_dw_conv.get_input_state()

  def get_output_state(self):
    return self.temporal_dw_conv.get_output_state()
####################################### MODELOS OBTIDOS KERAS TUNER ####################################################
def BM2(input_Shape):
    filterL1 = 20
    filterL2 = 28
    filterL3 = 28
    dropout1 = 0.1
    dropout2 = 0.3
    dropout3 = 0.4
    CNN_kernel_size_1 = 7
    CNN_kernel_size_11 = 5
    CNN_kernel_size_2 = 7
    CNN_kernel_size_22 = 3
    CNN_kernel_size_3 = 3
    CNN_kernel_size_33 = 7
    pool_kernel_size_2 = 1
    pool_kernel_size_22 = 2
    pool_kernel_size_3 = 2
    pool_kernel_size_33 = 2
    pooling_1 = 'max'
    pooling_2 = 'max'
    pooling_3 = 'avg'
    units = 256

    input = tf.keras.layers.Input(shape=input_Shape)
    # CNN layer #1:
    x = tf.keras.layers.Conv2D(filters=filterL1, kernel_size=(CNN_kernel_size_1, CNN_kernel_size_11), padding='same',
                               name="Conv-1")(input)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)

    # CNN layer #2:
    x = tf.keras.layers.Conv2D(filters=filterL2, kernel_size=(CNN_kernel_size_2, CNN_kernel_size_22), activation='relu',
                               padding='same', name="Conv-2")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    if pooling_1 == 'max':
        x = tf.keras.layers.MaxPooling2D(pool_size=(pool_kernel_size_2, pool_kernel_size_22))(x)
    else:
        x = tf.keras.layers.AveragePooling2D(pool_size=(pool_kernel_size_2, pool_kernel_size_22))(x)
    x = tf.keras.layers.Dropout(dropout1)(x)

    # CNN layer #3:
    x = tf.keras.layers.Conv2D(filters=filterL3, kernel_size=(CNN_kernel_size_3, CNN_kernel_size_33), activation='relu',
                               padding='same', name="Conv-3")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    if pooling_2 == 'max':
        x = tf.keras.layers.MaxPooling2D(pool_size=(pool_kernel_size_3, pool_kernel_size_33))(x)
    else:
        x = tf.keras.layers.AveragePooling2D(pool_size=(pool_kernel_size_3, pool_kernel_size_33))(x)
    x = tf.keras.layers.Dropout(dropout2)(x)
    
    
 # CNN layer #4:
    x = tf.keras.layers.Conv2D(filters=16, kernel_size=(7, 5), activation='relu', padding='same', name="Conv-3_0")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Dropout(0.3)(x)

    if pooling_3 == 'max':
        x = tf.keras.layers.GlobalMaxPool2D()(x)
    elif pooling_3 == 'avg':
        x = tf.keras.layers.GlobalAvgPool2D()(x)
    elif pooling_3 == 'clear':
        pass

    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(units, activation='relu', name="lastLayer")(x)
    x = tf.keras.layers.Dropout(dropout3)(x)
    output = tf.keras.layers.Dense(10, activation='softmax')(x)
    model = tf.keras.Model(input, output)

    # hp_optimizer = hp.Choice('optimizer', values=['adam', 'SGD', 'rmsprop'])
    # optimizer = tf.keras.optimizers.get(hp_optimizer)
    optimizer = tf.keras.optimizers.get('adam')
    # optimizer.learning_rate = hp.Choice("learning_rate", [0.1, 0.01, 0.001,0.0001], default=0.01)
    optimizer.learning_rate = 0.001
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    return model
def BM1(input_Shape):
    filterL1 = 20
    filterL2 = 16
    filterL3 = 16
    dropout1 = 0.2
    dropout2 = 0.3
    dropout3 = 0.3
    CNN_kernel_size_1 = 7
    CNN_kernel_size_11 = 5
    CNN_kernel_size_2 = 7
    CNN_kernel_size_22 = 3
    CNN_kernel_size_3 = 7
    CNN_kernel_size_33 = 3
    pool_kernel_size_2 = 2
    pool_kernel_size_22 = 1
    pool_kernel_size_3 = 2
    pool_kernel_size_33 = 2
    pooling_1 = 'max'
    pooling_2 = 'avg'
    pooling_3 = 'avg'
    units = 160

    input = tf.keras.layers.Input(shape=input_Shape)
    # CNN layer #1:
    x = tf.keras.layers.Conv2D(filters=filterL1, kernel_size=(CNN_kernel_size_1, CNN_kernel_size_11), padding='same',
                               name="Conv-1")(input)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)

    # CNN layer #2:
    x = tf.keras.layers.Conv2D(filters=filterL2, kernel_size=(CNN_kernel_size_2, CNN_kernel_size_22), activation='relu',
                               padding='same', name="Conv-2")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    if pooling_1 == 'max':
        x = tf.keras.layers.MaxPooling2D(pool_size=(pool_kernel_size_2, pool_kernel_size_22))(x)
    else:
        x = tf.keras.layers.AveragePooling2D(pool_size=(pool_kernel_size_2, pool_kernel_size_22))(x)
    x = tf.keras.layers.Dropout(dropout1)(x)

    # CNN layer #3:
    x = tf.keras.layers.Conv2D(filters=filterL3, kernel_size=(CNN_kernel_size_3, CNN_kernel_size_33), activation='relu',
                               padding='same', name="Conv-3")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    if pooling_2 == 'max':
        x = tf.keras.layers.MaxPooling2D(pool_size=(pool_kernel_size_3, pool_kernel_size_33))(x)
    else:
        x = tf.keras.layers.AveragePooling2D(pool_size=(pool_kernel_size_3, pool_kernel_size_33))(x)
    x = tf.keras.layers.Dropout(dropout2)(x)

    x = tf.keras.layers.Conv2D(filters=16, kernel_size=(5, 5), activation='relu', padding='same', name="Conv-3_0")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Dropout(0.4)(x)

    x = tf.keras.layers.Conv2D(filters=12, kernel_size=(5, 3), activation='relu', padding='same', name="Conv-3_1")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Dropout(0.2)(x)

    if pooling_3 == 'max':
        x = tf.keras.layers.GlobalMaxPool2D()(x)
    elif pooling_3 == 'avg':
        x = tf.keras.layers.GlobalAvgPool2D()(x)
    elif pooling_3 == 'clear':
        pass

    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(units, activation='relu', name="lastLayer")(x)
    x = tf.keras.layers.Dropout(dropout3)(x)
    output = tf.keras.layers.Dense(10, activation='softmax')(x)
    model = tf.keras.Model(input, output)
    return model


def BM2_resnet(input_Shape):
    filterL1 = 20
    filterL2 = 28
    filterL3 = 28
    dropout1 = 0.1
    dropout2 = 0.3
    dropout3 = 0.4
    CNN_kernel_size_1 = 7
    CNN_kernel_size_11 = 5
    CNN_kernel_size_2 = 7
    CNN_kernel_size_22 = 3
    CNN_kernel_size_3 = 3
    CNN_kernel_size_33 = 7
    pool_kernel_size_2 = 1
    pool_kernel_size_22 = 2
    pool_kernel_size_3 = 2
    pool_kernel_size_33 = 2
    pooling_1 = 'max'
    pooling_2 = 'max'
    pooling_3 = 'avg'
    units = 256

    input = tf.keras.layers.Input(shape=input_Shape)
    # CNN layer #1:
    x = tf.keras.layers.Conv2D(filters=filterL1, kernel_size=(CNN_kernel_size_1, CNN_kernel_size_11), padding='same',
                               name="Conv-1")(input)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)



    # CNN layer #2:
    x = tf.keras.layers.Conv2D(filters=filterL2, kernel_size=(CNN_kernel_size_2, CNN_kernel_size_22), activation='relu',
                               padding='same', name="Conv-2")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    if pooling_1 == 'max':
        x = tf.keras.layers.MaxPooling2D(pool_size=(pool_kernel_size_2, pool_kernel_size_22))(x)
    else:
        x = tf.keras.layers.AveragePooling2D(pool_size=(pool_kernel_size_2, pool_kernel_size_22))(x)
    x = tf.keras.layers.Dropout(dropout1)(x)

    # CNN layer #3:
    x = tf.keras.layers.Conv2D(filters=filterL3, kernel_size=(CNN_kernel_size_3, CNN_kernel_size_33), activation='relu',
                               padding='same', name="Conv-3")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    if pooling_2 == 'max':
        x = tf.keras.layers.MaxPooling2D(pool_size=(pool_kernel_size_3, pool_kernel_size_33))(x)
    else:
        x = tf.keras.layers.AveragePooling2D(pool_size=(pool_kernel_size_3, pool_kernel_size_33))(x)
    x = tf.keras.layers.Dropout(dropout2)(x)

    # Save the input value
    X_shortcut = x

    # CNN layer #4:
    x = tf.keras.layers.Conv2D(filters=16, kernel_size=(7, 5), activation='relu', padding='same', name="Conv-3_0")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Dropout(0.3)(x)

    ##### SHORTCUT PATH #### tem de ter as mesma dimensoes
    X_shortcut = tf.keras.layers.Conv2D(filters=16, kernel_size=(7, 5), padding='same')(X_shortcut)
    x = tf.keras.layers.BatchNormalization()(x)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation (≈2 lines)
    x = tf.keras.layers.Add()([x, X_shortcut])
    x = tf.keras.layers.Activation('relu')(x)

    if pooling_3 == 'max':
        x = tf.keras.layers.GlobalMaxPool2D()(x)
    elif pooling_3 == 'avg':
        x = tf.keras.layers.GlobalAvgPool2D()(x)
    elif pooling_3 == 'clear':
        pass

    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(units, activation='relu', name="lastLayer")(x)
    x = tf.keras.layers.Dropout(dropout3)(x)
    output = tf.keras.layers.Dense(10, activation='softmax')(x)
    model = tf.keras.Model(input, output)

    # hp_optimizer = hp.Choice('optimizer', values=['adam', 'SGD', 'rmsprop'])
    # optimizer = tf.keras.optimizers.get(hp_optimizer)
    optimizer = tf.keras.optimizers.get('adam')
    # optimizer.learning_rate = hp.Choice("learning_rate", [0.1, 0.01, 0.001,0.0001], default=0.01)
    optimizer.learning_rate = 0.001
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    return model

def IRMod(input_Shape):
    tf.keras.backend.clear_session()
    def conv2d_bn(x, filters, kernel_size, strides=1, padding='same', activation='relu', use_bias=False, name=None):
        x = tf.keras.layers.Conv2D(filters, kernel_size, strides=strides, padding=padding, use_bias=use_bias,name=name)(x)
        if not use_bias:
            bn_name = None if name is None else name + '_bn'
            bn_axis = 3  # channellast
            x = tf.keras.layers.BatchNormalization(axis=bn_axis, scale=False, name=bn_name)(x)
        if activation is not None:
            ac_name = None if name is None else name + '_ac'
            x = tf.keras.layers.Activation(activation, name=ac_name)(x)
        return x

    def inception_resnet_block(x, scale, block_idx, activation='relu'):

        f1=12
        f3=20
        f6=24
        convKernel1=3
        convKernel11=1

        branch_0 = conv2d_bn(x, f1, 1)
        branch_1 = conv2d_bn(x, (f3/2), 1)
        branch_1 = conv2d_bn(branch_1, f3, (convKernel1,convKernel11)) #(3,3)
        branch_2 = conv2d_bn(x, (f6/2), 1)
        branch_2 = conv2d_bn(branch_2, f6, (convKernel1,convKernel11)) #(3,3)
        branch_2 = conv2d_bn(branch_2, f6, (convKernel1,convKernel11)) #(3,3)
        branches = [branch_0, branch_1, branch_2]

        block_type="blockA"
        block_name = block_type + '_' + str(block_idx)
        channel_axis = 3
        mixed = tf.keras.layers.Concatenate(axis=channel_axis, name=block_name + '_mixed')(branches)
        up = conv2d_bn(mixed,tf.keras.backend.int_shape(x)[channel_axis],1,activation=None,use_bias=True,name=block_name + '_conv')
        x = tf.keras.layers.Lambda(lambda inputs, scale: inputs[0] + inputs[1] * scale,output_shape=tf.keras.backend.int_shape(x)[1:],arguments={'scale': scale},name=block_name)([x, up])
        if activation is not None:
            x = tf.keras.layers.Activation(activation, name=block_name + '_ac')(x)
        return x

    channel_axis=3
    input = tf.keras.layers.Input(shape=input_Shape) # [batch,freq,time,1]

    Convf1 = 40
    Convf3 = 24
    convKernel_1 = 3
    convKernel_11 = 1
    pool_1 = 2
    pool_11 = 1
    x = conv2d_bn(input,Convf1, (convKernel_1,convKernel_11)) #padding='valid', removido strides=2
    # x = conv2d_bn(x, 16, 1, padding='valid')
    # x = conv2d_bn(x, 32, (3,3), padding='valid')
    # x = tf.keras.layers.MaxPooling2D((3,3),strides=(2,1))(x) #remove strides=2
    x = conv2d_bn(x,(Convf3/2), 1, padding='valid') #valid
    x = conv2d_bn(x, Convf3, (convKernel_1,convKernel_11), padding='valid')
    x = tf.keras.layers.MaxPooling2D((pool_1,pool_11))(x) #remove strides=2


    Convf4 = 12
    Convf6 = 16
    Convf9 = 8
    Convf10 = 16
    convKernel_3 = 3
    convKernel_33 = 1
    pool_2 = 2
    pool_22 = 2

    # Mixed 5b (Inception-A block): _ x _ x 320
    branch_0 = conv2d_bn(x, Convf4, 1)
    branch_1 = conv2d_bn(x, (Convf6/2), 1)
    branch_1 = conv2d_bn(branch_1, Convf6, (convKernel_3,convKernel_33))
    branch_2 = conv2d_bn(x, (Convf9/2), 1)
    branch_2 = conv2d_bn(branch_2, Convf9, (convKernel_3,convKernel_33))
    branch_2 = conv2d_bn(branch_2, Convf9, (convKernel_3,convKernel_33))
    branch_pool = tf.keras.layers.AveragePooling2D((pool_2,pool_22), strides=1, padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, Convf10, 1)
    branches = [branch_0, branch_1, branch_2, branch_pool]
    x = tf.keras.layers.Concatenate(axis=channel_axis, name='mixed_5b')(branches)

    # 10x block35 (Inception-ResNet-A block): _ x _ x 320
    # for block_idx in range(1, 2):
    #     x = inception_resnet_block(x, scale=0.17, block_type='blockA', block_idx=block_idx)
    # for i in range(2):
    x = inception_resnet_block(x,0.3,block_idx=f'{0}')
    x = inception_resnet_block(x,0.3,block_idx=f'{1}')



    Convf11 = 16
    Convf13 = 16
    convKernel_4 = 1
    convKernel_44 =3
    # pool_3 = hp.Choice(f'pool_3', [1, 2])
    # pool_33= hp.Choice(f'pool_33', [1, 2])
    # Mixed 6a (Reduction-A block): 17 x 17 x 1088
    branch_0 = conv2d_bn(x, Convf11, (convKernel_4,convKernel_44),strides=(2,1), padding='valid')#strides=2
    branch_1 = conv2d_bn(x, (Convf13/2), 1)
    branch_1 = conv2d_bn(branch_1, Convf13, (convKernel_4,convKernel_44), strides=(2,1), padding='valid')
    branch_pool = tf.keras.layers.MaxPooling2D((convKernel_4,convKernel_44), strides=(2,1), padding='valid')(x)
    branches = [branch_0, branch_1, branch_pool]
    x = tf.keras.layers.Concatenate(axis=channel_axis, name='mixed_6a')(branches)     #18/9/384

    x = inception_resnet_block(x, scale=1., activation=None,block_idx=10)

    Convf14 = 116
    # Final convolution block:
    x = conv2d_bn(x, Convf14, 1, name='conv_7b')
    # x = tf.keras.layers.Dropout(0.3)(x)

    # Classification block
    x = tf.keras.layers.GlobalAveragePooling2D(name='avg_pool')(x)
    output = tf.keras.layers.Dense(10, activation='softmax', name='predictions')(x)
    model = tf.keras.Model(input, output)
    return model
def inception_run2_test1(input_Shape):
    tf.keras.backend.clear_session()
    def conv2d_bn(x, filters, kernel_size, strides=1, padding='same', activation='relu', use_bias=False, name=None):
        x = tf.keras.layers.Conv2D(filters, kernel_size, strides=strides, padding=padding, use_bias=use_bias,name=name)(x)
        if not use_bias:
            bn_name = None if name is None else name + '_bn'
            bn_axis = 3  # channellast
            x = tf.keras.layers.BatchNormalization(axis=bn_axis, scale=False, name=bn_name)(x)
        if activation is not None:
            ac_name = None if name is None else name + '_ac'
            x = tf.keras.layers.Activation(activation, name=ac_name)(x)
        return x

    def inception_resnet_block(x, scale, block_idx, activation='relu'):

        f1=16
        f3=12
        f6=8
        convKernel1=3
        convKernel11=3

        branch_0 = conv2d_bn(x, f1, 1)
        branch_1 = conv2d_bn(x, (f3/2), 1)
        branch_1 = conv2d_bn(branch_1, f3, (convKernel1,convKernel11)) #(3,3)
        branch_2 = conv2d_bn(x, (f6/2), 1)
        branch_2 = conv2d_bn(branch_2, f6, (convKernel1,convKernel11)) #(3,3)
        branch_2 = conv2d_bn(branch_2, f6, (convKernel1,convKernel11)) #(3,3)
        branches = [branch_0, branch_1, branch_2]

        block_type="blockA"
        block_name = block_type + '_' + str(block_idx)
        channel_axis = 3
        mixed = tf.keras.layers.Concatenate(axis=channel_axis, name=block_name + '_mixed')(branches)
        up = conv2d_bn(mixed,tf.keras.backend.int_shape(x)[channel_axis],1,activation=None,use_bias=True,name=block_name + '_conv')
        x = tf.keras.layers.Lambda(lambda inputs, scale: inputs[0] + inputs[1] * scale,output_shape=tf.keras.backend.int_shape(x)[1:],arguments={'scale': scale},name=block_name)([x, up])
        if activation is not None:
            x = tf.keras.layers.Activation(activation, name=block_name + '_ac')(x)
        return x

    channel_axis=3
    input = tf.keras.layers.Input(shape=input_Shape) # [batch,freq,time,1]

    Convf1 = 60
    Convf3 = 16
    convKernel_1 = 3
    convKernel_11 = 3
    pool_1 = 2
    pool_11 = 1
    x = conv2d_bn(input,Convf1, (convKernel_1,convKernel_11)) #padding='valid', removido strides=2
    # x = conv2d_bn(x, 16, 1, padding='valid')
    # x = conv2d_bn(x, 32, (3,3), padding='valid')
    # x = tf.keras.layers.MaxPooling2D((3,3),strides=(2,1))(x) #remove strides=2
    x = conv2d_bn(x,(Convf3/2), 1, padding='valid') #valid
    x = conv2d_bn(x, Convf3, (convKernel_1,convKernel_11), padding='valid')
    x = tf.keras.layers.MaxPooling2D((pool_1,pool_11))(x) #remove strides=2


    Convf4 = 16
    Convf6 = 4
    Convf9 = 20
    Convf10 = 16
    convKernel_3 = 3
    convKernel_33 = 1
    pool_2 = 2
    pool_22 = 2

    # Mixed 5b (Inception-A block): _ x _ x 320
    branch_0 = conv2d_bn(x, Convf4, 1)
    branch_1 = conv2d_bn(x, (Convf6/2), 1)
    branch_1 = conv2d_bn(branch_1, Convf6, (convKernel_3,convKernel_33))
    branch_2 = conv2d_bn(x, (Convf9/2), 1)
    branch_2 = conv2d_bn(branch_2, Convf9, (convKernel_3,convKernel_33))
    branch_2 = conv2d_bn(branch_2, Convf9, (convKernel_3,convKernel_33))
    branch_pool = tf.keras.layers.AveragePooling2D((pool_2,pool_22), strides=1, padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, Convf10, 1)
    branches = [branch_0, branch_1, branch_2, branch_pool]
    x = tf.keras.layers.Concatenate(axis=channel_axis, name='mixed_5b')(branches)

    # 10x block35 (Inception-ResNet-A block): _ x _ x 320
    # for block_idx in range(1, 2):
    #     x = inception_resnet_block(x, scale=0.17, block_type='blockA', block_idx=block_idx)
    # for i in range(2):
    x = inception_resnet_block(x,0.17,block_idx=f'{0}')

    Convf11 = 12
    Convf13 = 13
    convKernel_4 = 3
    convKernel_44 =1
    # pool_3 = hp.Choice(f'pool_3', [1, 2])
    # pool_33= hp.Choice(f'pool_33', [1, 2])
    # Mixed 6a (Reduction-A block): 17 x 17 x 1088
    branch_0 = conv2d_bn(x, Convf11, (convKernel_4,convKernel_44),strides=(2,1), padding='valid')#strides=2
    branch_1 = conv2d_bn(x, (Convf13/2), 1)
    branch_1 = conv2d_bn(branch_1, Convf13, (convKernel_4,convKernel_44), strides=(2,1), padding='valid')
    branch_pool = tf.keras.layers.MaxPooling2D((convKernel_4,convKernel_44), strides=(2,1), padding='valid')(x)
    branches = [branch_0, branch_1, branch_pool]
    x = tf.keras.layers.Concatenate(axis=channel_axis, name='mixed_6a')(branches)     #18/9/384

    x = inception_resnet_block(x, scale=1., activation=None,block_idx=10)

    Convf14 = 144
    # Final convolution block:
    x = conv2d_bn(x, Convf14, 1, name='conv_7b')
    # x = tf.keras.layers.Dropout(0.3)(x)

    # Classification block
    x = tf.keras.layers.GlobalAveragePooling2D(name='avg_pool')(x)
    output = tf.keras.layers.Dense(10, activation='softmax', name='predictions')(x)
    model = tf.keras.Model(input, output)
    return model

##################################### Baseline  ########################################################################
#Modelos implementados
def Baseline(input_Shape):
    # input = tf.keras.Input(input_Shape)
    input = tf.keras.layers.Input(shape=input_Shape)
    # CNN layer #1:
    x = tf.keras.layers.Conv2D(filters=16, kernel_size=(7, 7), padding='same', name="Conv-1")(input)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)

    # CNN layer #2:
    x = tf.keras.layers.Conv2D(filters=16, kernel_size=(7, 7), activation='relu', padding='same', name="Conv-2")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(5, 5))(x)
    x = tf.keras.layers.Dropout(0.3)(x)

    # CNN layer #3:
    x = tf.keras.layers.Conv2D(filters=32, kernel_size=(7, 7), activation='relu', padding='same', name="Conv-3")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(4, 10))(x)
    x = tf.keras.layers.Dropout(0.3)(x)

    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(100, activation='relu', name="lastLayer")(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    output = tf.keras.layers.Dense(10, activation='softmax')(x)
    model = tf.keras.Model(input, output)
    return model
def BaselineMod(input_Shape):
    # = Baseline Max pool (2,2)
    input = tf.keras.layers.Input(shape=input_Shape)
    # CNN layer #1:
    x = tf.keras.layers.Conv2D(filters=16, kernel_size=(7, 7), padding='same', name="Conv-1")(input)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)

    #CNN layer #2:
    x = tf.keras.layers.Conv2D(filters=16, kernel_size=(7, 7), activation='relu', padding='same',name="Conv-2")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = tf.keras.layers.Dropout(0.3)(x)

    # CNN layer #3:
    x = tf.keras.layers.Conv2D(filters=32, kernel_size=(7, 7), activation='relu', padding='same', name="Conv-3")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = tf.keras.layers.Dropout(0.3)(x)

    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(100, activation='relu', name="lastLayer")(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    output = tf.keras.layers.Dense(10, activation='softmax')(x)
    model = tf.keras.Model(input, output)
    return model
#minibaseline emsemble output5
def miniBaseline(input_Shape):
    # input = tf.keras.Input(input_Shape)
    input = tf.keras.layers.Input(shape=input_Shape)
    # CNN layer #1:
    x = tf.keras.layers.Conv2D(filters=9, kernel_size=(7, 7), padding='same', name="Conv-1")(input)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)

    # CNN layer #2:
    x = tf.keras.layers.Conv2D(filters=9, kernel_size=(7, 7), activation='relu', padding='same', name="Conv-2")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(5, 5))(x)
    x = tf.keras.layers.Dropout(0.5)(x)

    # CNN layer #3:
    x = tf.keras.layers.Conv2D(filters=32, kernel_size=(7, 7), activation='relu', padding='same', name="Conv-3")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(4, 10))(x)
    x = tf.keras.layers.Dropout(0.5)(x)

    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(100, activation='relu', name="lastLayer")(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    output = tf.keras.layers.Dense(5, activation='softmax')(x)
    model = tf.keras.Model(input, output)
    return model
#minibaseline2 emsemble output6
def miniBaseline2(input_Shape):
    # input = tf.keras.Input(input_Shape)
    input = tf.keras.layers.Input(shape=input_Shape)
    # CNN layer #1:
    x = tf.keras.layers.Conv2D(filters=9, kernel_size=(7, 7), padding='same', name="Conv-1")(input)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)

    # CNN layer #2:
    x = tf.keras.layers.Conv2D(filters=9, kernel_size=(7, 7), activation='relu', padding='same', name="Conv-2")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(5, 5))(x)
    x = tf.keras.layers.Dropout(0.5)(x)

    # CNN layer #3:
    x = tf.keras.layers.Conv2D(filters=32, kernel_size=(7, 7), activation='relu', padding='same', name="Conv-3")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(4, 10))(x)
    x = tf.keras.layers.Dropout(0.5)(x)

    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(100, activation='relu', name="lastLayer")(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    output = tf.keras.layers.Dense(6, activation='softmax')(x)
    model = tf.keras.Model(input, output)
    return model  #
#minibaselineKD emsembleKD output2
def miniBaseline_emKD(input_Shape):
    # input = tf.keras.Input(input_Shape)
    input = tf.keras.layers.Input(shape=input_Shape)
    # CNN layer #1:
    x = tf.keras.layers.Conv2D(filters=16, kernel_size=(7, 7), padding='same', name="Conv-1")(input)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)

    # CNN layer #2:
    x = tf.keras.layers.Conv2D(filters=16, kernel_size=(7, 7), activation='relu', padding='same', name="Conv-2")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(5, 5))(x)
    x = tf.keras.layers.Dropout(0.5)(x)

    # CNN layer #3:
    x = tf.keras.layers.Conv2D(filters=32, kernel_size=(7, 7), activation='relu', padding='same', name="Conv-3")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(4, 10))(x)
    x = tf.keras.layers.Dropout(0.5)(x)

    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(100, activation='relu', name="lastLayer")(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    output = tf.keras.layers.Dense(2, activation='softmax')(x)
    model = tf.keras.Model(input, output)
    return model
########################################################################################################################

################################################ Outros ###########################################################
#Rede Base de keyword spotting
def BM2_emKD(input_Shape):
    filterL1 = 20
    filterL2 = 28
    filterL3 = 28
    dropout1 = 0.1
    dropout2 = 0.3
    dropout3 = 0.4
    CNN_kernel_size_1 = 7
    CNN_kernel_size_11 = 5
    CNN_kernel_size_2 = 7
    CNN_kernel_size_22 = 3
    CNN_kernel_size_3 = 3
    CNN_kernel_size_33 = 7
    pool_kernel_size_2 = 1
    pool_kernel_size_22 = 2
    pool_kernel_size_3 = 2
    pool_kernel_size_33 = 2
    pooling_1 = 'max'
    pooling_2 = 'max'
    pooling_3 = 'avg'
    units = 256

    input = tf.keras.layers.Input(shape=input_Shape)
    # CNN layer #1:
    x = tf.keras.layers.Conv2D(filters=filterL1, kernel_size=(CNN_kernel_size_1, CNN_kernel_size_11), padding='same',
                               name="Conv-1")(input)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)

    # CNN layer #2:
    x = tf.keras.layers.Conv2D(filters=filterL2, kernel_size=(CNN_kernel_size_2, CNN_kernel_size_22), activation='relu',
                               padding='same', name="Conv-2")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    if pooling_1 == 'max':
        x = tf.keras.layers.MaxPooling2D(pool_size=(pool_kernel_size_2, pool_kernel_size_22))(x)
    else:
        x = tf.keras.layers.AveragePooling2D(pool_size=(pool_kernel_size_2, pool_kernel_size_22))(x)
    x = tf.keras.layers.Dropout(dropout1)(x)

    # CNN layer #3:
    x = tf.keras.layers.Conv2D(filters=filterL3, kernel_size=(CNN_kernel_size_3, CNN_kernel_size_33), activation='relu',
                               padding='same', name="Conv-3")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    if pooling_2 == 'max':
        x = tf.keras.layers.MaxPooling2D(pool_size=(pool_kernel_size_3, pool_kernel_size_33))(x)
    else:
        x = tf.keras.layers.AveragePooling2D(pool_size=(pool_kernel_size_3, pool_kernel_size_33))(x)
    x = tf.keras.layers.Dropout(dropout2)(x)

    x = tf.keras.layers.Conv2D(filters=16, kernel_size=(7, 5), activation='relu', padding='same', name="Conv-3_0")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Dropout(0.3)(x)

    if pooling_3 == 'max':
        x = tf.keras.layers.GlobalMaxPool2D()(x)
    elif pooling_3 == 'avg':
        x = tf.keras.layers.GlobalAvgPool2D()(x)
    elif pooling_3 == 'clear':
        pass

    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(units, activation='relu', name="lastLayer")(x)
    x = tf.keras.layers.Dropout(dropout3)(x)
    output = tf.keras.layers.Dense(2, activation='softmax')(x)
    model = tf.keras.Model(input, output)
    return model
#mais 2 filtros em cada class para tentar tranferir mais informaçao do teacher para o student
def BM2_big(input_Shape):
    filterL1 = 22
    filterL2 = 30
    filterL3 = 30
    dropout1 = 0.1
    dropout2 = 0.3
    dropout3 = 0.4
    CNN_kernel_size_1 = 7
    CNN_kernel_size_11 = 5
    CNN_kernel_size_2 = 7
    CNN_kernel_size_22 = 3
    CNN_kernel_size_3 = 3
    CNN_kernel_size_33 = 7
    pool_kernel_size_2 = 1
    pool_kernel_size_22 = 2
    pool_kernel_size_3 = 2
    pool_kernel_size_33 = 2
    pooling_1 = 'max'
    pooling_2 = 'max'
    pooling_3 = 'avg'
    units = 256

    input = tf.keras.layers.Input(shape=input_Shape)
    # CNN layer #1:
    x = tf.keras.layers.Conv2D(filters=filterL1, kernel_size=(CNN_kernel_size_1, CNN_kernel_size_11), padding='same',
                               name="Conv-1")(input)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)

    # CNN layer #2:
    x = tf.keras.layers.Conv2D(filters=filterL2, kernel_size=(CNN_kernel_size_2, CNN_kernel_size_22), activation='relu',
                               padding='same', name="Conv-2")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    if pooling_1 == 'max':
        x = tf.keras.layers.MaxPooling2D(pool_size=(pool_kernel_size_2, pool_kernel_size_22))(x)
    else:
        x = tf.keras.layers.AveragePooling2D(pool_size=(pool_kernel_size_2, pool_kernel_size_22))(x)
    x = tf.keras.layers.Dropout(dropout1)(x)

    # CNN layer #3:
    x = tf.keras.layers.Conv2D(filters=filterL3, kernel_size=(CNN_kernel_size_3, CNN_kernel_size_33), activation='relu',
                               padding='same', name="Conv-3")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    if pooling_2 == 'max':
        x = tf.keras.layers.MaxPooling2D(pool_size=(pool_kernel_size_3, pool_kernel_size_33))(x)
    else:
        x = tf.keras.layers.AveragePooling2D(pool_size=(pool_kernel_size_3, pool_kernel_size_33))(x)
    x = tf.keras.layers.Dropout(dropout2)(x)

    x = tf.keras.layers.Conv2D(filters=18, kernel_size=(7, 5), activation='relu', padding='same', name="Conv-3_0")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Dropout(0.3)(x)

    if pooling_3 == 'max':
        x = tf.keras.layers.GlobalMaxPool2D()(x)
    elif pooling_3 == 'avg':
        x = tf.keras.layers.GlobalAvgPool2D()(x)
    elif pooling_3 == 'clear':
        pass

    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(units, activation='relu', name="lastLayer")(x)
    x = tf.keras.layers.Dropout(dropout3)(x)
    output = tf.keras.layers.Dense(10, activation='softmax')(x)
    model = tf.keras.Model(input, output)
    return model
def miniBM2(input_Shape):
    filterL1 = 10
    filterL2 = 14
    filterL3 = 28
    dropout1 = 0.1
    dropout2 = 0.3
    dropout3 = 0.4
    CNN_kernel_size_1 = 7
    CNN_kernel_size_11 = 5
    CNN_kernel_size_2 = 7
    CNN_kernel_size_22 = 3
    CNN_kernel_size_3 = 3
    CNN_kernel_size_33 = 7
    pool_kernel_size_2 = 1
    pool_kernel_size_22 = 2
    pool_kernel_size_3 = 2
    pool_kernel_size_33 = 2
    pooling_1 = 'max'
    pooling_2 = 'max'
    pooling_3 = 'avg'
    units = 256

    input = tf.keras.layers.Input(shape=input_Shape)
    # CNN layer #1:
    x = tf.keras.layers.Conv2D(filters=filterL1, kernel_size=(CNN_kernel_size_1, CNN_kernel_size_11), padding='same',
                               name="Conv-1")(input)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)

    # CNN layer #2:
    x = tf.keras.layers.Conv2D(filters=filterL2, kernel_size=(CNN_kernel_size_2, CNN_kernel_size_22), activation='relu',
                               padding='same', name="Conv-2")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    if pooling_1 == 'max':
        x = tf.keras.layers.MaxPooling2D(pool_size=(pool_kernel_size_2, pool_kernel_size_22))(x)
    else:
        x = tf.keras.layers.AveragePooling2D(pool_size=(pool_kernel_size_2, pool_kernel_size_22))(x)
    x = tf.keras.layers.Dropout(dropout1)(x)

    # CNN layer #3:
    x = tf.keras.layers.Conv2D(filters=filterL3, kernel_size=(CNN_kernel_size_3, CNN_kernel_size_33), activation='relu',
                               padding='same', name="Conv-3")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    if pooling_2 == 'max':
        x = tf.keras.layers.MaxPooling2D(pool_size=(pool_kernel_size_3, pool_kernel_size_33))(x)
    else:
        x = tf.keras.layers.AveragePooling2D(pool_size=(pool_kernel_size_3, pool_kernel_size_33))(x)
    x = tf.keras.layers.Dropout(dropout2)(x)

    x = tf.keras.layers.Conv2D(filters=16, kernel_size=(7, 5), activation='relu', padding='same', name="Conv-3_0")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Dropout(0.3)(x)

    if pooling_3 == 'max':
        x = tf.keras.layers.GlobalMaxPool2D()(x)
    elif pooling_3 == 'avg':
        x = tf.keras.layers.GlobalAvgPool2D()(x)
    elif pooling_3 == 'clear':
        pass

    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(units, activation='relu', name="lastLayer")(x)
    x = tf.keras.layers.Dropout(dropout3)(x)
    output = tf.keras.layers.Dense(5, activation='softmax')(x)
    model = tf.keras.Model(input, output)
    return model
def miniBM2_1(input_Shape):
    filterL1 = 10
    filterL2 = 14
    filterL3 = 28
    dropout1 = 0.1
    dropout2 = 0.3
    dropout3 = 0.4
    CNN_kernel_size_1 = 7
    CNN_kernel_size_11 = 5
    CNN_kernel_size_2 = 7
    CNN_kernel_size_22 = 3
    CNN_kernel_size_3 = 3
    CNN_kernel_size_33 = 7
    pool_kernel_size_2 = 1
    pool_kernel_size_22 = 2
    pool_kernel_size_3 = 2
    pool_kernel_size_33 = 2
    pooling_1 = 'max'
    pooling_2 = 'max'
    pooling_3 = 'avg'
    units = 256

    input = tf.keras.layers.Input(shape=input_Shape)
    # CNN layer #1:
    x = tf.keras.layers.Conv2D(filters=filterL1, kernel_size=(CNN_kernel_size_1, CNN_kernel_size_11), padding='same',
                               name="Conv-1")(input)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)

    # CNN layer #2:
    x = tf.keras.layers.Conv2D(filters=filterL2, kernel_size=(CNN_kernel_size_2, CNN_kernel_size_22), activation='relu',
                               padding='same', name="Conv-2")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    if pooling_1 == 'max':
        x = tf.keras.layers.MaxPooling2D(pool_size=(pool_kernel_size_2, pool_kernel_size_22))(x)
    else:
        x = tf.keras.layers.AveragePooling2D(pool_size=(pool_kernel_size_2, pool_kernel_size_22))(x)
    x = tf.keras.layers.Dropout(dropout1)(x)

    # CNN layer #3:
    x = tf.keras.layers.Conv2D(filters=filterL3, kernel_size=(CNN_kernel_size_3, CNN_kernel_size_33), activation='relu',
                               padding='same', name="Conv-3")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    if pooling_2 == 'max':
        x = tf.keras.layers.MaxPooling2D(pool_size=(pool_kernel_size_3, pool_kernel_size_33))(x)
    else:
        x = tf.keras.layers.AveragePooling2D(pool_size=(pool_kernel_size_3, pool_kernel_size_33))(x)
    x = tf.keras.layers.Dropout(dropout2)(x)

    x = tf.keras.layers.Conv2D(filters=16, kernel_size=(7, 5), activation='relu', padding='same', name="Conv-3_0")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Dropout(0.3)(x)

    if pooling_3 == 'max':
        x = tf.keras.layers.GlobalMaxPool2D()(x)
    elif pooling_3 == 'avg':
        x = tf.keras.layers.GlobalAvgPool2D()(x)
    elif pooling_3 == 'clear':
        pass

    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(units, activation='relu', name="lastLayer")(x)
    x = tf.keras.layers.Dropout(dropout3)(x)
    output = tf.keras.layers.Dense(6, activation='softmax')(x)
    model = tf.keras.Model(input, output)
    return model

def BM2_5Dense(input_Shape):
    filterL1 = 20
    filterL2 = 28
    filterL3 = 28
    dropout1 = 0.1
    dropout2 = 0.3
    dropout3 = 0.4
    CNN_kernel_size_1 = 7
    CNN_kernel_size_11 = 5
    CNN_kernel_size_2 = 7
    CNN_kernel_size_22 = 3
    CNN_kernel_size_3 = 3
    CNN_kernel_size_33 = 7
    pool_kernel_size_2 = 1
    pool_kernel_size_22 = 2
    pool_kernel_size_3 = 2
    pool_kernel_size_33 = 2
    pooling_1 = 'max'
    pooling_2 = 'max'
    pooling_3 = 'avg'
    units = 256

    input = tf.keras.layers.Input(shape=input_Shape)
    # CNN layer #1:
    x = tf.keras.layers.Conv2D(filters=filterL1, kernel_size=(CNN_kernel_size_1, CNN_kernel_size_11), padding='same',
                               name="Conv-1")(input)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)

    # CNN layer #2:
    x = tf.keras.layers.Conv2D(filters=filterL2, kernel_size=(CNN_kernel_size_2, CNN_kernel_size_22), activation='relu',
                               padding='same', name="Conv-2")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    if pooling_1 == 'max':
        x = tf.keras.layers.MaxPooling2D(pool_size=(pool_kernel_size_2, pool_kernel_size_22))(x)
    else:
        x = tf.keras.layers.AveragePooling2D(pool_size=(pool_kernel_size_2, pool_kernel_size_22))(x)
    x = tf.keras.layers.Dropout(dropout1)(x)

    # CNN layer #3:
    x = tf.keras.layers.Conv2D(filters=filterL3, kernel_size=(CNN_kernel_size_3, CNN_kernel_size_33), activation='relu',
                               padding='same', name="Conv-3")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    if pooling_2 == 'max':
        x = tf.keras.layers.MaxPooling2D(pool_size=(pool_kernel_size_3, pool_kernel_size_33))(x)
    else:
        x = tf.keras.layers.AveragePooling2D(pool_size=(pool_kernel_size_3, pool_kernel_size_33))(x)
    x = tf.keras.layers.Dropout(dropout2)(x)

    # CNN layer #4:
    x = tf.keras.layers.Conv2D(filters=16, kernel_size=(7, 5), activation='relu', padding='same', name="Conv-3_0")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Dropout(0.3)(x)

    if pooling_3 == 'max':
        x = tf.keras.layers.GlobalMaxPool2D()(x)
    elif pooling_3 == 'avg':
        x = tf.keras.layers.GlobalAvgPool2D()(x)
    elif pooling_3 == 'clear':
        pass

    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(units, activation='relu', name="lastLayer")(x)
    x = tf.keras.layers.Dropout(dropout3)(x)
    output = tf.keras.layers.Dense(5, activation='softmax')(x)
    model = tf.keras.Model(input, output)
    return model
def miniBM2_1BackupALLmetade(input_Shape):
    filterL1 = 10
    filterL2 = 14
    filterL3 = 14
    dropout1 = 0.1
    dropout2 = 0.3
    dropout3 = 0.4
    CNN_kernel_size_1 = 7
    CNN_kernel_size_11 = 5
    CNN_kernel_size_2 = 7
    CNN_kernel_size_22 = 3
    CNN_kernel_size_3 = 3
    CNN_kernel_size_33 = 7
    pool_kernel_size_2 = 1
    pool_kernel_size_22 = 2
    pool_kernel_size_3 = 2
    pool_kernel_size_33 = 2
    pooling_1 = 'max'
    pooling_2 = 'max'
    pooling_3 = 'avg'
    units = 128

    input = tf.keras.layers.Input(shape=input_Shape)
    # CNN layer #1:
    x = tf.keras.layers.Conv2D(filters=filterL1, kernel_size=(CNN_kernel_size_1, CNN_kernel_size_11), padding='same',
                               name="Conv-1")(input)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)

    # CNN layer #2:
    x = tf.keras.layers.Conv2D(filters=filterL2, kernel_size=(CNN_kernel_size_2, CNN_kernel_size_22), activation='relu',
                               padding='same', name="Conv-2")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    if pooling_1 == 'max':
        x = tf.keras.layers.MaxPooling2D(pool_size=(pool_kernel_size_2, pool_kernel_size_22))(x)
    else:
        x = tf.keras.layers.AveragePooling2D(pool_size=(pool_kernel_size_2, pool_kernel_size_22))(x)
    x = tf.keras.layers.Dropout(dropout1)(x)

    # CNN layer #3:
    x = tf.keras.layers.Conv2D(filters=filterL3, kernel_size=(CNN_kernel_size_3, CNN_kernel_size_33), activation='relu',
                               padding='same', name="Conv-3")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    if pooling_2 == 'max':
        x = tf.keras.layers.MaxPooling2D(pool_size=(pool_kernel_size_3, pool_kernel_size_33))(x)
    else:
        x = tf.keras.layers.AveragePooling2D(pool_size=(pool_kernel_size_3, pool_kernel_size_33))(x)
    x = tf.keras.layers.Dropout(dropout2)(x)

    x = tf.keras.layers.Conv2D(filters=8, kernel_size=(7, 5), activation='relu', padding='same', name="Conv-3_0")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Dropout(0.3)(x)

    if pooling_3 == 'max':
        x = tf.keras.layers.GlobalMaxPool2D()(x)
    elif pooling_3 == 'avg':
        x = tf.keras.layers.GlobalAvgPool2D()(x)
    elif pooling_3 == 'clear':
        pass

    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(units, activation='relu', name="lastLayer")(x)
    x = tf.keras.layers.Dropout(dropout3)(x)
    output = tf.keras.layers.Dense(6, activation='softmax')(x)
    model = tf.keras.Model(input, output)
    return model

def BC_resnet(input_Shape):
    # make it [batch, time, feature, 1]
    blocks_n = [2, 2, 4, 4] #'Number of BC-ResBlocks.'
    dropouts = [0.5, 0.5, 0.5, 0.5] #'List of dropouts for BC-ResBlock.'
    filters = [8, 12, 16, 20] #'Number of filters in every BC-ResBlock.'
    strides = (1,1),(1,2),(1,2),(1,1) #'Strides applied in every TransitionBlock.'
    dilations = (1,1), (2,1), (3,1), (3,1) #'Dilations applied in every BC-ResBlocks.'
    paddings = 'same' #'Paddings in time applied in every BC-ResBlocks.'
    sub_groups = 4,4,4,4 #5 #'Number of groups for SubSpectralNormalization.'
    pools = 1, 1, 1, 1 #'Pooling in time after every BC-ResBlock.'
    max_pool = 0 #'Pooling type: 0 - average pooling; 1 - max pooling'
    first_filters = 16  # 'Number of filters in the first conv layer.'
    last_filters = 32  # 'Number of filters in the last conv layer.'

    # input = tf.keras.Input((input_Shape[0], input_Shape[1]))
    input = tf.keras.layers.Input(shape=input_Shape) #[batch, freq, time, 1]
    net = input
    net=tf.keras.backend.squeeze(net, axis=3)
    # make it [batch, time, feature, 1]
    net=tf.keras.layers.Permute(dims=(2,1))(net) #inverter axis
    net = tf.keras.backend.expand_dims(net, axis=-1)

    net = tf.keras.layers.Conv2D(filters=first_filters, kernel_size=5, strides=(2), padding='same')(net)
    for n, n_filters, dilation, stride, dropout, pool,sub_group in zip(blocks_n, filters, dilations, strides, dropouts, pools,sub_groups):
        net = TransitionBlock(
            n_filters,
            dilation,
            stride,
            paddings,
            dropout,
            sub_groups=sub_group)(
            net)

        for _ in range(n):
            net = NormalBlock(
                n_filters,
                dilation,
                1,
                'same',
                dropout,
                sub_groups=sub_group)(
                net)
        if pool > 1:
            if(max_pool==1):
                net = tf.keras.layers.MaxPooling2D(pool_size=(pool, 1),strides=(pool, 1))(net)
            else:
                net = tf.keras.layers.AveragePooling2D(pool_size=(pool, 1),strides=(pool, 1))(net)

    net = tf.keras.layers.DepthwiseConv2D(kernel_size=5,padding='same')(net)
    # average out frequency dim
    net = tf.keras.backend.mean(net, axis=2, keepdims=True)
    net = tf.keras.layers.Conv2D(filters=last_filters, kernel_size=1, use_bias=False)(net)

    # average out time dim
    net = tf.keras.layers.GlobalAveragePooling2D()(net) # tensorflow abaixo de 2.6 nao tem keepdims=True usar net = tf.keras.layers.Reshape((1, 1, input_shape[-1]))(y)
    net = tf.keras.layers.Reshape((1, 1, net.shape[-1]))(net)

    #decisao final
    label_count=10
    net = tf.keras.layers.Conv2D(filters=label_count,kernel_size=1, use_bias=False)(net)
    # 1 and 2 dims are equal to 1
    net = tf.squeeze(net, [1, 2])
    output = tf.keras.layers.Activation('softmax')(net)
    model = tf.keras.Model(input, output)
    return model
#Rede BC_rednet modificada para o DCASE2021
def BC_resnet_DCASE2021(input_Shape):
    # make it [batch, time, feature, 1]
    blocks_n = [2, 2, 2, 3]  #'Number of BC-ResBlocks.'
    dropouts = [0.5, 0.5, 0.5, 0.5] #'List of dropouts for BC-ResBlock.'
    filters = [8, 12, 16, 20] #'Number of filters in every BC-ResBlock.'
    strides = (1,1),(1,2),(1,2),(1,1) #'Strides applied in every TransitionBlock.'
    dilations = (1,1), (2,1), (3,1), (3,1) #'Dilations applied in every BC-ResBlocks.'
    paddings = 'same' #'Paddings in time applied in every BC-ResBlocks.'
    sub_groups = 4, 4, 1, 1 #5 #'Number of groups for SubSpectralNormalization.'  (sub_Group=1 = batch norm)
    max_pools = 1, 1, 0, 0  # 'Pooling type: 0 - no pooling; 1 - max pooling'
    first_filters = 10  # 'Number of filters in the first conv layer.'
    last_filters = 32  # 'Number of filters in the last conv layer.'


    # input = tf.keras.Input((input_Shape[0], input_Shape[1]))
    input = tf.keras.layers.Input(shape=input_Shape) #[batch, freq, time, 1]
    net = input
    # modificar inputs
    net=tf.keras.backend.squeeze(net, axis=3)
    # make it [batch, time, feature, 1]
    net=tf.keras.layers.Permute(dims=(2,1))(net) #inverter axis
    net = tf.keras.backend.expand_dims(net, axis=-1)

    net = tf.keras.layers.Conv2D(filters=first_filters, kernel_size=5, strides=(2), padding='same')(net)
    for n, n_filters, dilation, stride, dropout, sub_group,max_pool in zip(blocks_n, filters, dilations, strides, dropouts, sub_groups,max_pools):
        net = TransitionBlock(
            n_filters,
            dilation,
            stride,
            paddings,
            dropout,
            sub_groups=sub_group)(
            net)

        for _ in range(n):
            net = NormalBlock(
                n_filters,
                dilation,
                1,
                'same',
                dropout,
                sub_groups=sub_group)(
                net)
        if(max_pool==1):
            net = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(net)

    # net = tf.keras.layers.DepthwiseConv2D(kernel_size=5, padding='same')(net) #removido em BC_resnetDcase
    # average out frequency dim
    net = tf.keras.backend.mean(net, axis=2, keepdims=True)
    net = tf.keras.layers.Conv2D(filters=last_filters, kernel_size=1, use_bias=False)(net)

    # average out time dim
    net = tf.keras.layers.GlobalAveragePooling2D()(net)  # tensorflow abaixo de 2.6 nao tem keepdims=True usar net = tf.keras.layers.Reshape((1, 1, input_shape[-1]))(y)
    net = tf.keras.layers.Reshape((1, 1, net.shape[-1]))(net)

    label_count = 10
    net = tf.keras.layers.Conv2D(filters=label_count, kernel_size=1, use_bias=False)(net)
    # 1 and 2 dims are equal to 1
    net = tf.squeeze(net, [1, 2])
    output = tf.keras.layers.Activation('softmax')(net)
    model = tf.keras.Model(input, output)
    return model
#Rede BC_resnet_DCASE2021 Com batch norm em vez de SubSpectralnorm
def BC_resnet_DCASE2021_noSSN(input_Shape):
    # make it [batch, time, feature, 1]
    blocks_n = [2, 2, 2, 3]  #'Number of BC-ResBlocks.'
    dropouts = [0.5, 0.5, 0.5, 0.5] #'List of dropouts for BC-ResBlock.'
    filters = [8, 12, 16, 20] #'Number of filters in every BC-ResBlock.'
    strides = (1,1),(1,2),(1,2),(1,1) #'Strides applied in every TransitionBlock.'
    dilations = (1,1), (2,1), (3,1), (3,1) #'Dilations applied in every BC-ResBlocks.'
    paddings = 'same' #'Paddings in time applied in every BC-ResBlocks.'
    sub_groups = 1, 1, 1, 1 #5 #'Number of groups for SubSpectralNormalization.'  (sub_Group=1 = batch norm)
    max_pools = 1, 1, 0, 0  # 'Pooling type: 0 - no pooling; 1 - max pooling'
    first_filters = 10  # 'Number of filters in the first conv layer.'
    last_filters = 32  # 'Number of filters in the last conv layer.'


    # input = tf.keras.Input((input_Shape[0], input_Shape[1]))
    input = tf.keras.layers.Input(shape=input_Shape) #[batch, freq, time, 1]
    net = input
    # modificar inputs
    net=tf.keras.backend.squeeze(net, axis=3)
    # make it [batch, time, feature, 1]
    net=tf.keras.layers.Permute(dims=(2,1))(net) #inverter axis
    net = tf.keras.backend.expand_dims(net, axis=-1)

    net = tf.keras.layers.Conv2D(filters=first_filters, kernel_size=5, strides=(2), padding='same')(net)
    for n, n_filters, dilation, stride, dropout, sub_group,max_pool in zip(blocks_n, filters, dilations, strides, dropouts, sub_groups,max_pools):
        net = TransitionBlock(
            n_filters,
            dilation,
            stride,
            paddings,
            dropout,
            sub_groups=sub_group)(
            net)

        for _ in range(n):
            net = NormalBlock(
                n_filters,
                dilation,
                1,
                'same',
                dropout,
                sub_groups=sub_group)(
                net)
        if(max_pool==1):
            net = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(net)

    # net = tf.keras.layers.DepthwiseConv2D(kernel_size=5, padding='same')(net) #removido em BC_resnetDcase
    # average out frequency dim
    net = tf.keras.backend.mean(net, axis=2, keepdims=True)
    net = tf.keras.layers.Conv2D(filters=last_filters, kernel_size=1, use_bias=False)(net)

    # average out time dim
    net = tf.keras.layers.GlobalAveragePooling2D()(net)  # tensorflow abaixo de 2.6 nao tem keepdims=True usar net = tf.keras.layers.Reshape((1, 1, input_shape[-1]))(y)
    net = tf.keras.layers.Reshape((1, 1, net.shape[-1]))(net)

    label_count = 10
    net = tf.keras.layers.Conv2D(filters=label_count, kernel_size=1, use_bias=False)(net)
    # 1 and 2 dims are equal to 1
    net = tf.squeeze(net, [1, 2])
    output = tf.keras.layers.Activation('softmax')(net)
    model = tf.keras.Model(input, output)
    return model
def BC_resnet_DCASE2021_MOD8(input_Shape):
    # make it [batch, time, feature, 1]
    blocks_n = [2, 2, 2, 3]  #'Number of BC-ResBlocks.'
    dropouts = [0.5, 0.5, 0.5, 0.5] #'List of dropouts for BC-ResBlock.'
    filters = [8, 12, 16, 20] #'Number of filters in every BC-ResBlock.'
    strides = (1,1),(1,2),(1,2),(1,1) #'Strides applied in every TransitionBlock.'
    dilations = (1,1), (2,1), (3,1), (3,1) #'Dilations applied in every BC-ResBlocks.'
    paddings = 'same' #'Paddings in time applied in every BC-ResBlocks.'
    sub_groups = 1, 1, 1, 1 #5 #'Number of groups for SubSpectralNormalization.'  (sub_Group=1 = batch norm)
    max_pools = 1, 1, 0, 0  # 'Pooling type: 0 - no pooling; 1 - max pooling'
    first_filters = 80  # 'Number of filters in the first conv layer.'
    last_filters = 32  # 'Number of filters in the last conv layer.'


    # input = tf.keras.Input((input_Shape[0], input_Shape[1]))
    input = tf.keras.layers.Input(shape=input_Shape) #[batch, freq, time, 1]
    net = input
    # modificar inputs
    net=tf.keras.backend.squeeze(net, axis=3)
    # make it [batch, time, feature, 1]
    net=tf.keras.layers.Permute(dims=(2,1))(net) #inverter axis
    net = tf.keras.backend.expand_dims(net, axis=-1)

    net = tf.keras.layers.Conv2D(filters=first_filters, kernel_size=5, strides=(2), padding='same')(net)
    for n, n_filters, dilation, stride, dropout, sub_group,max_pool in zip(blocks_n, filters, dilations, strides, dropouts, sub_groups,max_pools):
        net = TransitionBlock(
            n_filters,
            dilation,
            stride,
            paddings,
            dropout,
            sub_groups=sub_group)(
            net)

        for _ in range(n):
            net = NormalBlock(
                n_filters,
                dilation,
                1,
                'same',
                dropout,
                sub_groups=sub_group)(
                net)
        if(max_pool==1):
            net = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(net)

    # net = tf.keras.layers.DepthwiseConv2D(kernel_size=5, padding='same')(net) #removido em BC_resnetDcase
    # average out frequency dim
    net = tf.keras.backend.mean(net, axis=2, keepdims=True)
    net = tf.keras.layers.Conv2D(filters=last_filters, kernel_size=1, use_bias=False)(net)

    # average out time dim
    net = tf.keras.layers.GlobalAveragePooling2D()(net)  # tensorflow abaixo de 2.6 nao tem keepdims=True usar net = tf.keras.layers.Reshape((1, 1, input_shape[-1]))(y)
    net = tf.keras.layers.Reshape((1, 1, net.shape[-1]))(net)

    label_count = 10
    net = tf.keras.layers.Conv2D(filters=label_count, kernel_size=1, use_bias=False)(net)
    # 1 and 2 dims are equal to 1
    net = tf.squeeze(net, [1, 2])
    output = tf.keras.layers.Activation('softmax')(net)
    model = tf.keras.Model(input, output)
    return model


#### MobileNetV3
def relu(x):
    return tf.keras.layers.ReLU()(x)
def hard_sigmoid(x):
    return tf.keras.layers.ReLU(6.)(x + 3.) * (1. / 6.)
def hard_swish(x):
   return tf.keras.layers.Multiply()([x, hard_sigmoid(x)])
def _inverted_res_block(x, expansion, filters, kernel_size, stride, se_ratio,activation, block_id):
  channel_axis = -1
  shortcut = x
  prefix = 'expanded_conv/'
  infilters = tf.keras.backend.int_shape(x)[channel_axis]
  if block_id:
    # Expand
    prefix = 'expanded_conv_{}/'.format(block_id)
    x = tf.keras.layers.Conv2D(
        _depth(infilters * expansion),
        kernel_size=1,
        padding='same',
        use_bias=False,
        name=prefix + 'expand')(
            x)
    x = tf.keras.layers.BatchNormalization(
        axis=channel_axis,
        epsilon=1e-3,
        momentum=0.999,
        name=prefix + 'expand/BatchNorm')(
            x)
    x = activation(x)

  if stride == 2:
    x = tf.keras.layers.ZeroPadding2D(
        padding=correct_pad(x, kernel_size),
        name=prefix + 'depthwise/pad')(
            x)
  x = tf.keras.layers.DepthwiseConv2D(
      kernel_size,
      strides=stride,
      padding='same' if stride == 1 else 'valid',
      use_bias=False,
      name=prefix + 'depthwise')(
          x)
  x = tf.keras.layers.BatchNormalization(
      axis=channel_axis,
      epsilon=1e-3,
      momentum=0.999,
      name=prefix + 'depthwise/BatchNorm')(
          x)
  x = activation(x)

  if se_ratio:
    x = _se_block(x, _depth(infilters * expansion), se_ratio, prefix)

  x = tf.keras.layers.Conv2D(
      filters,
      kernel_size=1,
      padding='same',
      use_bias=False,
      name=prefix + 'project')(
          x)
  x = tf.keras.layers.BatchNormalization(
      axis=channel_axis,
      epsilon=1e-3,
      momentum=0.999,
      name=prefix + 'project/BatchNorm')(
          x)

  if stride == 1 and infilters == filters:
    x = tf.keras.layers.Add(name=prefix + 'Add')([shortcut, x])
  return x
def _depth(v, divisor=8, min_value=None):
  if min_value is None:
    min_value = divisor
  new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
  # Make sure that round down does not go down by more than 10%.
  if new_v < 0.9 * v:
    new_v += divisor
  return new_v
def correct_pad(inputs, kernel_size):
    """Returns a tuple for zero-padding for 2D convolution with downsampling.
    # Arguments
        input_size: An integer or tuple/list of 2 integers.
        kernel_size: An integer or tuple/list of 2 integers.
    # Returns
        A tuple.
    """
    img_dim = 2 if tf.keras.backend.image_data_format() == 'channels_first' else 1
    input_size = tf.keras.backend.int_shape(inputs)[img_dim:(img_dim + 2)]

    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)

    if input_size[0] is None:
        adjust = (1, 1)
    else:
        adjust = (1 - input_size[0] % 2, 1 - input_size[1] % 2)

    correct = (kernel_size[0] // 2, kernel_size[1] // 2)

    return ((correct[0] - adjust[0], correct[0]),
            (correct[1] - adjust[1], correct[1]))
def _se_block(inputs, filters, se_ratio, prefix):
  x = tf.keras.layers.GlobalAveragePooling2D(name=prefix + 'squeeze_excite/AvgPool')( inputs)  # keepdims=True não funca
  x = tf.keras.layers.Reshape((1, 1, x.shape[-1]))(x) # keepdims=True não funca em tf2.3 usar a seguinte linha
  x = tf.keras.layers.Conv2D(
      _depth(filters * se_ratio),
      kernel_size=1,
      padding='same',
      name=prefix + 'squeeze_excite/Conv')(
          x)
  x = tf.keras.layers.ReLU(name=prefix + 'squeeze_excite/Relu')(x)
  x = tf.keras.layers.Conv2D(
      filters,
      kernel_size=1,
      padding='same',
      name=prefix + 'squeeze_excite/Conv_1')(
          x)
  x = hard_sigmoid(x)
  x = tf.keras.layers.Multiply(name=prefix + 'squeeze_excite/Mul')([inputs, x])
  return x
def MobileNetV3(stack_fn,input_shape,last_point_ch,alpha=1.0,model_type='large',minimalistic=False,include_top=True,classes=10,pooling=None,dropout_rate=0.2,classifier_activation='softmax'):

  channel_axis = -1
  img_input = tf.keras.Input(input_shape)  # [batch,freq,time,1]

  if minimalistic:
    kernel = 3
    activation = relu
    se_ratio = None
  else:
    kernel = 5
    activation = hard_swish
    se_ratio = 0.25

  x = img_input
  x = tf.keras.layers.Conv2D(
      16,
      kernel_size=3,
      strides=(2, 2),
      padding='same',
      use_bias=False,
      name='Conv')(x)
  x = tf.keras.layers.BatchNormalization(
      axis=channel_axis, epsilon=1e-3,
      momentum=0.999, name='Conv/BatchNorm')(x)
  x = activation(x)

  x = stack_fn(x, kernel, activation, se_ratio)

  last_conv_ch = _depth(tf.keras.backend.int_shape(x)[channel_axis] * 6)

  # if the width multiplier is greater than 1 we
  # increase the number of output channels
  if alpha > 1.0:
    last_point_ch = _depth(last_point_ch * alpha)
  x = tf.keras.layers.Conv2D(
      last_conv_ch,
      kernel_size=1,
      padding='same',
      use_bias=False,
      name='Conv_1')(x)
  x = tf.keras.layers.BatchNormalization(
      axis=channel_axis, epsilon=1e-3,
      momentum=0.999, name='Conv_1/BatchNorm')(x)
  x = activation(x)
  if include_top:
    x = tf.keras.layers.GlobalAveragePooling2D()(x) #keepdims=True
    x = tf.keras.layers.Reshape((1, 1, x.shape[-1]))(x)  # keepdims=True não funca em tf2.3 usar a seguinte linha
    x = tf.keras.layers.Conv2D(
        last_point_ch,
        kernel_size=1,
        padding='same',
        use_bias=True,
        name='Conv_2')(x)
    x = activation(x)

    if dropout_rate > 0:
      x = tf.keras.layers.Dropout(dropout_rate)(x)
    x = tf.keras.layers.Conv2D(classes, kernel_size=1, padding='same', name='Logits')(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Activation('softmax',name='Predictions')(x)
    inputs = img_input
    model = tf.keras.Model(inputs, x, name='MobilenetV3' + model_type)
    return model
  # else:
  #   if pooling == 'avg':
  #     x = tf.keras.layers.GlobalAveragePooling2D(name='avg_pool')(x)
  #   elif pooling == 'max':
  #     x = tf.keras.layers.GlobalMaxPooling2D(name='max_pool')(x)
  #   inputs = img_input
  #   model = tf.keras.Model(inputs, x, name='MobilenetV3' + model_type)
  #   return model
def MobileNetV3Small(input_shape,
                     alpha=1.0,
                     minimalistic=False,
                     include_top=True,
                     classes=10,
                     pooling=None,
                     dropout_rate=0.2,
                     classifier_activation='softmax'):
  def stack_fn(x, kernel, activation, se_ratio):

    def depth(d):
      return _depth(d * alpha)

    x = _inverted_res_block(x, 1, depth(16), 3, 2, se_ratio, relu, 0)
    x = _inverted_res_block(x, 72. / 16, depth(24), 3, 2, None, relu, 1)
    x = _inverted_res_block(x, 88. / 24, depth(24), 3, 1, None, relu, 2)
    x = _inverted_res_block(x, 4, depth(40), kernel, 2, se_ratio, activation, 3)
    x = _inverted_res_block(x, 6, depth(40), kernel, 1, se_ratio, activation, 4)
    x = _inverted_res_block(x, 6, depth(40), kernel, 1, se_ratio, activation, 5)
    x = _inverted_res_block(x, 3, depth(48), kernel, 1, se_ratio, activation, 6)
    x = _inverted_res_block(x, 3, depth(48), kernel, 1, se_ratio, activation, 7)
    x = _inverted_res_block(x, 6, depth(96), kernel, 2, se_ratio, activation, 8)
    x = _inverted_res_block(x, 6, depth(96), kernel, 1, se_ratio, activation, 9)
    x = _inverted_res_block(x, 6, depth(96), kernel, 1, se_ratio, activation,10)
    return x

  return MobileNetV3(stack_fn,input_shape, 1024, alpha, 'small', minimalistic,  include_top, classes, pooling,
  dropout_rate, classifier_activation)
