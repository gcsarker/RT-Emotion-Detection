import tensorflow as tf

IMG_SIZE = 299
IMAGE_SIZE = (IMG_SIZE,IMG_SIZE)
n_classes = 6

def get_model():

    inception_resnet = tf.keras.applications.InceptionResNetV2(
    include_top=False,
    weights= None,
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    pooling='max',
    classes=n_classes,
    classifier_activation='softmax'
    )
    
    x = tf.keras.layers.Dense(512, activation= 'relu')(inception_resnet.output)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dense(n_classes, activation = 'softmax')(x)
    model = tf.keras.models.Model(inputs = inception_resnet.input, outputs = x)

    return model