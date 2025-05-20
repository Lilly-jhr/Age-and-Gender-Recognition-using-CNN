
import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Conv2D, BatchNormalization, Activation, MaxPooling2D,
    GlobalAveragePooling2D, Dense, Dropout
)
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
import os 

DEFAULT_INPUT_SHAPE = (128, 128, 3)
DEFAULT_L2_REG = 1e-5 # L2 regularization factor
DEFAULT_LEARNING_RATE = 0.0005

def build_age_gender_model(input_shape=DEFAULT_INPUT_SHAPE,
                           l2_reg=DEFAULT_L2_REG,
                           optimizer_lr=DEFAULT_LEARNING_RATE):
    """
    Builds a multi-output CNN model for age and gender prediction.
    
    Args:
        input_shape (tuple): Shape of the input images (height, width, channels).
        l2_reg (float): L2 regularization factor for Conv and Dense layers.
        optimizer_lr (float): Learning rate for the Adam optimizer.

    Returns:
        tf.keras.Model: The compiled Keras model.
    """
    inputs = Input(shape=input_shape, name='input_image')

    # Shared Convolutional Base 
    x = inputs

    # Block 1
    x = Conv2D(32, (3, 3), padding='same', use_bias=False, kernel_regularizer=l2(l2_reg), name='conv1_1')(x)
    x = BatchNormalization(name='bn1_1')(x)
    x = Activation('relu', name='relu1_1')(x)
    x = MaxPooling2D(pool_size=(2, 2), name='pool1')(x)
    x = Dropout(0.2, name='drop1')(x)

    # Block 2
    x = Conv2D(64, (3, 3), padding='same', use_bias=False, kernel_regularizer=l2(l2_reg), name='conv2_1')(x)
    x = BatchNormalization(name='bn2_1')(x)
    x = Activation('relu', name='relu2_1')(x)
    x = MaxPooling2D(pool_size=(2, 2), name='pool2')(x)
    x = Dropout(0.25, name='drop2')(x)

    # Block 3 (128 filters, 2 conv layers)
    x = Conv2D(128, (3, 3), padding='same', use_bias=False, kernel_regularizer=l2(l2_reg), name='conv3_1')(x)
    x = BatchNormalization(name='bn3_1')(x)
    x = Activation('relu', name='relu3_1')(x)
    x = Conv2D(128, (3, 3), padding='same', use_bias=False, kernel_regularizer=l2(l2_reg), name='conv3_2')(x)
    x = BatchNormalization(name='bn3_2')(x)
    x = Activation('relu', name='relu3_2')(x)
    x = MaxPooling2D(pool_size=(2, 2), name='pool3')(x)
    x = Dropout(0.3, name='drop3')(x)

    # Block 4 (256 filters, 3 conv layers)
    x = Conv2D(256, (3, 3), padding='same', use_bias=False, kernel_regularizer=l2(l2_reg), name='conv4_1')(x)
    x = BatchNormalization(name='bn4_1')(x)
    x = Activation('relu', name='relu4_1')(x)
    x = Conv2D(256, (3, 3), padding='same', use_bias=False, kernel_regularizer=l2(l2_reg), name='conv4_2')(x)
    x = BatchNormalization(name='bn4_2')(x)
    x = Activation('relu', name='relu4_2')(x)
    x = Conv2D(256, (3, 3), padding='same', use_bias=False, kernel_regularizer=l2(l2_reg), name='conv4_3')(x)
    x = BatchNormalization(name='bn4_3')(x)
    x = Activation('relu', name='relu4_3')(x)
    x = MaxPooling2D(pool_size=(2, 2), name='pool4')(x)
    x = Dropout(0.35, name='drop4')(x)

    # Global Average Pooling before FC layers
    shared_features = GlobalAveragePooling2D(name='global_avg_pool')(x)

    # Gender Prediction Branch
    gender_branch = Dense(64, use_bias=False, kernel_regularizer=l2(l2_reg), name='gender_fc1')(shared_features)
    gender_branch = BatchNormalization(name='gender_bn1')(gender_branch)
    gender_branch = Activation('relu', name='gender_relu1')(gender_branch)
    gender_branch = Dropout(0.4, name='gender_drop1')(gender_branch)

    gender_branch = Dense(32, use_bias=False, kernel_regularizer=l2(l2_reg), name='gender_fc2')(gender_branch)
    gender_branch = BatchNormalization(name='gender_bn2')(gender_branch)
    gender_branch = Activation('relu', name='gender_relu2')(gender_branch)
    gender_branch = Dropout(0.5, name='gender_drop2')(gender_branch)
    
    gender_output = Dense(1, activation='sigmoid', name='gender_output')(gender_branch)

    # Age Prediction Branch
    age_branch = Dense(128, use_bias=False, kernel_regularizer=l2(l2_reg), name='age_fc1')(shared_features)
    age_branch = BatchNormalization(name='age_bn1')(age_branch)
    age_branch = Activation('relu', name='age_relu1')(age_branch)
    age_branch = Dropout(0.3, name='age_drop1')(age_branch)

    age_branch = Dense(64, use_bias=False, kernel_regularizer=l2(l2_reg), name='age_fc2')(age_branch)
    age_branch = BatchNormalization(name='age_bn2')(age_branch)
    age_branch = Activation('relu', name='age_relu2')(age_branch)
    age_branch = Dropout(0.4, name='age_drop2')(age_branch)

    age_branch = Dense(32, use_bias=False, kernel_regularizer=l2(l2_reg), name='age_fc3')(age_branch)
    age_branch = BatchNormalization(name='age_bn3')(age_branch)
    age_branch = Activation('relu', name='age_relu3')(age_branch)
    age_branch = Dropout(0.5, name='age_drop3')(age_branch)
    
    age_output = Dense(1, activation='linear', name='age_output')(age_branch)


    model = Model(inputs=inputs, outputs=[age_output, gender_output], name="age_gender_cnn_v2")

    # compile the model
    losses = {
        'age_output': 'mean_squared_error',
        'gender_output': 'binary_crossentropy'
    }
    loss_weights = { 
        'age_output': 1.0, 
        'gender_output': 1.0 
    }
    metrics = {
        'age_output': 'mae',
        'gender_output': 'accuracy'
    }
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=optimizer_lr)

    model.compile(
        optimizer=optimizer,
        loss=losses,
        loss_weights=loss_weights,
        metrics=metrics
    )
    return model

# for testing model building and saving diagram 
if __name__ == '__main__':
    print(f"Using TensorFlow version: {tf.__version__}")

    print("Building the age-gender CNN model...")
    model = build_age_gender_model()
    
    print("\nModel Summary:")
    model.summary(line_length=120) 
    

    output_dir = "models"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    # save model architecture diagram
    diagram_path = os.path.join(output_dir, 'model_architecture_v2.png')
    try:
        tf.keras.utils.plot_model(
            model,
            to_file=diagram_path,
            show_shapes=True,
            show_dtype=False, 
            show_layer_names=True,
            rankdir='TB', 
            expand_nested=False, 
            show_layer_activations=True 
        )
        print(f"\nModel architecture diagram saved to {diagram_path}")
    except ImportError:
        print("\nPlotting model architecture skipped: pydot or graphviz not found.")
    except Exception as e:
        print(f"\nError plotting model: {e}. Skipping model plot.")

    # model_json = model.to_json()
    # with open(os.path.join(output_dir, "model_architecture.json"), "w") as json_file:
    #     json_file.write(model_json)
    # print(f"Model architecture saved to {os.path.join(output_dir, 'model_architecture.json')}")