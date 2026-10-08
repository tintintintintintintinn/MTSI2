import numpy as np

all_images = np.load("Pipeline/pipeline_images.npy")
predections = np.load("Pipeline/pipeline_predections.npy")
assert len(all_images) == len(predections)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os

SCALE = 0.5
SEED = 0

def tan_to_theta(tan_theta):
    return np.arctan(tan_theta)

def theta_to_target(theta):
    phi = 2.0 * theta
    return np.stack([np.cos(phi), np.sin(phi)], axis=-1).astype('float32')

def target_to_theta(target):
    phi = np.arctan2(target[..., 1], target[..., 0])
    return 0.5 * phi

@keras.utils.register_keras_serializable(package="Custom")
def angular_error_deg(y_true, y_pred):
    phi_true = tf.atan2(y_true[:, 1], y_true[:, 0])
    phi_pred = tf.atan2(y_pred[:, 1], y_pred[:, 0])
    d = phi_pred - phi_true
    d = tf.atan2(tf.sin(d), tf.cos(d))
    return tf.abs(d/2) * (180.0 / np.pi)

@keras.utils.register_keras_serializable(package="Custom")
def mse_angle_loss(y_true, y_pred):
    return tf.square(angular_error_deg(y_true, y_pred))

@keras.utils.register_keras_serializable(package="Custom")
def mse_cos_sin_loss(y_true, y_pred):
    return tf.reduce_sum(tf.square((y_true - y_pred) * 30), axis=-1)

@keras.utils.register_keras_serializable(package="Custom")
def orientation_loss(y_true, y_pred):
    p = tf.math.l2_normalize(y_pred, axis=-1)
    cos_dphi = tf.reduce_sum(y_true * p, axis=-1)
    return 1.0 - cos_dphi

tan_theta = predections[:, 0]
theta = np.arctan(tan_theta)
targets = theta_to_target(theta)

np.save('Pipeline/pipeline_targets.npy', targets)

custom_objects = {'orientation_loss': orientation_loss,
                  'angular_error_deg': angular_error_deg}

def to_yellow(images_bgr):
    yellow = images_bgr[..., 1].astype(np.int16) - images_bgr[..., 0].astype(np.int16)
    return np.clip(yellow, 0, 255).astype(np.uint8)[..., np.newaxis]

images = to_yellow(np.load('Pipeline/pipeline_images.npy'))
targets = np.load('Pipeline/pipeline_targets.npy')
N, H, W, C = images.shape

rng = np.random.default_rng(SEED)
idx = rng.permutation(N)
n_train, n_val = int(0.70 * N), int(0.15 * N)
i_train, i_val, i_test = np.split(idx, [n_train, n_train + n_val])

x_train, y_train = images[i_train], targets[i_train]
x_val, y_val = images[i_val], targets[i_val]
x_test, y_test = images[i_test], targets[i_test]
# print(f'train {len(x_train)} | val {len(x_val)} | test {len(x_test)}')

new_h, new_w = int(round(H * SCALE)), int(round(W * SCALE))

import cv2

def resize_all(images, new_h, new_w):
    out = np.zeros((len(images), new_h, new_w) + images.shape[3:], dtype=images.dtype)
    for i, im in enumerate(images):
        resized = cv2.resize(im, (new_w, new_h), interpolation=cv2.INTER_AREA)
        if resized.ndim == 2:              # cv2.resize a supprimé l'axe de canal
            resized = resized[..., np.newaxis]
        out[i] = resized
    return out

images_small = resize_all(to_yellow(np.load('Pipeline/pipeline_images.npy')), new_h, new_w)
np.save('Pipeline/pipeline_yellow_small.npy', images_small)


model = keras.Sequential([
    layers.Input(shape=(H, W, C)),
    layers.Rescaling(1.0 / 255),
    layers.Resizing(new_h, new_w),
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.GlobalAveragePooling2D(),
    layers.Dense(64, activation='relu'),
    layers.Dense(2),
])

model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
              loss=mse_angle_loss,
              metrics=[angular_error_deg])

# model.compile(optimizer=keras.optimizers.Adam(1e-3),
#               loss=orientation_loss,
#               metrics=[angular_error_deg])

EPOCHS = int(os.environ.get('EPOCHS', 50))
BATCH_SIZE = 32

history = model.fit(x_train, y_train,
                    validation_data=(x_val, y_val),
                    epochs=EPOCHS, batch_size=BATCH_SIZE,
                    )

# ---------------------------------------------------------------- évaluation
test_loss, test_err = model.evaluate(x_test, y_test, verbose=0)
print(f'Test : perte = {test_loss:.4f} | erreur angulaire moyenne = {test_err:.2f} degrés')

model.save('Pipeline/pipeline_cnn.keras')

model = keras.models.load_model('Pipeline/pipeline_cnn.keras', custom_objects=custom_objects)

pred = model.predict(x_test[:8], verbose=0)
theta_pred = np.degrees(0.5 * np.arctan2(pred[:, 1], pred[:, 0]))
theta_true = np.degrees(0.5 * np.arctan2(y_test[:8, 1], y_test[:8, 0]))
for tp, tt in zip(theta_pred, theta_true):
    print(f'theta vrai = {tt:+7.2f} | prédit = {tp:+7.2f}')

