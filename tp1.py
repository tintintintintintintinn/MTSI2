import numpy as np
import tensorflow
from tensorflow import keras
from tensorflow.keras import layers

SEED_1 = 42
SEED_2 = 43

print(tensorflow.config.list_physical_devices('GPU'))


def stats(image):
    def rgb_stats(img):
        return np.concatenate([img.mean(axis=(0, 1)), img.std(axis=(0, 1))])

    def to_gray(img):
        return img.mean(axis=-1)

    def gray_stats(img):
        gray_img = to_gray(img) if img.ndim == 3 else img
        return np.array([gray_img.mean(), gray_img.std()])

    def rgb_distribution(img, n_bands=10):
        n_channels = img.shape[-1]
        histogram = np.empty(n_channels * n_bands, dtype=np.float64)
        for channel in range(n_channels):
            counts, _ = np.histogram(img[:, :, channel], bins=n_bands, range=(0, 255))
            histogram[channel * n_bands:(channel + 1) * n_bands] = counts
        return histogram

    def energy_stats(img):
        magnitude = np.abs(np.fft.fftshift(np.fft.fft2(to_gray(img))))
        h, w = magnitude.shape
        y, x = np.indices((h, w))
        r = np.sqrt((x - w / 2) ** 2 + (y - h / 2) ** 2)
        band_edges = np.linspace(0, r.max(), 4)
        energies = np.empty(3, dtype=np.float64)
        for band in range(3):
            in_band = (r >= band_edges[band]) & (r < band_edges[band + 1])
            energies[band] = np.sum(magnitude[in_band] ** 2)
        return energies

    def divide_img(img):
        h, w = img.shape[:2]
        hh, hw = h // 2, w // 2
        return img[:hh, :], img[hh:, :], img[:, :hw], img[:, hw:]

    img_up, img_down, img_left, img_right = divide_img(image)
    gray = to_gray(image)

    return np.concatenate([
        rgb_distribution(image), rgb_stats(image),
        rgb_distribution(img_up), rgb_stats(img_up),
        rgb_distribution(img_down), rgb_stats(img_down),
        rgb_distribution(img_left), rgb_stats(img_left),
        rgb_distribution(img_right), rgb_stats(img_right),
        gray_stats(gray),
        gray_stats(img_up), gray_stats(img_down),
        gray_stats(img_left), gray_stats(img_right),
        energy_stats(image),
    ])

ernest_celestine_01 = np.load("Movies_npy/ernest_celestine_01.npy")
ernest_celestine_02 = np.load("Movies_npy/ernest_celestine_02.npy")
toy_story_1_01 = np.load("Movies_npy/toy_story_1_01.npy")
toy_story_1_02 = np.load("Movies_npy/toy_story_1_02.npy")

stats_ec_01 = stats(ernest_celestine_01)
stats_ec_02 = stats(ernest_celestine_02)
stats_ts1_01 = stats(toy_story_1_01)
stats_ts1_02 = stats(toy_story_1_02)

print(stats_ec_01.shape)

# binary

X_train = np.concatenate([stats_ec_01, stats_ts1_01,] ,axis=0)
X_test = np.concatenate([stats_ec_02, stats_ts1_02,] ,axis=0)
y_train = np.array([0]*ernest_celestine_01.shape[0] + [1]*toy_story_1_01.shape[0])
y_test = np.array([0]*ernest_celestine_02.shape[0] + [1]*toy_story_1_02.shape[0])

permutation = np.random.default_rng(SEED_1).permutation(X_train.shape[0])
X_train = X_train[permutation]
y_train = y_train[permutation]

X_train, X_val = X_train[:250], X_train[250:]
y_train, y_val = y_train[:250], y_train[250:]

X_train_flatten = X_train.reshape(X_train.shape[0], -1)
X_val_flatten = X_val.reshape(X_val.shape[0], -1)
X_test_flatten = X_test.reshape(X_test.shape[0], -1)

X_train_flatten = X_train_flatten.astype('float32') / 255.0
X_val_flatten = X_val_flatten.astype('float32') / 255.0
X_test_flatten = X_test_flatten.astype('float32') / 255.0

HEIGHT = X_train[0].shape[0]
WIDTH = X_train[0].shape[1]
COLORS = 3

model = keras.Sequential([
    layers.Dense(1024, activation="relu", input_shape=(HEIGHT*WIDTH*COLORS,)),
    layers.Dense(512, activation="relu"),
    layers.Dense(1, activation="sigmoid")
])

model.compile(optimizer="rmsprop", loss="binary_crossentropy", metrics=["accuracy"])

with tensorflow.device('/GPU:0'):
    history = model.fit(X_train_flatten, y_train, epochs=35, batch_size=32, validation_data=(X_val_flatten, y_val))

# Epoch 35/35
# 8/8 ━━━━━━━━━━━━━━━━━━━━ 1s 70ms/step - accuracy: 0.8992 - loss: 0.0271 - val_accuracy: 0.8040 - val_loss: 0.0772

model.evaluate(X_test_flatten, y_test)

# 32/32 ━━━━━━━━━━━━━━━━━━━━ 1s 17ms/step - accuracy: 0.8608 - loss: 0.809

# multi-class classification

toy_story_2_01 = np.load("Movies_npy/toy_story_2_01.npy")
toy_story_2_02 = np.load("Movies_npy/toy_story_2_02.npy")
toy_story_3_01 = np.load("Movies_npy/toy_story_3_01.npy")
toy_story_3_02 = np.load("Movies_npy/toy_story_3_02.npy")

stats_ts2_01 = stats(toy_story_2_01)
stats_ts2_02 = stats(toy_story_2_02)
stats_ts3_01 = stats(toy_story_3_01)
stats_ts3_02 = stats(toy_story_3_02)

X_train = np.concatenate([stats_ec_01, stats_ts1_01, stats_ts2_01,stats_ts3_01,] ,axis=0)
X_test = np.concatenate([stats_ec_02, stats_ts1_02, stats_ts2_02,stats_ts3_02,] ,axis=0)
train_labels = np.concatenate([[0]*ernest_celestine_01.shape[0], [1]*toy_story_1_01.shape[0],
                          [2]*toy_story_2_01.shape[0], [3]*toy_story_3_01.shape[0]], axis=0)
test_labels = np.concatenate([[0]*ernest_celestine_02.shape[0], [1]*toy_story_1_02.shape[0],
                         [2]*toy_story_2_02.shape[0], [3]*toy_story_3_02.shape[0]], axis=0)

def to_one_hot(labels: np.ndarray, dim:int=4) -> np.ndarray:
    results = np.zeros((len(labels), dim))
    for i, label in enumerate(labels):
        results[i, label] = 1
    return results

y_train = to_one_hot(train_labels)
y_test = to_one_hot(test_labels)

permutation = np.random.default_rng(SEED_2).permutation(X_train.shape[0])
X_train = X_train[permutation]
y_train = y_train[permutation]

X_train, X_val = X_train[:1000], X_train[1000:]
y_train, y_val = y_train[:1000], y_train[1000:]

X_train_flatten = X_train.reshape(X_train.shape[0], -1)
X_val_flatten = X_val.reshape(X_val.shape[0], -1)
X_test_flatten = X_test.reshape(X_test.shape[0], -1)

X_train_flatten = X_train_flatten.astype('float32') / 255.0
X_val_flatten = X_val_flatten.astype('float32') / 255.0
X_test_flatten = X_test_flatten.astype('float32') / 255.0

HEIGHT = X_train[0].shape[0]
WIDTH = X_train[0].shape[1]
COLORS = 3

model = keras.Sequential([
    layers.Dense(1024, activation="relu", input_shape=(HEIGHT*WIDTH*COLORS,)),
    layers.Dense(512, activation="relu"),
    layers.Dense(4, activation="softmax")
])

model.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"])

with tensorflow.device('/GPU:0'):
    history = model.fit(X_train_flatten, y_train, epochs=30, batch_size=32, validation_data=(X_val_flatten, y_val))

# Epoch 30/30
# 16/16 ━━━━━━━━━━━━━━━━━━━━ 1s 85ms/step - accuracy: 0.7456 - loss: 0.4562 - val_accuracy: 0.0.8730 - val_loss: 0.3448

model.evaluate(X_test_flatten, y_test)

# 63/63 ━━━━━━━━━━━━━━━━━━━━ 1s 13ms/step - accuracy: 0.5837 - loss: 1.3490

# résultats moyens du fait que trois films sont quasiment indifférentiables (TS1, TS2 et TS3)