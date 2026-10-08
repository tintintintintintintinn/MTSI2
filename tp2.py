import numpy as np
import tensorflow
from tensorflow import keras
from tensorflow.keras import layers

print(tensorflow.config.list_physical_devices('GPU'))

SEED_1 = 42
SEED_2 = 43

X_train_ernest = np.load("Movies_npy/ernest_celestine_01.npy")
X_train_toy = np.load("Movies_npy/toy_story_1_01.npy")
X_train = np.concatenate([X_train_ernest, X_train_toy], axis=0)
y_train = np.array([0]*500 + [1]*500)

permutation = np.random.default_rng(SEED_1).permutation(X_train.shape[0])
X_train = X_train[permutation]
y_train = y_train[permutation]

TRAIN_RATIO = 0.5
split = int(TRAIN_RATIO * len(X_train))
X_train, X_val = X_train[:split], X_train[split:]
y_train, y_val = y_train[:split], y_train[split:]

X_test_ernest = np.load("Movies_npy/ernest_celestine_02.npy")
X_test_toy = np.load("Movies_npy/toy_story_1_02.npy")
X_test = np.concatenate([X_test_ernest, X_test_toy], axis=0)  # Shape: (1000, 540, 920)
y_test = np.array([0]*500 + [1]*500)

X_train_flatten = X_train.astype('float32') / 255.0
X_val_flatten = X_val.astype('float32') / 255.0
X_test_flatten = X_test.astype('float32') / 255.0

inputs = keras.Input(shape=X_train.shape[1:])
x = layers.Conv2D(filters=32, kernel_size=3, activation="relu")(inputs)
x = layers.MaxPooling2D(pool_size=2)(x)
x = layers.Conv2D(filters=64, kernel_size=3, activation="relu")(x)
x = layers.MaxPooling2D(pool_size=2)(x)
x = layers.Conv2D(filters=128, kernel_size=3, activation="relu")(x)
x = layers.MaxPooling2D(pool_size=2)(x)
x = layers.Flatten()(x)
outputs = layers.Dense(1, activation="sigmoid")(x)
model = keras.Model(inputs=inputs, outputs=outputs)
model.compile(optimizer="rmsprop", loss="binary_crossentropy", metrics=["accuracy"])

BATCH_SIZE = 32
EPOCHS = 12

with tensorflow.device('/GPU:0'):
    history = model.fit(
        X_train_flatten, y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(X_val_flatten, y_val),
    )

#Epoch 12/12
#16/16 ━━━━━━━━━━━━━━━━━━━━ 2s 106ms/step - accuracy: 0.9927 - loss: 0.0167 - val_accuracy: 0.9680 - val_loss: 0.0838

model.evaluate(X_test_flatten, y_test)

#32/32 ━━━━━━━━━━━━━━━━━━━━ 1s 25ms/step - accuracy: 0.9498 - loss: 0.1324
#Out[26]: [0.26703932881355286, 0.9089999794960022]

# confusion matrix

y_prob = model.predict(X_test_flatten)
y_pred = (y_prob.ravel() > 0.5).astype(int)

from sklearn.metrics import confusion_matrix

conf_matrix = confusion_matrix(y_test, y_pred)
print(conf_matrix)

# [[479  21]
#  [ 70 430]]

# multi-classes

X_train_ts2 = np.load("Movies_npy/toy_story_2_01.npy")
X_test_ts2 = np.load("Movies_npy/toy_story_2_02.npy")
X_train_ts3 = np.load("Movies_npy/toy_story_3_01.npy")
X_test_ts3 = np.load("Movies_npy/toy_story_3_02.npy")

X_train = np.concatenate([X_train_ernest, X_train_toy, X_train_ts2, X_train_ts3], axis=0)
X_test = np.concatenate([X_test_ernest, X_test_toy, X_test_ts2, X_test_ts3])
train_labels = np.array([0]*X_train_ernest.shape[0]
                        + [1]*X_train_toy.shape[0]
                        + [2]*X_train_ts2.shape[0]
                        + [3]*X_train_ts3.shape[0])
test_labels = np.array([0]*X_test_ernest.shape[0]
                       + [1]*X_test_toy.shape[0]
                       + [2]*X_test_ts2.shape[0]
                       + [3]*X_test_ts3.shape[0])

NUM_CLASSES = len(np.unique(train_labels))

def to_one_hot(labels: np.ndarray, dim: int) -> np.ndarray:
    results = np.zeros((len(labels), dim))
    for i, label in enumerate(labels):
        results[i, label] = 1
    return results

y_train = to_one_hot(train_labels, NUM_CLASSES)
y_test = to_one_hot(test_labels, NUM_CLASSES)

permutation = np.random.default_rng(SEED_2).permutation(X_train.shape[0])
X_train = X_train[permutation]
y_train = y_train[permutation]

TRAIN_RATIO = 0.8
split = int(TRAIN_RATIO * len(X_train))
X_train, X_val = X_train[:split], X_train[split:]
y_train, y_val = y_train[:split], y_train[split:]

X_train_flatten = X_train.astype('float32') / 255.0
X_val_flatten = X_val.astype('float32') / 255.0
X_test_flatten = X_test.astype('float32') / 255.0

BATCH_SIZE = 32
EPOCHS = 9

inputs = keras.Input(shape=X_train.shape[1:])
x = layers.Conv2D(filters=32, kernel_size=3, activation="relu")(inputs)
x = layers.MaxPooling2D(pool_size=2)(x)
x = layers.Conv2D(filters=64, kernel_size=3, activation="relu")(x)
x = layers.MaxPooling2D(pool_size=2)(x)
x = layers.Conv2D(filters=128, kernel_size=3, activation="relu")(x)
x = layers.MaxPooling2D(pool_size=2)(x)
x = layers.Flatten()(x)
outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)
model = keras.Model(inputs=inputs, outputs=outputs)
model.compile(
    optimizer="rmsprop",
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

history = model.fit(
    X_train_flatten, y_train,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=(X_val_flatten, y_val),
)

# Epoch 9/9
# 50/50 ━━━━━━━━━━━━━━━━━━━━ 4s 88ms/step - accuracy: 0.9641 - loss: 0.1125 - val_accuracy: 0.8200 - val_loss: 0.5780

test_loss, test_accuracy = model.evaluate(X_test_flatten, y_test)

# 63/63 ━━━━━━━━━━━━━━━━━━━━ 2s 23ms/step - accuracy: 0.6689 - loss: 1.5579

y_prob = model.predict(X_test_flatten)
y_pred = np.argmax(y_prob, axis=1)

conf_matrix = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)
for true_label, predicted_label in zip(test_labels, y_pred):
    conf_matrix[true_label, predicted_label] += 1
print(conf_matrix)

# [[465  22   3  10]
#  [ 18  81 163 238]
#  [ 10  69 216 205]
#  [  3 210  69 218]]

# difficulté à reconnaitre les toy story entre eux; toy story 3 sur-prédit (car 10+238+205+218 > 500)
# de plus, toy story 1 et 2 difficiles à reconnaître
# par contre, ernest et celestine bien reconnu