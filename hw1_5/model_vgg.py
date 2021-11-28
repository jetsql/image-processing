import tensorflow as tf
import numpy as np
import time
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization

##注意我是在colab上面執行，所以在本地端執行這個code會有問題，這個檔案是複製colab中的程式碼，請去執行另一個名為vgg.py的檔案

model_path = 'VGG16.h5'

cifar10 = tf.keras.datasets.cifar10
((x_train, y_train), (x_test, y_test)) = cifar10.load_data()

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

for i in range(9):
    plt.subplot(3,3,i+1)
    plt.imshow(x_train[i])
    plt.title(class_names[int(y_train[i])])
plt.show()


x_train_n, x_test_n = x_train / 255.0, x_test / 255.0



#hyperparameters
batch_size = 32
learning_rate = 0.001
optimizer = 'Adam'
epochs = 20


model = tf.keras.models.Sequential([
    Conv2D(64, (3, 3), input_shape=(32, 32, 3), padding='same', activation='relu'), # conv1_1
    Conv2D(64, (3, 3), padding='same', activation='relu'), # conv1_2
    MaxPooling2D(pool_size=(2, 2), strides=2), # maxpool1
    BatchNormalization(),
    Conv2D(128, (3, 3), padding='same', activation='relu'), # conv2_1
    Conv2D(128, (3, 3), padding='same', activation='relu'), # conv2_2
    MaxPooling2D(pool_size=(2, 2), strides=2), # maxpool2
    BatchNormalization(),
    Conv2D(256, (3, 3), padding='same', activation='relu'), # conv3_1
    Conv2D(256, (3, 3), padding='same', activation='relu'), # conv3_2
    Conv2D(256, (3, 3), padding='same', activation='relu'), # conv3_3
    MaxPooling2D(pool_size=(2, 2), strides=2), # maxpool3
    BatchNormalization(),
    Conv2D(512, (3, 3), padding='same', activation='relu'), # conv4_1
    Conv2D(512, (3, 3), padding='same', activation='relu'), # conv4_2
    Conv2D(512, (3, 3), padding='same', activation='relu'), # conv4_3
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2), strides=2), # maxpool4
    Conv2D(512, (3, 3), padding='same', activation='relu'), # conv5_1
    Conv2D(512, (3, 3), padding='same', activation='relu'), # conv5_2
    Conv2D(512, (3, 3), padding='same', activation='relu'), # conv5_3
    MaxPooling2D(pool_size=(2, 2), strides=2), # maxpool5
    BatchNormalization(),
    Flatten(), # flatten
    Dense(4096, activation='relu'), # fc1
    Dense(4096, activation='relu'), # fc2
    Dense(10, activation='softmax'), # output (with softmax) 
])

model.summary()


if optimizer.upper() == 'ADAM':
    opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)



model.compile(
    optimizer=opt,
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False), # from_logits=False: output layer is already softmax
    metrics=['accuracy']
)


history_training = model.fit(x_train_n, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test_n, y_test))


model.save_weights(model_path)

training_accuracy = np.array(history_training.history['accuracy']) * 100
training_loss = history_training.history['loss']
testing_accuracy = np.array(history_training.history['val_accuracy']) * 100

plt.figure('Accuracy and Loss')
plt.subplot(2, 1, 1)
plt.title('Accuracy')
plt.plot(training_accuracy, label='Training')
plt.plot(testing_accuracy, label = 'Testing')
plt.ylabel('%')
plt.legend(loc='lower right')
plt.subplot(2, 1, 2)
plt.plot(training_loss)
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()
t = time.localtime()
plt.savefig('/' + time.strftime("%Y%m%d_%H%M%S", t) + '.png')