from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras import utils
from tensorflow.keras.preprocessing import image

import io
import requests
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
%matplotlib inline 

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

classes = ['футболка', 'брюки', 'свитер', 'платье', 'пальто', 'туфли', 'рубашка', 'кроссовки', 'сумка', 'ботинки']

plt.figure(figsize=(10,10))
for i in range(100,150):
    plt.subplot(5,10,i-100+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(x_train[i], cmap=plt.cm.binary)
    plt.xlabel(classes[y_train[i]])

x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)

# Âåêòîðèçîâàííûå îïåðàöèè
# Ïðèìåíÿþòñÿ ê êàæäîìó ýëåìåíòó ìàññèâà îòäåëüíî
x_train = x_train / 255 
x_test = x_test / 255 

n = 0

print(y_train[n])

y_train = utils.to_categorical(y_train, 10)

y_test = utils.to_categorical(y_test, 10)

print(y_train[n])

# Ñîçäàåì ïîñëåäîâàòåëüíóþ ìîäåëü
model = Sequential()
# Âõîäíîé ïîëíîñâÿçíûé ñëîé, 800 íåéðîíîâ, 784 âõîäà â êàæäûé íåéðîí
model.add(Dense(800, input_dim=784, activation="relu"))
# Âûõîäíîé ïîëíîñâÿçíûé ñëîé, 10 íåéðîíîâ (ïî êîëè÷åñòâó ðóêîïèñíûõ öèôð)
model.add(Dense(10, activation="softmax"))

model.compile(loss="categorical_crossentropy", optimizer="SGD", metrics=["accuracy"])

print(model.summary())

history = model.fit(x_train, y_train, 
                    batch_size=200, 
                    epochs=100,
                    validation_split=0.2,
                    verbose=1)

model.save('fashion_mnist_dense.h5')

scores = model.evaluate(x_test, y_test, verbose=1)

print("Äîëÿ âåðíûõ îòâåòîâ íà òåñòîâûõ äàííûõ, â ïðîöåíòàõ:", round(scores[1] * 100, 4))

n_rec = 496

plt.imshow(x_test[n_rec].reshape(28, 28), cmap=plt.cm.binary)
plt.show()

x = x_test[n_rec]
x = np.expand_dims(x, axis=0)

prediction = model.predict(x)

prediction = np.argmax(prediction[0])
print("Íîìåð êëàññà:", prediction)
print("Íàçâàíèå êëàññà:", classes[prediction])

label = np.argmax(y_test[0])
print("Íîìåð êëàññà:", label)
print("Íàçâàíèå êëàññà:", classes[label])

#ñîçäàåì ôóíêöèþ çàãðóçêè ôàéëà

def load_image():
    uploaded_file = st.file_uploader(label='Âûáåðèòå èçîáðàæåíèå äëÿ ðàñïîçíàâàíèÿ')
    if uploaded_file is not None:
        image_data = uploaded_file.getvalue()
        st.image(image_data)
        return Image.open(io.BytesIO(image_data))
    else:
        return None

def preprocess_image(img):
    img = image.load_img(img_path, target_size=(28, 28), color_mode = "grayscale")
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x


st.title('Ðàñïîçíàâàíèå îäåæäû íà ôîòî')
img = load_image()
result = st.button('Ðàñïîçíàòü èçîáðàæåíèå')
if result:
    x = preprocess_image(img)
    preds = model.predict(x)
    st.write('**Ðåçóëüòàòû ðàñïîçíàâàíèÿ:**')
    print_predictions(preds)
