from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
from keras.applications.vgg16 import VGG16

import pandas as pd

print('loading model')
model = VGG16()
print('done')

df = pd.read_csv('train.csv')
imgs_train = df.IMAGE_NAME
city_train = df.CITY

df = pd.read_csv('test.csv')
imgs_test = df.IMAGE_NAME
city_test = df.CITY

l2i = {}
i2l = []
train_features = []

for i in range(len(imgs_train)):
    print(i, len(i2l))
    file = "{}_IMG/{}".format(city_train[i], imgs_train[i])
    image = load_img(file, target_size=(224, 224))
    image = img_to_array(image)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    image = preprocess_input(image)
    yhat = model.predict(image)
    label = decode_predictions(yhat)
    for item in label[0]:
        try:
            l2i[item[1]]
        except:
            l2i[item[1]] = len(i2l)
            i2l.append(item[1])
    train_features.append(label[0])



test_features = []
for i in range(len(imgs_test)):
    print(i, len(i2l))
    file = "{}_IMG/{}".format(city_test[i], imgs_test[i])
    image = load_img(file, target_size=(224, 224))
    image = img_to_array(image)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    image = preprocess_input(image)
    yhat = model.predict(image)
    label = decode_predictions(yhat)
    for item in label[0]:
        try:
            l2i[item[1]]
        except:
            l2i[item[1]] = len(i2l)
            i2l.append(item[1])
    test_features.append(label[0])

train = []
for i in range(len(imgs_train)):
    container = [0] * len(i2l)
    for item in train_features[i]:
        container[l2i[item[1]]] = item[2] * 100
    train.append(container)


test = []
for i in range(len(imgs_test)):
    container = [0] * len(i2l)
    for item in test_features[i]:
        container[l2i[item[1]]] = item[2] * 100
    test.append(container)


df = pd.DataFrame(train, index = imgs_train, columns = i2l)
df.to_csv('keras_train.csv')

df = pd.DataFrame(test, index = imgs_test, columns = i2l)
df.to_csv('keras_test.csv')

    


    
