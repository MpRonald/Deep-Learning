# -*- coding: utf-8 -*-

# -- Sheet --

import tensorflow
from tensorflow import keras

dataset = keras.datasets.fashion_mnist
((imagens_treino, identificacoes_treino),
 (imagens_test, identificacoes_test)) = dataset.load_data()

imagens_treino

len(imagens_treino)

imagens_treino.shape

len(imagens_test)

imagens_test.shape

import matplotlib.pyplot as plt
%matplotlib inline

plt.imshow(imagens_treino[0])
plt.title(identificacoes_treino[0])

plt.imshow(imagens_treino[1])

plt.imshow(imagens_treino[2])

print('Menor identificador: ', identificacoes_treino.min())
print('Maior identificador: ', identificacoes_treino.max())

for imagem in range(10):
    plt.subplot(2, 5, imagem+1)
    plt.imshow(imagens_treino[imagem])
    plt.title(identificacoes_treino[imagem])

nomes_de_classificacoes = [
    'Camiseta', 'Calca', 'Pullover', 'Vestido', 'Casaco', 'Sandalia',
    'Camisa', 'Tenis', 'Bolsa', 'Bota'
]
for imagem in range(10):
    plt.subplot(2, 5, imagem+1)
    plt.imshow(imagens_treino[imagem])
    plt.title(nomes_de_classificacoes[identificacoes_treino[imagem]])

modelo = keras.Sequential([
    keras.layers.Flatten(input_shape = (28, 28)),
    keras.layers.Dense(256, activation = tensorflow.nn.relu),
    keras.layers.Dense(128, activation = tensorflow.nn.relu),
    keras.layers.Dense(64, activation = tensorflow.nn.relu),
    keras.layers.Dense(10, activation = tensorflow.nn.softmax)
])

modelo.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy',
               metrics = ['accuracy'])
historico = modelo.fit(imagens_treino, identificacoes_treino, epochs = 5,
           validation_split = 0.2)

plt.plot(historico.history['accuracy'])
plt.plot(historico.history['val_accuracy'])
plt.title('Acurácia por épocas')
plt.xlabel('Épocas')
plt.ylabel('Acurácia')
plt.legend(['Treino', 'Validação'])

plt.plot(historico.history['loss'])
plt.plot(historico.history['val_loss'])
plt.title('Acurácia por épocas')
plt.xlabel('Épocas')
plt.ylabel('Perdas')
plt.legend(['Treino', 'Validação'])

total_de_classificadores = 10
nomes_de_classificacoes = [
    'Camiseta', 'Calca', 'Pullover', 'Vestido', 'Casaco', 'Sandalia',
    'Camisa', 'Tenis', 'Bolsa', 'Bota'
]

plt.imshow(imagens_treino[0])
plt.colorbar()

teste = modelo.predict(imagens_test)

import numpy as np
print('Resultado teste', np.argmax(teste[0]))
print('Número da imagem de teste: ', identificacoes_test[0])

for i in range(1,5):
    print('Resultado teste', np.argmax(teste[i]))
    print('Número da imagem de teste: ', identificacoes_test[i])
    print('\n')

for i in range(50,55):
    print('Resultado teste', np.argmax(teste[i]))
    print('Número da imagem de teste: ', identificacoes_test[i])
    print('\n')

perda_test, acuracia_test = modelo.evaluate(imagens_test, identificacoes_test)
print('Valor da perda no teste: %.4f ' % perda_test)
print('Valor da acuracia no teste: %.4f' % acuracia_test)

# alterando a quantidade de época de 5 para 10
modelo = keras.Sequential([
    keras.layers.Flatten(input_shape = (28, 28)),
    keras.layers.Dense(256, activation = tensorflow.nn.relu),
    keras.layers.Dense(128, activation = tensorflow.nn.relu),
    keras.layers.Dense(64, activation = tensorflow.nn.relu),
    keras.layers.Dense(10, activation = tensorflow.nn.softmax)
])

modelo.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy',
               metrics = ['accuracy'])
historico = modelo.fit(imagens_treino, identificacoes_treino, epochs = 10,
           validation_split = 0.2)

plt.plot(historico.history['accuracy'])
plt.plot(historico.history['val_accuracy'])
plt.title('Acurácia por épocas')
plt.xlabel('Épocas')
plt.ylabel('Acurácia')
plt.legend(['Treino', 'Validação'])

plt.plot(historico.history['loss'])
plt.plot(historico.history['val_loss'])
plt.title('Acurácia por épocas')
plt.xlabel('Épocas')
plt.ylabel('Perdas')
plt.legend(['Treino', 'Validação'])

# realizando mais um teste de normalização
modelo = keras.Sequential([
    keras.layers.Flatten(input_shape = (28, 28)),
    keras.layers.Dense(256, activation = tensorflow.nn.relu),
    keras.layers.Dense(128, activation = tensorflow.nn.relu),
    keras.layers.Dense(64, activation = tensorflow.nn.relu),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(10, activation = tensorflow.nn.softmax)
])

modelo.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy',
               metrics = ['accuracy'])
historico = modelo.fit(imagens_treino, identificacoes_treino, epochs = 5,
           validation_split = 0.2)

plt.plot(historico.history['accuracy'])
plt.plot(historico.history['val_accuracy'])
plt.title('Acurácia por épocas')
plt.xlabel('Épocas')
plt.ylabel('Acurácia')
plt.legend(['Treino', 'Validação'])

plt.plot(historico.history['loss'])
plt.plot(historico.history['val_loss'])
plt.title('Acurácia por épocas')
plt.xlabel('Épocas')
plt.ylabel('Perdas')
plt.legend(['Treino', 'Validação'])

# ### podemos ver que o segundo modelo sai do overfiting


# salvando modelo
from tensorflow.keras.models import load_model

modelo.save('modelo.h5')
modelo_salvo = load_model('modelo.h5')

testes = modelo.predict(imagens_test)
testes_modelo_salvo = modelo_salvo.predict(imagens_test)
for i in range(1,5):
    print('Resultado teste: ', np.argmax(testes[i]))
    print('Número da imagem de teste: ', identificacoes_test[i])
    print('Resultado teste MODELO SALVO: ', np.argmax(testes_modelo_salvo[i]))
    print('Número da imagem de teste: ', identificacoes_test[i])
    print('\n')



