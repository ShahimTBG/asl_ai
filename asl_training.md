```python
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
```


```python
data_dir = "/media/marksman/2C32-CEEB/asl-ai/asl_alphabet_train"
```


```python
classes = os.listdir(data_dir)
```


```python
print("Number of classes:", len(classes))
```

    Number of classes: 29



```python
print("Some class names:", classes[:5])
```

    Some class names: ['A', 'B', 'C', 'D', 'del']



```python
datagen = ImageDataGenerator(rescale=1.0/255)
```


```python
train_data = datagen.flow_from_directory(
    data_dir,
    target_size=(64,64),
    batch_size=32,
    class_mode='categorical'
)
```

    Found 87000 images belonging to 29 classes.



```python
print(train_data.class_indices)
```

    {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'J': 9, 'K': 10, 'L': 11, 'M': 12, 'N': 13, 'O': 14, 'P': 15, 'Q': 16, 'R': 17, 'S': 18, 'T': 19, 'U': 20, 'V': 21, 'W': 22, 'X': 23, 'Y': 24, 'Z': 25, 'del': 26, 'nothing': 27, 'space': 28}



```python
from tensorflow.keras import layers, models
model = models.Sequential()
```


```python
model.add(layers.Input(shape=(64, 64, 3)))

# First conv block 
model.add(layers.Conv2D(32,(3,3), activation ='relu', input_shape=(64,64,3)))
model.add(layers.MaxPooling2D((2,2)))

# Second conv block
model.add(layers.Conv2D(64,(3,3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))

# Third conv block
model.add(layers.Conv2D(128,(3,3), activation='relu'))
model.add(layers.MaxPooling2D(2,2))

# Now we flatten to 1D for fully connected layers
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(29, activation='softmax'))

print("we just dubbed without errors")
```

    we just dubbed without errors


    /home/marksman/asl-env/lib/python3.11/site-packages/keras/src/layers/convolutional/base_conv.py:113: UserWarning: Do not pass an `input_shape`/`input_dim` argument to a layer. When using Sequential models, prefer using an `Input(shape)` object as the first layer in the model instead.
      super().__init__(activity_regularizer=activity_regularizer, **kwargs)



```python
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
```


```python
model.fit(train_data, epochs=10)
```

    /home/marksman/asl-env/lib/python3.11/site-packages/keras/src/trainers/data_adapters/py_dataset_adapter.py:121: UserWarning: Your `PyDataset` class should call `super().__init__(**kwargs)` in its constructor. `**kwargs` can include `workers`, `use_multiprocessing`, `max_queue_size`. Do not pass these arguments to `fit()`, as they will be ignored.
      self._warn_if_super_not_called()


    Epoch 1/10
    [1m  64/2719[0m [37m━━━━━━━━━━━━━━━━━━━━[0m [1m8:37[0m 195ms/step - accuracy: 0.0385 - loss: 3.3671
