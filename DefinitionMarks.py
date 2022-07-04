!unzip -q "/content/drive/My Drive/neural/middle_fmr.zip" -d /content/cars # Указываем путь к базе в Google Drive

train_path = '/content/cars' #Папка с папками картинок, рассортированных по категориям
batch_size = 25 #Размер выборки
img_width = 96 #Ширина изображения
img_height = 54 #Высота изображения

datagen = ImageDataGenerator(
    rescale=1. / 255, #Значения цвета меняем на дробные показания
    rotation_range=10, #Поворачиваем изображения при генерации выборки
    width_shift_range=0.1, #Двигаем изображения по ширине при генерации выборки
    height_shift_range=0.1, #Двигаем изображения по высоте при генерации выборки
    zoom_range=0.1, #Зумируем изображения при генерации выборки
    horizontal_flip=True, #Включаем отзеркаливание изображений
    fill_mode='nearest', #Заполнение пикселей вне границ ввода
    validation_split=0.2 #Указываем разделение изображений на обучающую и тестовую выборку
)
train_generator = datagen.flow_from_directory(
    train_path, #Путь ко всей выборке выборке
    target_size=(img_width, img_height), #Размер изображений
    batch_size=batch_size, #Размер batch_size
    class_mode='categorical', #Категориальный тип выборки. Разбиение выборки по маркам авто
    shuffle=True, #Перемешивание выборки
    subset='training' # устанавливаем как набор для обучения
)
validation_generator = datagen.flow_from_directory(
    train_path, #Путь ко всей выборке выборке
    target_size=(img_width, img_height), #Размер изображений
    batch_size=batch_size, #Размер batch_size
    class_mode='categorical', #Категориальный тип выборки. Разбиение выборки по маркам авто
    shuffle=True, #Перемешивание выборки
    subset='validation' # устанавливаем как валидационный набор
)
model = Sequential()
model.add(Conv2D(256, (3, 3), padding='same', activation='relu', input_shape=(img_width, img_height, 3)))
# 1
model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
# 2
model.add(MaxPooling2D(pool_size=(3, 3)))

model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
# 3
model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
# 4
model.add(MaxPooling2D(pool_size=(3, 3)))

model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
# 5
model.add(Dropout(0.3))

model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
# 6
model.add(MaxPooling2D(pool_size=(3, 3)))

model.add(Dropout(0.3))

model.add(Flatten())

model.add(Dense(2048, activation='relu'))

model.add(Dense(4096, activation='relu'))

model.add(Dense(len(train_generator.class_indices), activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.00001), metrics=['accuracy'])

history= model.fit_generator(train_generator, steps_per_epoch=train_generator.samples//batch_size,
                             validation_data= validation_generator, validation_steps= validation_generator.samples//batch_size,
                             epochs=50, verbose=1)


#Дообучение модели

model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.00001), metrics=['accuracy'])
history = model.fit_generator(
    train_generator,
    steps_per_epoch = train_generator.samples // batch_size,
    validation_data = validation_generator,
    validation_steps = validation_generator.samples // batch_size,
    epochs=50,
    verbose=1
)

#дообучение модели
model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.00005), metrics=['accuracy'])
history = model.fit_generator(
    train_generator,
    steps_per_epoch = train_generator.samples // batch_size,
    validation_data = validation_generator,
    validation_steps = validation_generator.samples // batch_size,
    epochs=50,
    verbose=1
)
#Дообучение модели
model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.00008), metrics=['accuracy'])
history = model.fit_generator(
    train_generator,
    steps_per_epoch = train_generator.samples // batch_size,
    validation_data = validation_generator,
    validation_steps = validation_generator.samples // batch_size,
    epochs=50,
    verbose=1
)
#Дообучение модели
model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.00005), metrics=['accuracy'])
history = model.fit_generator(
    train_generator,
    steps_per_epoch = train_generator.samples // batch_size,
    validation_data = validation_generator,
    validation_steps = validation_generator.samples // batch_size,
    epochs=50,
    verbose=1
)

#Дообучение модели
model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.00002), metrics=['accuracy'])
history = model.fit_generator(
    train_generator,
    steps_per_epoch = train_generator.samples // batch_size,
    validation_data = validation_generator,
    validation_steps = validation_generator.samples // batch_size,
    epochs=50,
    verbose=1
)
#Дообучение модели
model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.00004), metrics=['accuracy'])
history = model.fit_generator(
    train_generator,
    steps_per_epoch = train_generator.samples // batch_size,
    validation_data = validation_generator,
    validation_steps = validation_generator.samples // batch_size,
    epochs=50,
    verbose=1
)