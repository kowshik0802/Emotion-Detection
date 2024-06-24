from keras.applications import MobileNet
from keras.models import Sequential, Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler
import os

def create_mobileNet_model(input_shape, num_classes):
    base_model = MobileNet(weights='imagenet', include_top=False, input_shape=input_shape)

    for layer in base_model.layers:
        layer.trainable = True

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dense(1024, activation='relu')(x)
    x = Dense(512, activation='relu')(x)
    predictions = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    return model

img_rows, img_cols = 224, 224
num_classes = 5
input_shape = (img_rows, img_cols, 3)

model = create_mobileNet_model(input_shape, num_classes)
model.summary()

train_data_dir = 'Deep Learning Stuff/Emotion Classification/fer2013/train'

validation_data_dir = 'Deep Learning Stuff/Emotion Classification/fer2013/validation'

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.3,
    height_shift_range=0.3,
    horizontal_flip=True,
    fill_mode='nearest'
)

validation_datagen = ImageDataGenerator(rescale=1./255)

batch_size = 32

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_rows, img_cols),
    batch_size=batch_size,
    class_mode='categorical'
)

validation_generator = validation_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_rows, img_cols),
    batch_size=batch_size,
    class_mode='categorical'
)

checkpoint = ModelCheckpoint(
    'emotion_face_mobileNet.h5',
    monitor='val_loss',
    mode='min',
    save_best_only=True,
    verbose=1
)

earlystop = EarlyStopping(
    monitor='val_loss',
    min_delta=0,
    patience=10,
    verbose=1,
    restore_best_weights=True
)

def lr_schedule(epoch):
    """Learning Rate Schedule

    Learning rate is scheduled to be reduced after 5, 10, 15, 20 epochs.
    Adjust the learning rate to fit your needs.
    """
    lr = 0.001
    if epoch > 20:
        lr *= 0.2
    elif epoch > 15:
        lr *= 0.2
    elif epoch > 10:
        lr *= 0.2
    elif epoch > 5:
        lr *= 0.2
    return lr

lr_scheduler = LearningRateScheduler(lr_schedule)

callbacks = [earlystop, checkpoint, lr_scheduler]

model.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr=0.001),
              metrics=['accuracy']
)

nb_train_samples = 24176
nb_validation_samples = 3006

epochs = 25

history = model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples//batch_size,
    epochs=epochs,
    callbacks=callbacks,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples//batch_size
)
