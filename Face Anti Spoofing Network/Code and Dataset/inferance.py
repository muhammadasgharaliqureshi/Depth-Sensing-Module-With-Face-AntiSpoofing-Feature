import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
import time
#gpus = tf.config.experimental.list_physical_devices('GPU')
#tf.config.experimental.set_memory_growth(gpus[0], True)
from tensorflow import keras
from keras.models import Sequential,Model,load_model
from keras.models import model_from_yaml
from keras.layers import Dense, MaxPool2D, Dropout, Activation, Flatten, Conv2D, PReLU, BatchNormalization, LeakyReLU, GlobalMaxPooling2D
'''
yaml_file = open('FeatherNetB-32.yaml', 'r')
loaded_model_yaml = yaml_file.read()
yaml_file.close()
loaded_model = model_from_yaml(loaded_model_yaml)
# load weights into new model
loaded_model.load_weights("new_second_testing_face_antispoof_classifier.h5")
print("Loaded model from disk")
model = loaded_model
'''
from keras.layers import Dense, MaxPool2D, Dropout, Activation, Flatten, Conv2D, PReLU, BatchNormalization, LeakyReLU
input_shape =(100,100,3)

#Visualization
from tensorflow.keras.callbacks import TensorBoard
name = "Inferance_face_antispoofing-{}".format(int(time.time()))
tensorboard = TensorBoard( histogram_freq=1, log_dir= 'logs/{}'.format(name))


tf.random.set_seed(5)
#importing ImageDataGenerator to scale datasets
from keras.preprocessing.image import ImageDataGenerator
img_gen = ImageDataGenerator(#rotation_range = 0,
                            #width_shift_range = 0,
                            #height_shift_range = 0.1,
                            rescale = 1/255,
                            #shear_range = 0,
                            #zoom_range = 0.1,
                            #horizontal_flip = False,

                            fill_mode = 'nearest'
)
img_gen.flow_from_directory('./face_antispoofing/train')#, target_size = (150,150,3))
#########now train###########
from tensorflow import keras

opt =  keras.optimizers.SGD(learning_rate=0.0003, momentum = 0.0004)
#opt =  keras.optimizers.Adam(learning_rate=0.03)
from tensorflow.keras.applications.resnet50 import ResNet50
base_model=ResNet50(weights='imagenet',include_top=False)

x = base_model.output


#x = Conv2D(filters =16, kernel_size=(3,3))(x)
#model.add(BatchNormalization())

#model.add(PReLU(alpha_initializer = 'zeros'))
#x = LeakyReLU(alpha = 0.1)(x)

#x = MaxPool2D(pool_size = (2,2))(x)
#model.add(Dropout(0.5))



#x = Conv2D(filters = 32, kernel_size=(3,3))(x)
#model.add(BatchNormalization())

#model.add(PReLU(alpha_initializer = 'zeros'))
#x = LeakyReLU(alpha = 0.1)(x)

#x = MaxPool2D(pool_size = (2,2))(x)
#model.add(Dropout(0.5))

x = GlobalMaxPooling2D()(x)
#x = Flatten()(x)

#x = Dense(512, activation = 'elu')(x)
x = Dense(2, activation = 'sigmoid')(x)

model = Model(inputs = base_model.input, outputs = x)

'''
model = tf.keras.applications.VGG16(
    include_top=False,
    weights="imagenet",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=2,
    classifier_activation="softmax",
)
'''
model.compile(loss = 'binary_crossentropy', optimizer = opt, metrics = ['accuracy'])
model.summary()

##creating image batch size for trainnig

batch_size = 16


train_img_gen = img_gen.flow_from_directory('./face_antispoofing/train', target_size = input_shape[:2], batch_size = batch_size,
                                                                             classes = ['Face','Nface'])


test_img_gen = img_gen.flow_from_directory('./face_antispoofing/test', target_size = input_shape[:2], batch_size = batch_size,
                                                                             classes = ['Face','Nface'])

print("\n\n\nClass inecies are\n\n", train_img_gen.class_indices)

import warnings
warnings.filterwarnings('ignore')

steps_per_epoch = len(train_img_gen)//batch_size
validation_steps = len(test_img_gen)//batch_size # if you have test data
#epoch 10 for batch 8
results = model.fit_generator(train_img_gen,epochs = 50, steps_per_epoch= steps_per_epoch, validation_data = test_img_gen, validation_steps = validation_steps, callbacks = [tensorboard])
model.save(str(name)+'_____inferance_face_antispoof_classifier.h5')

#from sklearn.metrics import confusion_matrix
#print("\n\n\nmy confusion_matrix is \n\n",confusion_matrix(test_img_gen, results))

'''
plt.plot(results.history['accuracy'])
#print(results.history['accuracy'])

plt.xlabel('time - axis')
plt.ylabel('accuracy - axis')
plt.title('accuracy graph!')
#plt.show()
plt.savefig('1accuracy_fig')

plt.plot(results.history['val_accuracy'])

plt.xlabel('time - axis')
plt.ylabel('Validation accuracy - axis')
plt.title('Validation Accuracy graph!')
#plt.show()
plt.savefig('1val_accuracy_fig')


plt.plot(results.history['loss'])

plt.xlabel('time - axis')
plt.ylabel('Loss - axis')
plt.title('Loss graph!')
#plt.show()
plt.savefig('1Loss_fig')


plt.plot(results.history['val_loss'])

plt.xlabel('time - axis')
plt.ylabel('Validation Loss - axis')
plt.title('Validation Loss graph!')
#plt.show()
plt.savefig('1val_loss_fig')
'''

from keras.utils.vis_utils import plot_model
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
