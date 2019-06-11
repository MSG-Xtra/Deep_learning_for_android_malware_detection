"""GAN built here is to generate adversarial malware samples that will be used
to test SmartAM1, train SmartAM2, test SmartAM2"""

from __future__ import print_function, division
from keras.layers import Input, Dense, Activation
from keras.layers.merge import Maximum, Concatenate
from keras.models import Model
from keras.optimizers import Adam
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time
start_time=time.time()

class SmartAM():
    def __init__(self):
        self.feature_dims = 357
        self.noise_dims = 40   
        self.hide_layers = 700
        self.generator_layers = [self.feature_dims+self.noise_dims, 
                                 self.hide_layers, self.feature_dims]
        self.discriminator_layers = [self.feature_dims, self.hide_layers, 1]
        self.blackbox = 'MLP'
        optimizer = Adam(lr=0.001) 
       

        # Build and Train blackbox_detector
        self.blackbox_detector = self.build_blackbox_detector()

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy', 
                                   optimizer=optimizer, metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()

        # The generator takes malware and noise as input and 
        #generates adversarial malware examples
        example = Input(shape=(self.feature_dims,))
        noise = Input(shape=(self.noise_dims,))
        input = [example, noise]
        malware_examples = self.generator(input)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated samples as input 
        #and determines validity
        validity = self.discriminator(malware_examples)

        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.combined = Model(input, validity)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)

    def build_blackbox_detector(self):

        if self.blackbox is 'MLP':
            blackbox_detector = MLPClassifier(hidden_layer_sizes=(20,),
                                              max_iter=10, alpha=1e-4,
                                              solver='sgd', verbose=0, 
                                              tol=1e-4, random_state=1,
                                              learning_rate_init=.1)
        return blackbox_detector

    def build_generator(self):

        example = Input(shape=(self.feature_dims,))
        noise = Input(shape=(self.noise_dims,))
        x = Concatenate(axis=1)([example, noise])
        for dim in self.generator_layers[1:]:
            x = Dense(dim)(x)
            x = Activation(activation='sigmoid')(x)
        x = Maximum()([example, x])
        generator = Model([example, noise], x, name='generator')
        generator.summary()
        return generator

    def build_discriminator(self):

        input = Input(shape=(self.discriminator_layers[0],))
        x = input
        for dim in self.discriminator_layers[1:]:
            x = Dense(dim)(x)
            x = Activation(activation='sigmoid')(x)
        discriminator = Model(input, x, name='discriminator')
        discriminator.summary()
        return discriminator

    def load_data(self, filename):

        data = np.load(filename)
        xmal, ymal, xben, yben = data['xmal'], data['ymal'], data['xben'],data['yben']
        return (xmal, ymal), (xben, yben)

    def train(self, epochs, batch_size=80):

        # Load the dataset
        (xmal, ymal), (xben, yben) = self.load_data('dataset_if.npz')
        xtrain_mal, xtest_mal, ytrain_mal, ytest_mal = train_test_split(xmal, ymal, test_size=0.25)
        xtrain_ben, xtest_ben, ytrain_ben, ytest_ben = train_test_split(xben, yben, test_size=0.25)

        # Train blackbox_detctor
        self.blackbox_detector.fit(np.concatenate([xmal, xben]),
                                   np.concatenate([ymal, yben]))

        ytrain_ben_blackbox = self.blackbox_detector.predict(xtrain_ben)
        Original_Train_TPR = self.blackbox_detector.score(xtrain_mal, ytrain_mal)
        Original_Test_TPR = self.blackbox_detector.score(xtest_mal, ytest_mal)
        Train_TPR, Test_TPR = [], []

        for epoch in range(epochs):

            for step in range(1):#range(xtrain_mal.shape[0] // batch_size):
                # ---------------------
                #  Train discriminator
                # ---------------------

                # Select a random batch of malware examples
                idx = np.random.randint(0, xtrain_mal.shape[0], batch_size)
                xmal_batch = xtrain_mal[idx]
                noise = np.random.uniform(0, 1, (batch_size, self.noise_dims))
                idx = np.random.randint(0, xmal_batch.shape[0], batch_size)
                xben_batch = xtrain_ben[idx]
                yben_batch = ytrain_ben_blackbox[idx]

                # Generate a batch of new malware examples
                gen_examples = self.generator.predict([xmal_batch, noise])
                ymal_batch = self.blackbox_detector.predict(np.ones
                                                            (gen_examples.shape)*
                                                            (gen_examples > 0.7))
                                
                #save adversarial samples
                np.savez("adverV8", xmal_adver = gen_examples)
                
                # Train the discriminator
                d_loss_real = self.discriminator.train_on_batch(gen_examples, ymal_batch)
                d_loss_fake = self.discriminator.train_on_batch(xben_batch, yben_batch)
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

                # ---------------------
                #  Train Generator
                # ---------------------

                idx = np.random.randint(0, xtrain_mal.shape[0], batch_size)
                xmal_batch = xtrain_mal[idx]
                noise = np.random.uniform(0, 1, (batch_size, self.noise_dims))

                # Train the generator
                g_loss = self.combined.train_on_batch([xmal_batch, noise], 
                                                      np.zeros((batch_size, 1)))

            # Compute Train TRR
            noise = np.random.uniform(0, 1, (xtrain_mal.shape[0], self.noise_dims))
            gen_examples = self.generator.predict([xtrain_mal, noise])
            TPR = self.blackbox_detector.score(np.ones(gen_examples.shape) *
                                               (gen_examples > 0.5), ytrain_mal)
            Train_TPR.append(TPR)

            # Compute Test TRR
            noise = np.random.uniform(0, 1, (xtest_mal.shape[0], self.noise_dims))
            gen_examples = self.generator.predict([xtest_mal, noise])
            TPR = self.blackbox_detector.score(np.ones(gen_examples.shape) * 
                                               (gen_examples > 0.5), ytest_mal)
            Test_TPR.append(TPR)

            # Plot the progress
            print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0],
                  100*d_loss[1], g_loss))

        print('Original_Train_TPR: {0}, Adver_Train_TPR: {1}'.format(Original_Train_TPR, Train_TPR[-1]))
        print('Original_Test_TPR: {0}, Adver_Test_TPR: {1}'.format(Original_Test_TPR, Test_TPR[-1]))

        # Plot TRR
        plt.figure()
        plt.plot(range(epochs), Train_TPR, c='g', label='Training Set', linewidth=2)
        plt.plot(range(epochs), Test_TPR, c='r', linestyle='--', label='Validation Set', linewidth=2)
        plt.xlabel("Epoch")
        plt.ylabel("TPR")
        plt.legend()
        plt.show()

if __name__ == '__main__':
    model = SmartAM()
    model.train(epochs=1000, batch_size=300)

end_time=time.time()
print('Execution time: '+str(round( end_time -start_time, 3))+'seconds')
 