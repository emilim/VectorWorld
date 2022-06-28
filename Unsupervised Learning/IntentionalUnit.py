from unicodedata import category
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
from keras.models import Model
import os
os.environ['TFHUB_CACHE_DIR'] = '/home/user/workspace/tf_cache'

class IntentionalUnit:
    def __init__(self, robot):
        self.robot = robot
        self.model = keras.applications.resnet50.ResNet50(weights="imagenet") 
        self.VGG16 = keras.applications.vgg16.VGG16()
        self.VGG16 = Model(inputs=self.VGG16.inputs, outputs=self.VGG16.layers[-2].output)
        self.num_clusters = 10
        self.clusters = 0
        self.images = []
        self.y = 0
        self.features = 0
        self.motorsValue = [0, 0, 0, 0]
    
    def start(self, img_ori):
        img_ori = tf.image.resize([img_ori], [224, 224])

        #rLearnP = self.philogeneticModule(img_ori)
        #rLearnO = self.ontogeneticModule(self.features)
        #self.y = rLearnP + rLearnO
        self.features = self.categoryModule(img_ori, self.y)
        
        #self.motorsValue[0] = self.y*80*0
        #self.motorsValue[1] = self.y*80*0
       
        #self.intentionalityModule(self.motorsValue[0], self.motorsValue[1], 0, 0)
        
        return self.y

    def categoryModule(self, img_ori, y):
        features = self.VGG16.predict(img_ori)
        self.images.append(features)
        
        if len(self.images) >= 100 and len(self.images) % 100 == 0:
            print(len(self.images))
            images = np.squeeze(self.images, axis=(1, ))
            self.clusters = self.clustering(images)

            if len(self.images) > 1000:
                self.images.pop(0)
        return (features, y)

    def philogeneticModule(self, img_ori):
        inputs = keras.applications.resnet50.preprocess_input(img_ori)
        y = self.model.predict(inputs)
        topK = keras.applications.resnet50.decode_predictions(y, top=3)
        print(topK)
        return topK[0][0][2] #probability of the first element
        
    def ontogeneticModule(self, features):
        #calculate the distance between the features and the clusters
        distances = []
        if self.clusters is not 0:
            for i in range(len(features)):
                distances.append(np.linalg.norm(features[i] - self.clusters.cluster_centers_[self.clusters.labels_[i]]))
        return np.mean(distances)

        '''

        '''
        #return np.argmin(distances)

    def intentionalityModule(self, leftMV, rightMV, headV, liftV):
        self.robot.motors.set_wheel_motors(leftMV, rightMV)
        self.robot.motors.set_head_motor(headV)
        self.robot.motors.set_lift_motor(liftV)

    def clustering(self, features):
        #tsne = TSNE(n_components=2, random_state=22)
        #x_tsne = tsne.fit_transform(features)

        pca = PCA(n_components=100, random_state=22)
        x_pca = pca.fit_transform(features)

        kmeans = KMeans(n_clusters=self.num_clusters, random_state=22)
        print(features)
        print(features.shape)
        kmeans.fit(features)
        print("CLUSTERED")
        #ax = fig.add_subplot(projection='3d')
        #plt.scatter(x_tsne[:,0], x_tsne[:,1], c=kmeans.labels_)
        #plt.show()
        
        return kmeans #(kmeans.labels_, kmeans.cluster_centers_)
