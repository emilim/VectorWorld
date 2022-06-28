import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import os
from keras.applications.vgg16 import VGG16
from keras.models import Model
os.environ['TFHUB_CACHE_DIR'] = '/home/user/workspace/tf_cache'
fig, ax = plt.subplots()

#load the dataset
'''(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()'''
datasets = tf.keras.utils.image_dataset_from_directory("C:\\Users\\emili\Documents\\GitHub\\My own DeepDPM\\datasets", image_size=(360, 640), batch_size=500, shuffle=False)
for image_batch, labels_batch in datasets:
    images = image_batch
    labels = labels_batch
    break
x_test = images.numpy().astype(np.uint8)
x = x_test.reshape(x_test.shape[0], -1) #flatten x_test of dimension (n, 360, 640, 3) to (n, 360*640*3)
'''x = x_test.flatten().reshape(len(x_test), len(x_test[0]) * len(x_test[0][0]))'''

data_preprocessing = tf.keras.Sequential(
    [
        tf.keras.layers.Resizing(224, 224),
        tf.keras.layers.Normalization(),
    ]
)
data_preprocessing.layers[-1].adapt(x_test)
x_test = data_preprocessing(x_test)
#feature extraction
model = VGG16()
# remove the output layer
model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
feat = model.predict(x_test)

#dimensionality reduction
data = TSNE(n_components=2, init="random", learning_rate="auto").fit_transform(feat)
#data = PCA(20).fit_transform(x)

def imscatter(x, y, image, zoom=0.4):
    i = 0
    for x0, y0, images in zip(x, y, image):
        im = OffsetImage(images, zoom=zoom)
        ab = AnnotationBbox(im, (x0, y0), frameon=False)
        if i % 10 == 0:
            ax.add_artist(ab)
        i+=1
imscatter(data[:, 0], data[:, 1], x_test, zoom=0.05)

'''plt.style.use('seaborn-whitegrid')
plt.figure(figsize = (10,6))
c_map = plt.cm.get_cmap('jet', 10)
plt.scatter(data[:, 0], data[:, 1], s = 15, cmap = c_map, c = y_test)
plt.colorbar()'''

# clustering
db = DBSCAN(eps=5, min_samples=10).fit(data)
labels = db.labels_

core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)


# Black removed and is used for noise instead.
unique_labels = set(labels)
colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = [0, 0, 0, 1]

    class_member_mask = labels == k

    xy = data[class_member_mask & core_samples_mask]
    ax.plot(
        xy[:, 0],
        xy[:, 1],
        "o",
        markerfacecolor=tuple(col),
        markeredgecolor="k",
        markersize=7,
    )

    xy = data[class_member_mask & ~core_samples_mask]
    ax.plot(
        xy[:, 0],
        xy[:, 1],
        "o",
        markerfacecolor=tuple(col),
        markeredgecolor="k",
        markersize=3,
    )

print("Estimated number of noise points: %d" % n_noise_)
plt.title("Estimated number of clusters: %d" % n_clusters_)
plt.show()