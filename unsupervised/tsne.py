import numpy as np
from sklearn.datasets import fetch_mldata
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import time
from sklearn.manifold import TSNE
# import tensorflow.examples.tutorials.mnist.input_data as input_data





# mnist=input_data.read_data_sets("MNIST")
mnist = fetch_mldata("MNIST original")
X = mnist.data / 255.0
y = mnist.target
# X = mnist['data']
# Y = mnist['target']
# y = pd.Series(mnist.target).astype('int').astype('category')
# X = pd.DataFrame(mnist.data)

print (X.shape, y.shape)


feat_cols = [ 'pixel'+str(i) for i in range(X.shape[1]) ]

df = pd.DataFrame(X,columns=feat_cols)
df['label'] = y
df['label'] = df['label'].apply(lambda i: str(i))

X, y = None, None

print ('Size of the dataframe: {}'.format(df.shape))

rndperm = np.random.permutation(df.shape[0])

# # Plot the graph
# plt.gray()
# fig = plt.figure( figsize=(16,7) )
# for i in range(0,30):
# 	ax = fig.add_subplot(3,10,i+1, title='Digit: ' + str(df.loc[rndperm[i],'label']) )
# 	ax.matshow(df.loc[rndperm[i],feat_cols].values.reshape((28,28)).astype(float))
# plt.show()



# pca = PCA(n_components=3)
# pca_result = pca.fit_transform(df[feat_cols].values)

# df['pca-one'] = pca_result[:,0]
# df['pca-two'] = pca_result[:,1] 
# df['pca-three'] = pca_result[:,2]

# print ('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))


# # Create the figure
# fig = plt.figure( figsize=(8,8) )
# ax = fig.add_subplot(1, 1, 1, title='PCA' )
# # Create the scatter
# ax.scatter(
#     x=df['pca-one'], 
#     y=df['pca-two'], 
#     c=df['label'], 
#     cmap=plt.cm.get_cmap('Paired'), 
#     alpha=0.15)


n_sne = 7000

time_start = time.time()
tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
tsne_results = tsne.fit_transform(df.loc[rndperm[:n_sne],feat_cols].values)

print ('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))


df_tsne = df.loc[rndperm[:n_sne],:].copy()
df_tsne['x-tsne'] = tsne_results[:,0]
df_tsne['y-tsne'] = tsne_results[:,1]

# Create the figure
fig = plt.figure( figsize=(8,8) )
ax = fig.add_subplot(1, 1, 1, title='TSNE' )
# Create the scatter
ax.scatter(
    x=df_tsne['x-tsne'], 
    y=df_tsne['y-tsne'], 
    c=df_tsne['label'], 
    cmap=plt.cm.get_cmap('Paired'), 
    alpha=0.15)
plt.show()

# pca_50 = PCA(n_components=50)
# pca_result_50 = pca_50.fit_transform(df[feat_cols].values)
# print ('Cumulative explained variation for 50 principal components: {}'.format(np.sum(pca_50.explained_variance_ratio_)))




