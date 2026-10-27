import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.cluster import DBSCAN,KMeans

X , _ = make_moons(n_samples=300,noise=0.08,random_state=42)

db = DBSCAN(eps=0.2,min_samples=5).fit_predict(X)
km = KMeans(n_clusters=2,random_state=42).fit_predict(X)

fig,ax = plt.subplots(1,3,figsize=(14,4))

ax[0].scatter(X[:,0],X[:,1])
ax[0].set_title("Without label")

ax[1].scatter(X[:,0],X[:,1],c=km)
ax[1].set_title("Kmean label")

ax[2].scatter(X[:,0],X[:,1],c=db)
ax[2].set_title("DBSCAN label")

plt.show()