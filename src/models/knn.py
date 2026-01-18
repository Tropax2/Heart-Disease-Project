from sklearn.neighbors import KNeighborsClassifier as KNN  
def build_model(n_neighbors: int):
    return KNN(n_neighbors)