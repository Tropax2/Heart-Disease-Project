from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA 
def build_model():
    return QDA(reg_param=0.5)