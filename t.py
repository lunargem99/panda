def methods():
    print('''
    1) clustering
    2) ann
    3) decisiontree
    4) regression
    5) preprocessing
          ''')

def clustering():
    print('''
            
    data["Clusters"] = kmeans.labels_
    print(data)
            
    1) kmeans
            
    from sklearn.cluster import KMeans

    kmeans = KMeans(n_clusters=2)
    kmeans.fit(data)

    plt.scatter(df["xyz"],df["abc"],c=kmeans.labels_)
    plt.show()
            
    2) kmedoids

    !pip install scikit-learn-extra
    from sklearn_extra.cluster import KMedoids

    kmedoids = KMedoids(n_clusters=2)
    kmedoids.fit(data)

    plt.scatter(df["xyz"],df["abc"],c=kmedoids.labels_)
    plt.show()
    
    3) agglomerative

    from sklearn.cluster import AgglomerativeClustering
    import scipy.cluster.hierarchy as shc

    agglo = AgglomerativeClustering().fit(data)

    plt.figure(figsize =(8, 8))
    plt.title('Visualising the data')
    Dendrogram = shc.dendrogram((shc.linkage(data, method ='ward')))

    4) naive bayes
    
    from sklearn import datasets
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
          
    from sklearn.naive_bayes import GaussianNB
    
    clf = GaussianNB()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2f}")
        
''')

def ann():
    print('''

    import tensorflow as tf
    ann = tf.keras.models.Sequential() #initialize ANN

    ann.add(tf.keras.layers.Dense(units=6, activation="relu")) #relu - rectified linear unit #adding 1st hidden layer
    ann.add(tf.keras.layers.Dense(units=6, activation="relu")) #adding 2nd hidden layer
    ann.add(tf.keras.layers.Dense(units=1, activation="sigmoid")) # 1, sigmoid (binary) # >1, softmax (categorical)

    ann.compile(optimizer="adam",loss="binary_crossentropy",metrics=['accuracy']) #compile ANN // loss = categorical_crossentrophy
    ann.fit(x_train,y_train,batch_size=32,epochs=100)
          
    output = ann.predict(x_test)>0.5
    print(output)
    
    ann.save("ANN.h5")

          ''')

def decisiontree():
    print('''

    from sklearn.tree import DecisionTreeClassifier, plot_tree
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    dt_classifier = DecisionTreeClassifier(mmax_depth=3 ,criterion='gini', random_state=42)
    dt_classifier.fit(X_train, y_train)
    plt.figure(figsize=(12, 8))
    plot_tree(dt_classifier, feature_names=selected_features, class_names=dt_classifier.classes_, filled=True)
    plt.show()
          
    y_pred = dt_classifier.predict(X_test)

    !!)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    print("Evaluation Metrics:")
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1-Score: {f1:.2f}")     
          
        ''')

def regression():
    print('''
    
    1) Linear
          
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import  r2_score
    import matplotlib.pyplot as plt
    
    x=df.iloc[:,:-1]
    y=df.iloc[:,-1]
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)
    mod=LinearRegression()
    mod.fit(x_train,y_train)
    y_pred=mod.predict(x_test)
    r2=r2_score(y_test,y_pred)
    print(f"R2 Score:{r2}")
    plt.plot(x_test,y_pred,color='blue',label='Actual data vs Predicted data')
    plt.scatter(x_test,y_test,color='red')
    plt.show()
    
    2)Poly
    
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.pipeline import make_pipeline
    import matplotlib.pyplot as plt
    poly=make_pipeline(PolynomialFeatures(3), LinearRegression())
    poly.fit(x_train,y_train)
    y_pred_poly=poly.predict(x_test)  
    plt.scatter(x_test['YearsExperience'], y_test, color='blue', label='Actual Data')
    sort_order = np.argsort(x_test['YearsExperience'])
    plt.plot(x_test['YearsExperience'].values[sort_order], y_pred_poly[sort_order], color='red', label='Polynomial Regression')
    plt.xlabel('YearsExperience')
    plt.ylabel('Salary')
    plt.legend()
    plt.title('Polynomial Regression (Degree = {})'.format(degree))
    plt.show()
          
          ''')

def preprocessing():
    print('''
    
    1)
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    2)  
    import matplotlib.pyplot as plt
    import seaborn as sns
    cc = df.corr()
    sns.heatmap(cc, annot=True, cmap=plt.cm.Reds)
    plt.show()      
          
    3)
    from sklearn.feature_selection import RFE
    from sklearn.linear_model import LogisticRegression
    dataframe = pd.read_csv("diabetes.csv")
    array = dataframe.values
    X = array[:,0:8] #except last col
    Y = array[:,8] #only last col

    model = LogisticRegression(solver="lbfgs")
    rfe = RFE(model,step=3)
    fit = rfe.fit(X, Y)
    print("Num Features: %d" % fit.n_features_)
    print("Selected Features: %s" % fit.support_)
    print("Feature Ranking: %s" % fit.ranking_)
    
    4)
    x = pd.get_dummies(x,columns=["Gender","Geography"],drop_first=True)
    sc = StandardScaler()
    x_train,x_test = [sc.fit_transform(x_train),sc.fit_transform(x_test)]
                 ''')
