# Imports
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

# To ensure that the models keep the frist randomized data
np.random.seed(42)

# ### Baseline Models

#### 1. Logistic Regression

# Accuracy
print('Logistic Regression Test Score is: ', log_model.score(x_test, y_test))

#### 2. Decision Tree 

# Accuracy
print('DecisionTree Classifier Test Score is: ', dec_model.score(x_test, y_test))

#### 3. Linear Support Vector Classification

# Accuracy
print('LinearSVC Test Score is: ', lin_model.score(x_test, y_test))

### KFold Cross Validation

kfold = KFold(n_splits=5)

models = [('Logistic Regression: ', log_model), ('DecisionTree: ', dec_model), ('LinearSVC: ', lin_model)]

tot_results = []
for model_name, model in models:
    cv_results = cross_val_score(model, x_train_smote, y_train_smote, cv=kfold, scoring='accuracy')
    tot_results.append([model_name, cv_results])
    
# Print 5-fold cv performance
print(tot_results)

### Hyperparameter Tuning

# Defining hyper-parameters that'll be used by GridSearchCV
params = {
    'criterion':  ['gini', 'entropy'], 
    'max_depth':  [None, 2, 4, 6, 8, 10],
    'max_features': [None, 'sqrt', 'log2', 0.2, 0.4, 0.6, 0.8],
    'splitter': ['best', 'random']
}

# Model instantiation
grid_cv = GridSearchCV(estimator=dec_model,
                      param_grid=params,
                      cv=5,
                      n_jobs=5,
                      verbose=1)
                      
# print the best hyper parameters
print(grid_cv.best_params_)

# Re-training decisionTree classifier since it's the one that performed the best initially
dec_model_2 = DecisionTreeClassifier(criterion='entropy', max_depth=None,
                                     max_features=0.6, splitter='best')
dec_model_2.fit(x_train_smote, y_train_smote)

# Accuracy
print('DecisionTree Classifier Test Score is: ', dec_model_2.score(x_test, y_test))





