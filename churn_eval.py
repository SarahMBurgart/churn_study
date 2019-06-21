# import regressors
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier


# import evaluations tools
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import mean_squared_error, r2_score

# import plotting and libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def cv_train_scores(models, X, y, scoring1, scoring2):
    # already is Kfolds
    # takes fit models, names, X and y and returns ~ table of scores
    # returns name of model, MSE and R2 for 3 regressors
    res = []
    for model in models:
        mse = np.mean(cross_val_score(model, X, y, cv=5, scoring=scoring1))
        r2 = np.mean(cross_val_score(model, X, y, cv=5, scoring=scoring2))
       
        res.append(f"{model.__class__.__name__} Train CV | {scoring1}: {mse:.3f} | {scoring2}: {r2:.3f} ")
    return res

def stage_score_plot(estimator, Xtrain, ytrain, Xtest, ytest, learning_rate,n_estimators):
    '''
    Parameters: estimator: GradientBoostingRegressor or AdaBoostRegressor
                X_train: 2d numpy array
                y_train: 1d numpy array
                X_test: 2d numpy array
                y_test: 1d numpy array

    Returns: A plot of the number of iterations vs the MSE for the model for
    both the training set and test set.
    '''
   
    # staged_predict (self, X) predict regression target at each stage for X 
    # allows determining error on testing set after each stage - returns predicted value
    
    model = estimator(learning_rate=learning_rate, n_estimators=n_estimators)
    
    model.fit(Xtrain, ytrain)
    y_hat_train = model.staged_predict(Xtrain)
    y_hat_test = model.staged_predict(Xtest)
    
    y_hat_train_lst = []
    y_hat_test_lst = []
    
    xpts = range(1,(n_estimators+1))
    
    for y in y_hat_train:
        y_hat_train_lst.append(mean_squared_error(ytrain, y))
        
    for y in y_hat_test:
        y_hat_test_lst.append(mean_squared_error(ytest, y))
    
    
    fig, ax = plt.subplots(1,1, figsize= (20,8))
    
    ax.plot(xpts,y_hat_train_lst, color='b' )
    ax.plot(xpts, y_hat_test_lst, color='r')
    ax.set_ylabel("MSE")
    ax.set_xlabel("Iterations")

    plt.legend([f"{model.__class__.__name__} Train - learning rate {learning_rate}", f"{model.__class__.__name__} Test - learning rate {learning_rate}"])
    plt.show()

def plot_feature_importances(model):
    # takes fitted model as input
    # return plot of feature importances and standard deviations

    importance = np.std([tree.feature_importances_
                            for tree in model.estimators_], axis=0)

    x = range(importance.shape[0])
    y = model.feature_importances_
    yerr = importance

    # Extend the code to find the standard deviation of the importance for each feature across all trees. 
    fig, ax = plt.subplots(1,1, figsize=(20,8), sharex=True)
    #axs.flatten()

    ax.errorbar(x,y, yerr=yerr)
    ax.set_title('all errorbars')


    #fig.suptitle('Errors')
    plt.show()
    plt.tight_layout()

# Random Forest
def change_num_features(num_features, X, y, num_estimators):
    # num_featuers is stop for range only.
    # returns a plot of the model score against the num of features in each tree
    num_features = range(1, num_features)
    Xtrain, Xtest, ytrain, ytest = train_test_split(X, y)
    scores = []
    for n in num_features:
        model = RandomForestClassifier(oob_score=True, n_estimators=num_estimators, max_features=n)
        model.fit(Xtrain, ytrain)
        base_test = model.predict(Xtest)
        # accuracy of test set
        #model.score(Xtest, ytest)
        scores.append(model.score(Xtest, ytest))
    print(scores)
    plt.plot(num_features,scores)

def neural_network(X,y):
    # takes X and y as parameters
    # returns plot of yhat, train and test scores, parameters 
    Xtrain, Xtest, ytrain, ytest = train_test_split(X, y)
    model = MLPClassifier()
    model.fit(Xtrain, ytrain)
    model.predict(Xtest)
    train_score = model.score(Xtrain, ytrain)
    test_score = model.score(Xtest, ytest)
    params = model.get_params()

    print(f"train score: {train_score:.3f} | test_score: {test_score:.3f} | params: {params}")

    #xpts = range(len(model.predict(Xtrain)))
    #fig, ax = plt.subplots(1,1, figsize= (20,8))
    
    #ax.plot(xpts,model.predict(Xtrain), color='b' )
    #ax.plot(xpts, y_hat_test_lst, color='r')
    #ax.set_ylabel("yhat")
    #ax.set_xlabel("over xpts")

    #plt.legend([f"{gdbr.__class__.__name__} Train - learning rate 0.1", f"{model.__class__.__name__} Test - learning rate 0.1"])
    #plt.show()
'''

it is already implicitly in the cross_val_score above



def k_folds_cv(k, Xtrain, ytrain, model):
    'take "k" , Xtrain, ytrain and model (LinearRegresion, RandomForestRegressor)
    returns score'

    # randomly split dataset into kfolds (you may use sklearn's KFold class)
    kf = KFold(k, shuffle=True)
    indices =  kf.split(Xtrain, ytrain) 
    rmse_list = []
    # x[0]equals training set / x[1]equals test set   
    # for each fold:
    for i in indices:
        #print(i.shape)   
        
        # train the model with the other folds
        # x[0]equals training set / x[1]equals test set   
        r = model()
        r.fit(X_train[i[0]], y_train[i[0]])
        
        # use the trained model find y-hats for X_train in current fold
        t_predicted = r.predict(X_train[i[1]])
    
        # calculate the RMSE of y-hats
        y_t = y_train[i[1]]
        
        # store RMSE for current fold
        rmse_list.append(rmse(y_t, t_predicted))
        
    # average the k results of your error metric and return
    return (sum(rmse_list)/k)
'''
