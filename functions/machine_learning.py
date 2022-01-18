import time
import pandas as pd

from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

def modeling(features,target,model,display_results_list):

    start = time.time()
    
    X_train, X_test, y_train, y_test= train_test_split(features, target, test_size=0.2,shuffle=True,random_state=123)

    model.fit(X_train, y_train)
    
    y_pred_train=model.predict(X_train)
    y_pred_test=model.predict(X_test)
    acc_train=model.score(X_train, y_train) # retourne l'accuracy du dataset d'entrainement
    acc_test=model.score(X_test, y_test) # retourne l'accuracy du dataset d'entrainement
    
    if 'confusion_matrix' in display_results_list:
        cm_train = pd.crosstab (y_train, y_pred_train, rownames=['Reality'], colnames=['Prediction'])
        display(cm_train)
        
    if 'classification_report' in display_results_list:
        print('CLASSIF ON TRAIN:\n\n',
              classification_report(y_train, y_pred_train))
    
    if 'confusion_matrix' in display_results_list:
        cm_test = pd.crosstab (y_test, y_pred_test, rownames=['Reality'], colnames=['Prediction'])
        display(cm_test)
        
    if 'classification_report' in display_results_list:
        print('CLASSIF ON TEST:\n\n',
              classification_report(y_test, y_pred_test))

    if 'scores' in display_results_list:
        print('ACCURACIES:\n',
              '\n train dataset :', round(acc_train*100,2),'%',
              '\n test dataset :', round(acc_test*100,2),'%')
        
    end = time.time()
    elapsed_train = end - start
    print(f'Time elapsed to train model: {elapsed_train//60 :0}min {round(elapsed_train % 60,0):0}sec')
        
    return acc_train,acc_test,elapsed_train