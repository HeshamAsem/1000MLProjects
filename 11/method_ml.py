
from sklearn.metrics import classification_report , confusion_matrix , auc ,roc_auc_score , roc_curve
def check_df(dataframe, head=5):
    print("############Shape############")
    print(dataframe.shape)
    print("############Types############")
    print(dataframe.dtypes)
    print("############NA############")
    print(dataframe.isnull().sum())
    print("############Quantiles############")
    print(dataframe.describe())
# This is a function that generates a confusion matrix visualization.
def make_confusion_matrix(cf,cmap='Blues'):
    # CODE TO GENERATE TEXT INSIDE EACH SQUARE
    #convert the value in cf from array to list   
    cf_values = ["{0:0.0f}\n".format(value) for value in cf.flatten()]
    values_percentages = ["{0:.2%}".format(value) for value in cf.flatten()/np.sum(cf)]
    cf_matrix = [f"{v1}{v2}" for  v1, v2 in zip(cf_values,values_percentages)]
    cf_matrix = np.array(cf_matrix).reshape(cf.shape[0],cf.shape[1])
    # CODE TO GENERATE SUMMARY STATISTICS & TEXT FOR SUMMARY STATS
    #Accuracy is sum of diagonal divided by total observations
    accuracy  = np.trace(cf) / float(np.sum(cf))
    precision = cf[0,0] / sum(cf[:,0])
    recall    = cf[0,0] / sum(cf[0,:])
    f1_score  = 2*precision*recall / (precision + recall)
    stats_text = "\n\nAccuracy={:0.3f}\nPrecision={:0.3f}\nRecall={:0.3f}\nF1 Score={:0.3f}".format(
               accuracy,precision,recall,f1_score)

    # MAKE THE HEATMAP VISUALIZATION
    plt.figure(figsize=(8,6))
    sns.heatmap(cf,annot=cf_matrix,fmt="",cmap=cmap,cbar='True')
    plt.xlabel('Predicted label' + stats_text)
    plt.title('Confusion Matrix \n' , size = 20)

def f_importances(coef, names):
    coef , names=zip(*sorted(list(zip(coef, names))))
    # Show all features
  
    plt.figure(figsize=(16, 8), dpi=80)
    sns.barplot(x=list(coef[::-1]), y=list(names[::-1]), palette='RdYlGn')
    plt.yticks(range(len(names)), names[::-1])
    plt.title("Feature Importance");
    plt.show()

def try_model(data , model,x_train_res, y_train_res , x_test_res , y_test_res):
    model.fit(x_train_res, y_train_res)
    pred = model.predict(x_test_res)
    #calculat depend on precision , recall , f1-score and its more accurate 
    #than it calculate in model
    report =classification_report(y_test_res , pred )
    #print(report.split())
    print('Classification report after overSample : \n{}'.format(report))
    #calculate depend on predicting data for set of information
    #acurracy traning
    print('Tranning Accaracy : {} ' .format(model.score(x_train_res, y_train_res)))

    #acuracy test
    print('Test Accuracy : {} '.format (model.score(x_test_res, y_test_res)))
    return pred ,report 
def plot_auc_roc( modell,y_True , y_pre):
    fpr ,tpr ,threshould = roc_curve(y_True, y_pre)
    aucValue = auc(fpr , tpr)
    #auc_data = pd.DataFrame({'Thresoulds' : threshould ,
    #                         'Tpr' : tpr ,'fpr' : fpr})
    #print(auc_data)
    plt.figure()
    plt.title(modell + ' AUC')
    plt.plot(fpr , tpr , label = 'aucValue = %0.2f' % aucValue)
    plt.legend(loc = 'lower right')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.show()
    return fpr , tpr ,threshould 