def confusion_reporting(true_values, pred_values):
    """
    This function takes in the true values of a dataset and the predicted values
    of the dataset and prints out a classification report, accuracy score, and
    plots the confusion matrix of the true and predicted values for simple analysis
    """
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, precision_score

    print(confusion_matrix(true_values, pred_values))
    print(classification_report(true_values, pred_values))
    print('Accuracy score:', round(accuracy_score(true_values, pred_values), 4))
    print('F1 score:', round(f1_score(true_values, pred_values), 4))

    cm = confusion_matrix(true_values, pred_values)
    df_cm = pd.DataFrame(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis],
                         index=['F', 'T'],
                         columns=['F', 'T'])
    plt.figure(figsize=(7, 5))
    sns.heatmap(df_cm, annot=True, cmap='Greens', vmin=0, vmax=1)
    plt.xlabel('Pred Val')
    plt.ylabel('True Val')
    plt.show()

    return


def plot_roc(y_test, y_pred, model_type=''):
    """
    Plot the roc curve for a given classification problem
    :param y_test: the test set of dependent variable values
    :param y_pred: the predicted set of dependent variable values
    :param model_type: the name of the model used to generate the predicted values
    :return:
    """
    from sklearn.metrics import roc_auc_score
    from sklearn.metrics import roc_curve

    fpr, tpr, thresholds = roc_curve(y_test, y_pred[:, 1])
    plt.figure()
    plt.plot(fpr, tpr, label='{} (area = {})'.format(model_type, roc_auc_score(y_test, y_pred)))
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.show()

    return


def tts(df, dep_col, holdout=0.3, scale=True):
    """
    Takes a dataframe and performs a train test split based on the holdout value, and performs a Standard Scale of the
    data if scale is set to True.
    :param df: the dataframe to split and scale
    :param dep_col: the dependent column
    :param holdout: the percentage to hold out for the test set (must be a decimal between 0 and 1
    :param scale: whether or not to scale the data after doing a train test split (default=True)
    :return:
    """
    from sklearn.preprocessing import StandardScaler

    indep_cols = [x for x in df.columns if x != dep_col]

    y = df[dep_col]
    X = df[indep_cols]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=holdout, random_state=0)

    if scale == True:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test


def over_sample_smote(x_train, y_train, columns):
    """
    Takes a training set of independent values (x_train) and dependent values (y_train) and oversamples the set to
    create balanced classes using SMOTE.
    :param x_train: the independent variables of the training set
    :param y_train: the dependent variable of the training set
    :param columns:
    :return: the oversampled independent variables as a dataframe, and a series of the oversampled dependent variable
    """
    from imblearn.over_sampling import SMOTE

    os = SMOTE(random_state=0)
    os_data_X, os_data_y = os.fit_sample(x_train, y_train)
    os_data_X = pd.DataFrame(data=os_data_X, columns=columns)

    return os_data_X, os_data_y