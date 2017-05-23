from sklearn import metrics
def build_model_and_evaluate(clf, train_Xmat, train_y, test_Xmat, test_y, test_Xo, report=True):
    # training
    clf.fit(train_Xmat, train_y)
    # test
    pred = clf.predict(Xmat=test_Xmat, Xo=test_Xo)
    # score
    accuracy = metrics.accuracy_score(test_y, pred)
    if report:
        print('-' * 100)
        print('classifier:')
        print(clf)

        print("macro f1 score:   %0.3f" % metrics.f1_score(test_y, pred, average='macro'))
        print"accuracy:   %0.3f" % accuracy, '\n\n'
        print(metrics.classification_report(test_y, pred))
        print()
        print(metrics.confusion_matrix)
    print()
    return accuracy
