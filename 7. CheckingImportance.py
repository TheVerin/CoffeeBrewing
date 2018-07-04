import warnings
warnings.filterwarnings('ignore')

#Looking for most important values
from sklearn.feature_selection import RFE
acc_all =[]
stab_all = []
for t in np.arange(0, 20, 1):
    acc_loop = []
    stab_loop = []
    for b in np.arange(3, 13, 1):
        selector = RFE(XGB, b, 1)
        cv = cross_val_score(XGB, X_train.iloc[:, selector.fit(X, Y).support_],
                             Y_train, cv = 10, scoring = 'accuracy')
        acc_loop.append(cv.mean())
        stab_loop.append(cv.std()*100/cv.mean())
    acc_all.append(acc_loop)
    stab_all.append(stab_loop)
acc = pd.DataFrame(acc_all, columns = np.arange(3 , 13, 1))
stab = pd.DataFrame(stab_all, columns = np.arange(3, 13, 1))
print(acc.agg('mean'))
print(stab.agg('mean'))

selector = RFE(XGB, 9, 1)
values = X_train.iloc[:, selector.fit(X_train, Y_train).support_].columns
print(values)

XGB.fit(X_train, Y_train)
features = pd.DataFrame()
features['feature'] = X_train.columns
features['importance'] = XGB.feature_importances_
features.sort_values(by = 'importance', ascending = True, inplace = True)
features.set_index('feature', inplace = True)
features.plot(kind = 'barh')