from pandas import read_csv
from sklearn.preprocessing import LabelEncoder

exclude_future = ['PassengerId', 'Cabin', 'Name', 'Ticket']


def prepare_data(file_name):
    data_frame = read_csv(file_name).drop(exclude_future, axis=1)

    fill_na_age(data_frame['Age'])

    data_frame['Sex'] = encode(data_frame, 'Sex')
    data_frame['Embarked'] = encode(data_frame, 'Embarked')

    return data_frame


def encode(data_frame, column):
    encoder = LabelEncoder()
    result = data_frame[column].astype(str).transform(encoder.fit_transform)
    print('{}: {}'.format(column, encoder.classes_))
    return result


def fill_na_age(age):
    age.fillna(age.mean(), inplace=True)

# prepare_data('./data/train.csv')

# without_object = train_df.select_dtypes(exclude='object')
#
# X = without_object.drop(['Survived'], axis=1)
# Y = without_object['Survived']

# parameters = {
#     'max_features': [4, 5, 6, 7, 8, 9, 10, 0.5, 'log2'],
#     'min_samples_leaf': [2, 3, 4],
#     'n_estimators': [35, 40, 45]
# }

# search_cv = GridSearchCV(RandomForestClassifier(random_state=42, n_jobs=-1, n_estimators=45), parameters, n_jobs=-1)
#
# search_cv.fit(X, Y)
#
# print(search_cv.best_params_)

# best_params = {'max_features': 9, 'min_samples_leaf': 4}
#
# classifier = RandomForestClassifier(n_estimators=45, random_state=42, n_jobs=-1, oob_score=True, **best_params)
# classifier.fit(X, Y)

# test_df = prepare_data('./data/test.csv')
#
# expected_df = read_csv('./data/gender_submission.csv')
#
# not_na = test_df.notna().all(axis=1)
#
# X_test = test_df[not_na].select_dtypes(exclude='object')
# Y_test = expected_df[not_na]['Survived']
#
# error = mean_absolute_error(Y_test, classifier.predict(X_test))
#
# print(error)


# porter = Porter(classifier)
#
# export = porter.export(class_name='TitanicModel', embed_data=True)
#
# with open('TitanicModel.java', mode='w') as file:
#     file.write(export)
