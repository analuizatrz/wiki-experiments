import pandas as pd

number_for_class = {
    'stub':0,
    'start':1,
    'b':2,
    'a':3,
    'ga':4,
    'fa':5,
}

class_for_number = {v: k for k, v in number_for_class.items()}

def read_dataset():
    dataset_1 = pd.read_csv("data/6-join_multiple_class_without_C-evolution.csv")
    dataset_2 = pd.read_csv("data/6-join-single-class-without-C-evolution.csv")
    dataset = pd.concat([dataset_1, dataset_2])

    dataset["actual_category"] = dataset["actual_category"].apply(lambda x: number_for_class[x])
    dataset["past_category"] = dataset["past_category"].apply(lambda x: number_for_class[x])

    return dataset

def read_dataset_metadata():
    target_column = pd.read_csv("metadata/target-column", header=None)[0][0]
    feature_columns = list(pd.read_csv("metadata/features-columns", header=None)[0])
    feature_columns_with_delta = list(pd.read_csv("metadata/features-columns-with-delta", header=None)[0])
    feature_columns_only_delta = list(pd.read_csv("metadata/features-columns-only-delta", header=None)[0])

    return target_column, feature_columns, feature_columns_with_delta, feature_columns_only_delta

def get_X_y(dataset, feature_columns, target_column):
    X = dataset[feature_columns]
    y = dataset[target_column]
    return X, y

def get_X_y_without_duplicates(dataset, feature_columns, target_column):
    columns = [target_column] + feature_columns
    filtered = dataset[columns].drop_duplicates()
    return get_X_y(filtered, feature_columns, target_column)

def evaluate_functions(functions_dict, y_true, y_pred):
    return {k: v(y_true, y_pred) for k, v in functions_dict.items()}

def evaluate_method(method, metrics_functions, X_train, y_train, X_test, y_test):
    model = method.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    score = evaluate_functions(metrics_functions, y_test, y_pred)
    score["número de instâncias"] = len(X_train) + len(X_test)
    return [model], [score]
    
def create_method_evaluator(method, metrics_functions):
    def F(X_train, y_train, X_test, y_test):
        return evaluate_method(method, metrics_functions, X_train, y_train, X_test, y_test)
    return F

