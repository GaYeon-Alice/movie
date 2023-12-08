from sklearn.metrics import log_loss, roc_auc_score
from deepctr.models import DeepFM
from deepctr.feature_column import SparseFeat, get_feature_names

def train_model(data, sparse_features, train):

    fixlen_feature_columns = [SparseFeat(feat, vocabulary_size=data[feat].nunique(), embedding_dim=4)
                            for feat in sparse_features]
    dnn_feature_columns = fixlen_feature_columns
    linear_feature_columns = fixlen_feature_columns

    feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)

    train_model_input = {name: train[name] for name in feature_names}

    model = DeepFM(linear_feature_columns, dnn_feature_columns, task='binary')
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['AUC'])

    model.fit(train_model_input, train['target'].values, batch_size=256, epochs=10, verbose=2, validation_split=0.2)

    return feature_names, model


def evaluate_model(test, feature_names, model):

    test_model_input = {name: test[name] for name in feature_names}
    pred_ans = model.predict(test_model_input, batch_size=256)

    print("\ntest LogLoss:", round(log_loss(test['target'].values, pred_ans), 4))
    print("test AUC:", round(roc_auc_score(test['target'].values, pred_ans), 4))


def predict_all(data, feature_names, model):

    model_input = {name: data[name] for name in feature_names}
    pred_all = model.predict(model_input, batch_size=256)

    return pred_all
