from sklearn.metrics import precision_recall_curve, auc, roc_auc_score
import tensorflow as tf
import numpy as np

def dice_coef(y_true, y_pred, smooth=1e-10):
    """
    dice_coef =  2*TP/(|pred|+|true|)
    the code below works because labels are one-hot enconded.

    :param y_true:Tensor - shape (batch_size, 2)
        Truth labels one hot encoded
    :param y_pred:Tensor - shape (batch_size, 2)
        prediction value probabilities 
    :return: Tensor - single float value in range [0,1]
        Sorensen-Dice coefficient which is equal to F_1 score for binary classifications
    """
    y_true_f = tf.keras.backend.flatten(tf.cast(y_true, 'float32'))
    y_pred_f = tf.keras.backend.flatten(tf.cast(y_pred, 'float32'))
    return (2. * tf.reduce_sum(y_true_f * y_pred_f)) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

def specificity(y_true, y_pred, threshold=0.5):
    """
    :param y_true:Tensor - shape (batch_size, 2)
        Truth labels one hot encoded
    :param y_pred:Tensor - shape (batch_size, 2)
        prediction probabilities
    :return: Tensor - single float value in range [0,1]
        specificity value TN/(TN+FP)
    """
    m = tf.keras.metrics.Recall(thresholds=threshold, class_id=0)
    m.update_state(y_true,y_pred)
    specificity = m.result().numpy()

    """Differentiable version for using with loss in training
    true_negatives = tf.cast(tf.reduce_sum(tf.round(tf.clip_by_value((1 - y_true) * (1 - y_pred), 0, 1))), tf.float32)
    possible_negatives = tf.cast(tf.reduce_sum(tf.round(tf.clip_by_value(1 - y_true, 0, 1))), tf.float32)
    specificity = true_negatives / (possible_negatives + 1.0e-7)
    """
    
    return specificity


def precision(y_true, y_pred, threshold=0.5):
    """
    :param y_true:Tensor - shape (batch_size, 2)
        Truth labels one hot encoded
    :param y_pred:Tensor - shape (batch_size ,2)
        prediction probabilities
    :return: Tensor - single float value in range [0,1]
        precision value TP/(TP+FP)
    """
    m = tf.keras.metrics.Precision(thresholds=threshold)
    m.update_state(y_true,y_pred)
    precision = m.result().numpy()

    """Differentiable version for using with loss in training
    true_positives = tf.cast(tf.reduce_sum(tf.round(tf.clip_by_value(y_true * y_pred, 0, 1))), tf.float32)
    prp = tf.cast(tf.reduce_sum(tf.round(tf.clip_by_value(y_true, 0, 1))), tf.float32)
    precision = true_positives / (prp + 1e-7)
    """

    return precision
    
def sensitivity(y_true, y_pred, threshold=0.5):
    """
    :param y_true:Tensor - shape (batch_size, 2)
        Truth labels one hot encoded
    :param y_pred:Tensor - shape (batch_size, 2)
        prediction probabilities
    :return: Tensor - single float value in range [0,1]
        sensitivity value TP/(TP+FN)
    """

    m = tf.keras.metrics.Recall(thresholds=threshold, class_id=1)
    m.update_state(y_true,y_pred)
    sensitivity = m.result().numpy()
    
    """Differentiable version for using with loss in training
    true_positives = tf.cast(tf.reduce_sum(tf.round(tf.clip_by_value(y_true * y_pred, 0, 1))), tf.float32)
    possible_positives = tf.cast(tf.reduce_sum(tf.round(tf.clip_by_value(y_true, 0, 1))), tf.float32)
    sensitivity = true_positives / (possible_positives + 1.0e-7)
    """
    return sensitivity



# def aupr(targets, predictions):
#     aupr_array = []
        
#     for i in range(targets.shape[1]):        
#         precision, recall, _ = precision_recall_curve(targets[:,i], predictions[:,i], 
#                                                       pos_label=1)
        
#         auPR = auc(recall, precision)
#         if not math.isnan(auPR):
#             aupr_array.append(np.nan_to_num(auPR))
       
    
#     aupr_array = np.array(aupr_array)
#     mean = np.mean(aupr_array)
#     median = np.median(aupr_array)
#     var = np.var(aupr_array)
    
#     return (mean, median, var), aupr_array

# def auroc(targets, predictions):
#     auroc_array = []
#     for i in range(targets.shape[1]):        
#         auroc = roc_auc_score(targets[:,i], predictions[:,i])
#         auroc_array.append(auroc)
    
#     auroc_array = np.array(auroc_array)
#     mean = np.mean(auroc_array)
#     median = np.median(auroc_array)
#     var = np.var(auroc_array)
    
#     return (mean, median, var), auroc_array


# def evaluate_model(model, X_test, Y_test):
#     print(np.mean(model.call(X_test).numpy()))
    
#     _, aupr_array = aupr(Y_test, model.call(X_test))
#     _, auroc_array = auroc(Y_test, model.call(X_test))
#     _, test_accuracy = model.evaluate(X_test, Y_test)
#     return {
#         'accuracy': test_accuracy,
#         'aupr': aupr_array[0],
#         'auroc': auroc_array[0]
#     } 

def w_categorical_crossentropy(y_true, y_pred, weights):
    """https://www.programcreek.com/python/example/93764/keras.backend.categorical_crossentropy
    Keras-style categorical crossentropy loss function, with weighting for each class.
    Parameters
    ----------
    y_true : Tensor
        Truth labels one hot encoded
    y_pred : Tensor
        Predicted values one hot encoded
    weights: Tensor
        Multiplicative factor for loss per class.
    Returns
    -------
    loss : Tensor
        Weighted crossentropy loss between labels and predictions.
    """
    y_true_max = tf.argmax(y_true, axis=-1)
    weighted_true = tf.gather(weights, y_true_max)
    loss = tf.keras.metrics.categorical_crossentropy(y_pred, y_true) * weighted_true
    return loss 
    

        
if __name__ == '__main__':
    true = np.array([[1,0],[0,1]])
    pred = np.array([[0.9,0.1],[0,1]])
    print(specificity(true, pred))
    print(sensitivity(true,pred))

