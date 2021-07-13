# https: // gist.github.com/markloyman/61653179352aaaebe10183de52db0264

def l2_distance_matrix(x, exact=True):
    """
    Calculate an L2 distance matrix from embedding vectors)
    :param
        x: embedding tensor size (batch_size, embedding_length)
        exact: if False, skip the final sqrt
    :return: l2 distance matrix tensor tensor size (batch_size, batch_size)
    """
    r = K.sum(x*x, 1)
    r = K.reshape(r, [-1, 1])  # turn r into column vector
    dm = r - 2*K.dot(x, K.transpose(x)) + K.transpose(r)
    if exact:
        dm = K.sqrt(dm)
    return dm


def cosine_distance_matrix(x):
    """
    Calculate a Cosine distance matrix from embedding vectors)
    :param
        x: embedding tensor size (batch_size, embedding_length)
    :return: cosine distance matrix tensor tensor size (batch_size, batch_size)
    """
    return K.dot(x, K.transpose(x))


def pearson_correlation_of_embeddings(y_true, y_pred, distance_matrix_function, normalized=False):
    """
     Calculate pearson correlation loss   
    :param y_true: embedding tensor size (batch_size, embedding_length)
    :param y_pred: embedding tensor size (batch_size, embedding_length)
    :param distance_matrix_function: callable that calculates a distance matrix from an embedding
        (batch_size, embedding_length) -> (batch_size, batch_size)
    :param normalized: if True, Softmax is applied to the distance matrix
    :return: loss tensor
    """
    dm_true = distance_matrix_function(y_true)
    dm_pred = distance_matrix_function(y_pred)
    return pearson_correlation_of_distance_matrix(dm_true, dm_pred, normalized)


def pearson_correlation_of_distance_matrix(y_true, y_pred, normalized=False):
    """
     Calculate pearson correlation loss   
    :param y_true: distance matrix tensor tensor size (batch_size, batch_size)
    :param y_pred: distance matrix tensor tensor size (batch_size, batch_size)
    :param normalized: if True, Softmax is applied to the distance matrix
    :return: loss tensor
    """
    if normalized:
        y_true = K.softmax(y_true, axis=-1)
        y_pred = K.softmax(y_pred, axis=-1)

    sum_true = K.sum(y_true, axis=1)
    sum2_true = K.sum(K.square(y_true), axis=1)

    sum_pred = K.sum(y_pred, axis=1)
    sum2_pred = K.sum(K.square(y_pred), axis=1)

    prod = K.sum(y_true*y_pred, axis=1)
    n = K.cast(K.shape(y_true)[0], K.floatx())

    corr = n*prod - sum_true*sum_pred
    corr /= K.sqrt(n * sum2_true - sum_true * sum_true + K.epsilon())
    corr /= K.sqrt(n * sum2_pred - sum_pred * sum_pred + K.epsilon())

    return -corr
