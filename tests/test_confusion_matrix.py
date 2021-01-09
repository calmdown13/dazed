from dataclasses import dataclass

import numpy as np
import pandas as pd
import sklearn.metrics

import dazed.confusion_matrix as CM


@dataclass
class SingleLabelData:
    y1 = [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3]
    y2 = [0, 1, 2, 3, 1, 1, 3, 3, 1, 1, 2, 2, 1, 1, 2, 2]
    labels = [0, 1, 2, 3, 4]
    info = [
        "a",
        "b",
        "c",
        "d",
        "a",
        "b",
        "c",
        "d",
        "a",
        "b",
        "c",
        "d",
        "a",
        "b",
        "c",
        "d",
    ]
    sparse_matrix = sklearn.metrics.confusion_matrix(y1, y2)
    matrix = sklearn.metrics.confusion_matrix(y1, y2, labels=labels)
    bins = {
        (0, 1): ["a", "a", "a"],
        (1, 2): [],
        (3, 2): ["d", "d"],
    }
    most_confused = [(0, 1, 3), (3, 2, 2), (2, 3, 1)]


def multi_to_single(y, combinations):
    single = []
    for y_i in y:
        y_i_str = ", ".join([str(i) for i in y_i])
        single.append(combinations.index(y_i_str))
    return single


@dataclass
class MultiLabelData:
    y1 = [
        [],
        [0],
        [1],
        [0, 1],
        [],
        [0],
        [1],
        [0, 1],
        [],
        [0],
        [1],
        [0, 1],
        [],
        [0],
        [1],
        [0, 1],
    ]
    y2 = [
        [],
        [0],
        [1],
        [0, 1],
        [0],
        [0],
        [0, 1],
        [0, 1],
        [0],
        [0],
        [1],
        [1],
        [0],
        [0],
        [1],
        [1],
    ]
    info = [
        "a",
        "b",
        "c",
        "d",
        "a",
        "b",
        "c",
        "d",
        "a",
        "b",
        "c",
        "d",
        "a",
        "b",
        "c",
        "d",
    ]
    labels = [0, 1]
    all_combinations = CM._all_combinations(labels)
    y1m = multi_to_single(y1, all_combinations)
    y2m = multi_to_single(y2, all_combinations)
    sparse_matrix = sklearn.metrics.confusion_matrix(
        multi_to_single(y1, all_combinations), multi_to_single(y2, all_combinations)
    )
    matrix = sklearn.metrics.confusion_matrix(
        multi_to_single(y1, all_combinations),
        multi_to_single(y2, all_combinations),
        labels=list(range(len(all_combinations))),
    )
    bins = {
        ("", "0"): ["a", "a", "a"],
        ("0", "1"): [],
        ("0, 1", "1"): ["d", "d"],
    }
    most_confused = [("", "0", 3), ("0, 1", "1", 2), ("1", "0, 1", 1)]


def check_confusion_matrix_values(
    data,
    confusion_matrix,
    skip_matrix_check=False,
    skip_bin_check=False,
    skip_most_confused_check=False,
):
    if not skip_matrix_check:
        sparse_matrix, labels = confusion_matrix.as_array(present_only=True)
        matrix, labels = confusion_matrix.as_array(present_only=False)
        np.testing.assert_array_equal(matrix, data.matrix)
        np.testing.assert_array_equal(sparse_matrix, data.sparse_matrix)
    if not skip_bin_check:
        for bin_labels, bin_info in data.bins.items():
            assert confusion_matrix.get_bin(*bin_labels) == bin_info
    if not skip_most_confused_check:
        assert confusion_matrix.most_confused() == data.most_confused


def test_init_list_array():
    h, w = 4, 5
    val_type = list
    la = CM._init_list_array(h, w, val_type)
    assert len(la) == h
    assert len(la[0]) == w
    assert type(la[0][0]) == val_type


def test_init_grid_coords():
    h, w = 4, 5
    i, j = CM._init_grid_coords(h, w)
    assert len(i) == len(j) == h * w
    assert max(i) == h - 1
    assert max(j) == w - 1
    assert i[0] == 0 and j[0] == 0
    assert i[-1] == h - 1 and j[-1] == w - 1


def test_all_combinations_int():
    assert CM._all_combinations([0, 1]) == ["", "0", "0, 1", "1"]
    assert CM._all_combinations([1, 0]) == ["", "0", "1", "1, 0"]


def test_all_combinations_str():
    assert CM._all_combinations(["a", "b"]) == ["", "a", "a, b", "b"]
    assert CM._all_combinations(["b", "a"]) == ["", "a", "b", "b, a"]


def test_onehot_to_sparse_singlelabel_int():
    data = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]])
    assert CM._onehot_to_sparse(data) == [2, 1, 0]


def test_onehot_to_sparse_singlelabel_bool():
    data = np.array([[False, False, True], [False, True, False], [True, False, False]])
    assert CM._onehot_to_sparse(data) == [2, 1, 0]


def test_onehot_to_sparse_singlelabel_int_raise_error():
    data = np.array([[0, 0, 0], [0, 0, 1], [1, 0, 1]])
    try:
        CM._onehot_to_sparse(data)
        assert False
    except ValueError as e:
        pass
    else:
        assert False


def test_onehot_to_sparse_singlelabel_bool_raise_error():
    data = np.array([[False, False, False], [False, False, True], [True, False, True]])
    try:
        CM._onehot_to_sparse(data)
        assert False
    except ValueError as e:
        pass
    else:
        assert False


def test_onehot_to_sparse_multilabel_int():
    data = np.array([[0, 0, 0], [0, 0, 1], [1, 0, 1]])
    assert CM._onehot_to_sparse(data, multilabel=True) == [[], [2], [0, 2]]


def test_onehot_to_sparse_multilabel_bool():
    data = np.array([[False, False, False], [False, False, True], [True, False, True]])
    assert CM._onehot_to_sparse(data, multilabel=True) == [[], [2], [0, 2]]


def test_ConfusionMatrix_no_labels():
    data = SingleLabelData()
    cm = CM.ConfusionMatrix(data.y1, data.y2, info=data.info)
    data.matrix = data.sparse_matrix  # matrix is sparse as labels are inferred
    check_confusion_matrix_values(data, cm)


def test_ConfusionMatrix_from_df_no_labels_singlelabel():
    data = SingleLabelData()
    df = pd.DataFrame({"y1": data.y1, "y2": data.y2, "info": data.info})
    cm = CM.ConfusionMatrix.from_df(df, "y1", "y2", info_names=["info"])
    data.matrix = data.sparse_matrix  # matrix is sparse as labels are inferred
    check_confusion_matrix_values(data, cm, skip_bin_check=True)
    for bin_labels, bin_info in data.bins.items():
        df = pd.DataFrame(cm.get_bin(*bin_labels), columns=["info"])
        assert df["info"].tolist() == bin_info


def test_ConfusionMatrix_from_df_singlelabel():
    data = SingleLabelData()
    df = pd.DataFrame({"y1": data.y1, "y2": data.y2, "info": data.info})
    cm = CM.ConfusionMatrix.from_df(
        df, "y1", "y2", labels=data.labels, info_names=["info"]
    )
    check_confusion_matrix_values(data, cm, skip_bin_check=True)
    for bin_labels, bin_info in data.bins.items():
        df = pd.DataFrame(cm.get_bin(*bin_labels), columns=["info"])
        assert df["info"].tolist() == bin_info


def test_ConfusionMatrix_from_onehot_no_labels_singlelabel():
    data = SingleLabelData()
    y1_onehot = np.zeros((len(data.y1), len(data.labels)))
    y1_onehot[list(range(len(data.y1))), data.y1] = 1
    y2_onehot = np.zeros((len(data.y2), len(data.labels)))
    y2_onehot[list(range(len(data.y2))), data.y2] = 1
    cm = CM.ConfusionMatrix.from_onehot(y1_onehot, y2_onehot, info=data.info)
    data.matrix = data.sparse_matrix  # matrix is sparse as labels are inferred
    check_confusion_matrix_values(data, cm)


def test_ConfusionMatrix_from_onehot_no_labels_multilabel():
    data = MultiLabelData()
    y1_onehot = np.zeros((len(data.y1), len(data.labels)))
    y2_onehot = np.zeros((len(data.y2), len(data.labels)))
    for i, (y1i, y2i) in enumerate(zip(data.y1, data.y2)):
        for y1il in y1i:
            y1_onehot[i, y1il] = 1
        for y2il in y2i:
            y2_onehot[i, y2il] = 1
    cm = CM.ConfusionMatrix.from_onehot(
        y1_onehot, y2_onehot, info=data.info, multilabel=True
    )
    data.matrix = data.sparse_matrix  # matrix is sparse as labels are inferred
    check_confusion_matrix_values(data, cm)


def test_ConfusionMatrix_from_onehot_singlelabel():
    data = SingleLabelData()
    y1_onehot = np.zeros((len(data.y1), len(data.labels)))
    y1_onehot[list(range(len(data.y1))), data.y1] = 1
    y2_onehot = np.zeros((len(data.y2), len(data.labels)))
    y2_onehot[list(range(len(data.y2))), data.y2] = 1
    cm = CM.ConfusionMatrix.from_onehot(
        y1_onehot, y2_onehot, labels=data.labels, info=data.info
    )
    check_confusion_matrix_values(data, cm)


def test_ConfusionMatrix_from_onehot_multilabel():
    data = MultiLabelData()
    y1_onehot = np.zeros((len(data.y1), len(data.labels)))
    y2_onehot = np.zeros((len(data.y2), len(data.labels)))
    for i, (y1i, y2i) in enumerate(zip(data.y1, data.y2)):
        for y1il in y1i:
            y1_onehot[i, y1il] = 1
        for y2il in y2i:
            y2_onehot[i, y2il] = 1
    cm = CM.ConfusionMatrix.from_onehot(
        y1_onehot, y2_onehot, labels=data.labels, info=data.info, multilabel=True
    )
    check_confusion_matrix_values(data, cm)


def test_ConfusionMatrix_from_onehot_string_labels_singlelabel():
    data = SingleLabelData()
    y1_onehot = np.zeros((len(data.y1), len(data.labels)))
    y1_onehot[list(range(len(data.y1))), data.y1] = 1
    y2_onehot = np.zeros((len(data.y2), len(data.labels)))
    y2_onehot[list(range(len(data.y2))), data.y2] = 1
    data.labels = [str(l) for l in data.labels]
    data.bins = {(str(k1), str(k2)): v for (k1, k2), v in data.bins.items()}
    data.most_confused = [(str(v1), str(v2), v3) for v1, v2, v3 in data.most_confused]
    cm = CM.ConfusionMatrix.from_onehot(
        y1_onehot, y2_onehot, labels=data.labels, info=data.info
    )
    check_confusion_matrix_values(data, cm)


def test_ConfusionMatrix_from_onehot_string_labels_multilabel():
    data = MultiLabelData()
    y1_onehot = np.zeros((len(data.y1), len(data.labels)))
    y2_onehot = np.zeros((len(data.y2), len(data.labels)))
    for i, (y1i, y2i) in enumerate(zip(data.y1, data.y2)):
        for y1il in y1i:
            y1_onehot[i, y1il] = 1
        for y2il in y2i:
            y2_onehot[i, y2il] = 1
    data.labels = [str(l) for l in data.labels]
    data.bins = {(str(k1), str(k2)): v for (k1, k2), v in data.bins.items()}
    data.most_confused = [(str(v1), str(v2), v3) for v1, v2, v3 in data.most_confused]
    cm = CM.ConfusionMatrix.from_onehot(
        y1_onehot, y2_onehot, labels=data.labels, info=data.info, multilabel=True
    )
    check_confusion_matrix_values(data, cm)


def test_ConfusionMatrix_from_sparse_no_labels_singlelabel():
    data = SingleLabelData()
    cm = CM.ConfusionMatrix.from_sparse(data.y1, data.y2, info=data.info)
    data.matrix = data.sparse_matrix  # matrix is sparse as labels are inferred
    check_confusion_matrix_values(data, cm)


def test_ConfusionMatrix_from_sparse_no_labels_multilabel():
    data = MultiLabelData()
    cm = CM.ConfusionMatrix.from_sparse(
        data.y1, data.y2, info=data.info, multilabel=True
    )
    data.matrix = data.sparse_matrix  # matrix is sparse as labels are inferred
    data.bins = {(str(k1), str(k2)): v for (k1, k2), v in data.bins.items()}
    check_confusion_matrix_values(data, cm)


def test_ConfusionMatrix_from_sparse_singlelabel():
    data = SingleLabelData()
    cm = CM.ConfusionMatrix.from_sparse(
        data.y1, data.y2, labels=data.labels, info=data.info
    )
    check_confusion_matrix_values(data, cm)


def test_ConfusionMatrix_from_sparse_multilabel():
    data = MultiLabelData()
    cm = CM.ConfusionMatrix.from_sparse(
        data.y1, data.y2, labels=data.labels, info=data.info, multilabel=True
    )
    data.bins = {(str(k1), str(k2)): v for (k1, k2), v in data.bins.items()}
    check_confusion_matrix_values(data, cm)
