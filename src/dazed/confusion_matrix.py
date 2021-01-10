from itertools import combinations

import numpy as np
import pandas as pd


def _init_list_array(size1, size2, val_type):
    lists = []
    for i in range(size1):
        lists.append([])
        for _ in range(size2):
            lists[i].append(val_type())
    return lists


def _init_grid_coords(size1, size2):
    ix, jx = np.meshgrid(range(size1), range(size2))
    ix, jx = ix.flatten(), jx.flatten()
    return ix, jx


def _onehot_to_sparse(y, labels=None, multilabel=False):
    num_columns = y.shape[1]
    if not multilabel:
        if np.any(np.sum(y, axis=1) > 1):
            raise ValueError
        sparse = np.argmax(y, axis=1).tolist()
    else:
        sparse = []
        for row in y:
            sparse.append(np.where(row)[0].tolist())
    if labels is not None:
        assert len(labels) == num_columns
        sparse = indices_to_labels(sparse, labels)
    return sparse


def _all_combinations(labels):
    num_labels = len(labels)
    string_labels = [str(label) for label in labels]
    string_labels.sort()
    label_combinations = [""]
    for length in range(1, num_labels + 1):
        label_combinations.extend(
            [", ".join(c) for c in combinations(string_labels, length)]
        )
    label_combinations.sort()
    return label_combinations


def indices_to_labels(y, labels):
    y_ret = []
    for yi in y:
        if isinstance(yi, list):
            yi_ret = []
            for yii in yi:
                yi_ret.append(labels[yii])
            y_ret.append(yi_ret)
        else:
            y_ret.append(labels[yi])
    return y_ret


def _get_matrix_string(matrix):
    h, w = matrix.shape
    max_index_len = _str_len(h)
    max_val_len = max(_str_len(np.amax(matrix)), _str_len(w))
    rows = ["\n"]
    rows.append(_get_row_string("", max_index_len, [i for i in range(w)], max_val_len))
    rows.append(_get_dashed_line(max_index_len, max_val_len, w))
    for i in range(h):
        rows.append(_get_row_string(i, max_index_len, matrix[i], max_val_len))
    rows.append(_get_dashed_line(max_index_len, max_val_len, w))
    return rows


def _get_key_string(index_to_label):
    max_index_len = max(len("index"), max([_str_len(k) for k in index_to_label]))
    max_val_len = max(len("label"), max([_str_len(v) for v in index_to_label.values()]))
    rows = ["\n"]
    rows.append(_get_row_string("index", max_index_len, ["label"], max_val_len))
    rows.append(_get_dashed_line(max_index_len, max_val_len, 1))
    for k, v in index_to_label.items():
        rows.append(_get_row_string(k, max_index_len, [v], max_val_len))
    rows.append(_get_dashed_line(max_index_len, max_val_len, 1))
    return rows


def _get_dashed_line(max_index_len, max_val_len, num_vals):
    return "-" * (max_index_len + 2 + (max_val_len + 1) * num_vals) + " \n"


def _get_row_string(index, max_index_len, vals, max_val_len):
    index_len = _str_len(index)
    strings = [f"{' '*(max_index_len-index_len)}{index} |"]
    for val in vals:
        val_len = _str_len(val)
        strings.append(f"{' '*(max_val_len-val_len)}{val}")
    strings.append("\n")
    return " ".join(strings)


def _str_len(integer):
    return len(str(integer))


class ConfusionMatrix:
    def __init__(self, y1, y2, labels=None, info=None):
        self._y1 = y1
        self._y2 = y2

        # labels
        self._sparse_labels = np.unique(y1 + y2)
        if labels is None:
            self._labels = self._sparse_labels
        else:
            self._labels = labels

        # bin info
        if info is None:
            info = [None for _ in range(len(y1))]

        # label maps
        (
            self._label_to_sparse_index,
            self._sparse_index_to_label,
        ) = self._create_label_maps(self._sparse_labels)
        self._label_to_index, self._index_to_label = self._create_label_maps(
            self._labels
        )

        # cm and info_lists
        self._sparse_matrix = self._create_matrix(y1, y2, self._label_to_sparse_index)
        self._sparse_info_lists = self._create_info_lists(
            y1, y2, info, self._label_to_sparse_index
        )

    @classmethod
    def from_df(
        cls, df, y1_names, y2_names, labels=None, info_names=None, multilabel=False
    ):
        assert type(y1_names) == type(y2_names)
        if info_names is None:
            info = None
        else:
            info = [row for _, row in df[info_names].iterrows()]
        if isinstance(y1_names, str):
            return cls.from_sparse(
                df[y1_names].tolist(),
                df[y2_names].tolist(),
                labels=labels,
                info=info,
                multilabel=multilabel,
            )
        elif isinstance(y1_names, list):
            return cls.from_onehot(
                df[y1_names].values,
                df[y2_names].values,
                labels=labels,
                info=info,
                multilabel=multilabel,
            )
        else:
            raise ValueError

    @classmethod
    def from_sparse(cls, y1, y2, labels=None, info=None, multilabel=False):
        if multilabel:
            if labels is not None:
                labels = _all_combinations(labels)
            y1 = [", ".join([str(i) for i in yi]) for yi in y1]
            y2 = [", ".join([str(i) for i in yi]) for yi in y2]
        return cls(y1, y2, labels=labels, info=info)

    @classmethod
    def from_onehot(cls, y1, y2, labels=None, info=None, multilabel=False):
        y1 = _onehot_to_sparse(y1, labels, multilabel)
        y2 = _onehot_to_sparse(y2, labels, multilabel)
        if multilabel:
            if labels is not None:
                labels = _all_combinations(labels)
            y1 = [", ".join([str(i) for i in yi]) for yi in y1]
            y2 = [", ".join([str(i) for i in yi]) for yi in y2]
        return cls(y1, y2, labels=labels, info=info)

    @staticmethod
    def _create_label_maps(labels):
        label_to_index = {}
        index_to_label = {}
        for i, l in enumerate(labels):
            label_to_index[l] = i
            index_to_label[i] = l
        return label_to_index, index_to_label

    @staticmethod
    def _create_matrix(y1, y2, label_to_index):
        cm = np.zeros((len(label_to_index), len(label_to_index)), dtype=int)
        for y1_i, y2_i in zip(y1, y2):
            cm[label_to_index[y1_i], label_to_index[y2_i]] += 1
        return cm

    @staticmethod
    def _create_info_lists(y1, y2, info, label_to_index):
        info_lists = _init_list_array(len(label_to_index), len(label_to_index), list)
        for y1_i, y2_i, info_i in zip(y1, y2, info):
            info_lists[label_to_index[y1_i]][label_to_index[y2_i]].append(info_i)
        return info_lists

    def as_array(self, present_only=True):
        if present_only:
            return self._sparse_matrix, self._sparse_labels
        else:
            return (
                self._create_matrix(self._y1, self._y2, self._label_to_index),
                self._labels,
            )

    def as_df(self, present_only=True):
        matrix, labels = self.as_array(present_only=present_only)
        return pd.DataFrame(matrix, index=labels, columns=labels)

    def as_str(self, present_only=True):
        if present_only:
            matrix = _get_matrix_string(self._sparse_matrix)
            key = _get_key_string(self._sparse_index_to_label)
        else:
            matrix = _get_matrix_string(self.as_array(present_only=present_only)[0])
            key = _get_key_string(self._index_to_label)
        return "".join([m[:-2] + "     " + k for m, k in zip(matrix, key)])

    def label_pair_info(self, label_1, label_2):
        try:
            i = self._label_to_sparse_index[label_1]
            j = self._label_to_sparse_index[label_2]
            return self._sparse_info_lists[i][j]
        except KeyError:
            if label_1 in self._label_to_index and label_2 in self._label_to_index:
                return []
            else:
                raise ValueError

    def most_confused(self):
        num_labels = len(self._sparse_labels)
        ix, jx = _init_grid_coords(num_labels, num_labels)
        matrix = self._sparse_matrix.copy()
        np.fill_diagonal(matrix, 0)
        flat_matrix = matrix.flatten()
        indices = np.flip(np.argsort(flat_matrix))
        ret = []
        for i in indices:
            count = flat_matrix[i]
            if count > 0:
                ret.append(
                    (
                        self._sparse_index_to_label[jx[i]],
                        self._sparse_index_to_label[ix[i]],
                        count,
                    )
                )
            else:
                break
        return ret

    def __str__(self):
        return self.as_str()

    def __repr__(self):
        return self.as_str()
