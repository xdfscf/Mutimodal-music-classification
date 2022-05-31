import numpy as np
import os
import random
from PIL import Image
from sklearn.preprocessing import OneHotEncoder
import math
import pandas as pd
import metadata as meta

spectr_template = './in/mel-specs/{}'

idx_map = dict()


class MultiOutputDataset:
    """
    First layer of dataset class. The idea is to
    "encapsulate" logic inside each of data splits.
    The dataset output vector is not one-hot vector;
    it has multiple non-zero elements - that's why
    it's 'multi output dataset'.
    """

    def __init__(self, train_x, train_y, valid_x, valid_y, test_x, test_y):

        def _create_indices_mapping(train_y_all, valid_y_all, test_y_all):
            indices = []
            for genre_list in np.hstack((train_y_all, valid_y_all, test_y_all)):
                for genre_id in genre_list:
                    if genre_id not in indices:
                        indices.append(genre_id)
            indices = sorted(indices)
            for i, index in enumerate(indices):
                idx_map[index] = i

        _create_indices_mapping(train_y[1], valid_y[1], test_y[1])
        self.train = SplitData(train_x, train_y[0], train_y[1], 'train')
        self.valid = SplitData(valid_x, valid_y[0], valid_y[1], 'valid')
        self.test = SplitData(test_x, test_y[0], test_y[1], 'test')


class SplitData:
    """
    Inner layer that does the actual spectrogram images fetching
    for each batch and assigns expected values to output vector y.
    """

    def __init__(self, track_ids, y_top, y_all, dataset_label):
        self.top=y_top
        self.top_genre_significance = 0.75
        self.current_sample_idx = 0
        self.track_ids = track_ids
        self.labels = self._create_output_vector(y_top, y_all)
        self.dataset_label = dataset_label

    def _create_output_vector(self, y_top, y_all):
        """
        Instead of typical one-hot vector, a vector with
        more than single non-zero element is created for
        multi-output classification.
        For top_genre, top_genre_significance is assigned.
        For the rest of genres, if available,
        (1 - top_genre_significance) is evenly assigned.
        :param y_top:
        :param y_all:
        :return y:
        """

        # dim(y) = (number_of_samples, number_of_unique_indices)
        y = []
        vsize = len(idx_map)
        for i in range(y_top.shape[0]):
            yi = [0] * vsize
            if len(y_all[i]) == 1:
                yi[idx_map[y_top[i]]] = 1
            else:
                yi[idx_map[y_top[i]]] = self.top_genre_significance

                other_genres_significance = (1 - self.top_genre_significance) / (len(y_all[i]) - 1)
                for genre_id in y_all[i]:
                    if genre_id == y_top[i]:
                        continue
                    yi[idx_map[genre_id]] = other_genres_significance
            y.append(yi)

        return np.array(y)

    def _load_images(self, track_ids):
        """
        Private method for actual loading spectrogram data.
        :param track_ids:
        :return images:
        """
        images = []
        for track_id in track_ids:
            fpath = spectr_template.format(track_id[:3] + '/' + track_id + '.png')
            print('Loading spectrogram: {} ({})'.format(fpath, self.dataset_label))
            images.append(np.asarray(Image.open(fpath).getdata()).reshape((128,646)))

        return np.array(images)

    def load_all(self):
        """
        Returns all the data with loaded spectrograms.
        :return images, labels:
        """
        return self._load_images(self.track_ids), self.labels

    def load_imgae_with_onehot_labels(self):


        # creating instance of one-hot-encoder
        encoder = OneHotEncoder(handle_unknown='ignore')

        # perform one-hot encoding on 'team' column
        encoder_df = encoder.fit_transform(pd.DataFrame(self.top)).toarray()
        return self._load_images(self.track_ids),encoder_df
    def get_number_of_batches(self, batch_size):
        """
        :return number_of_batches:
        """
        return int(math.ceil(self.track_ids.shape[0] / batch_size))

    def get_output_size(self):
        """
        Returns label vector length, i.e. the number of classes.
        :return:
        """
        return self.labels.shape[1]

    def next_batch(self, batch_size):
        """
        Takes subset of input and output for interval
        (current_idx : current_idx + batch_size).
        :param batch_size:
        :return batch_images, batch_labels:
        """
        if self.current_sample_idx + batch_size >= self.track_ids.shape[0]:  # edge case when latter index is overflown
            filling_ids = random.sample(range(self.current_sample_idx),
                                        batch_size - (self.track_ids.shape[0] - self.current_sample_idx))
            batch_images = self._load_images(
                self.track_ids[list(range(self.current_sample_idx, self.track_ids.shape[0])) + filling_ids])
            batch_labels = self.labels[list(range(self.current_sample_idx, self.labels.shape[0])) + filling_ids]
            self.current_sample_idx = 0
        else:
            batch_images = self._load_images(
                self.track_ids[self.current_sample_idx:self.current_sample_idx + batch_size])
            batch_labels = self.labels[self.current_sample_idx:self.current_sample_idx + batch_size]
            self.current_sample_idx += batch_size

        return batch_images, batch_labels

    def shuffle(self):
        indices = np.arange(self.track_ids.shape[0])
        np.random.shuffle(indices)

        self.track_ids = self.track_ids[indices]
        self.labels = self.labels[indices]


def get_data():
    """
    Reads metadata, stacks input and output to a single object.
    Returns complex object that contain splitted set objects.
    X vectors contain track ids, not spectrograms, that is why
    they are prefixed with 'meta_' prefix.
    Y vectors are structured as ([[top_genre], [all_genres]]).
    :return Dataset(train, test, valid):
    """

    def _clean_track_ids(track_ids, labels):
        """
        Some spectrogram images might be missing as they
        failed to generate so dimensions wouldn't match
        if regular np.hstack is used. This function removes
        rows that would contain missing spectrograms.
        :param images:
        :param y_stack:
        :return:
        """
        all_cnt = 0
        dlt_cnt = 0
        for idx, track_id in enumerate(track_ids):
            track_id_str = str(track_id)
            all_cnt += 1

            if not os.path.isfile(spectr_template.format(track_id_str[:3] + '/' + track_id_str + '.png')):
                print(spectr_template.format(track_id_str[:3] + '/' + track_id_str + '.png'))
                track_ids = np.delete(track_ids, idx - dlt_cnt, 0)
                labels = np.delete(labels, idx - dlt_cnt, 1)
                dlt_cnt += 1

        return track_ids, labels, all_cnt, dlt_cnt

    meta_train_x, train_y, meta_valid_x, valid_y, meta_test_x, test_y = meta.get_metadata()

    meta_train_x, train_y, all_cnt, dlt_cnt = _clean_track_ids(meta_train_x, train_y)
    print('Removed {} of {} train records => {}'.format(dlt_cnt, all_cnt, all_cnt - dlt_cnt))
    meta_test_x, test_y, all_cnt, dlt_cnt = _clean_track_ids(meta_test_x, test_y)
    print('Removed {} of {} test records => {}'.format(dlt_cnt, all_cnt, all_cnt - dlt_cnt))
    meta_valid_x, valid_y, all_cnt, dlt_cnt = _clean_track_ids(meta_valid_x, valid_y)
    print('Removed {} of {} validation records => {}'.format(dlt_cnt, all_cnt, all_cnt - dlt_cnt))

    return MultiOutputDataset(meta_train_x, train_y, meta_test_x, test_y, meta_valid_x, valid_y)


if __name__ == '__main__':
    """
    Used for testing and debugging.
    """
    data = get_data()
    batch_size = 100
    for i in range(100):
        #batch = data.test.next_batch(batch_size)
        pass

    print(data.test.track_ids)
    data.test.shuffle()
    print(data.test.track_ids)

    data.test.all_loaded()