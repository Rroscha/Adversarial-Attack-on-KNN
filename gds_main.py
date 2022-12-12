'''
 * @author Rroscha
 * @date 2022/12/12 16:10
'''

import logging
import pickle
import pprint
import time
from copy import deepcopy

import faiss
import numpy as np
import torch
import scipy.stats as st

from lib.dknn import DKNNL2, KNNModel
from lib.utils.utils import get_logger
from lib.loaders import initialize_data
from lib.dknn_attack_v2 import DKNNAttackV2
from AttackModel.gds import AttackModel


class Main:
    @staticmethod
    def attack_in_batch(x, y, batch_size, mode, gep, attack, sc=1):
        x_adv = torch.zeros_like(x)
        total_num = x.size(0)  # x's number
        num_batches = int(np.ceil(total_num / batch_size))
        sw_params = {
            'm': gep['k'] * 2,  # m = 2*k
            'guide_layer': ['identity'],
            'binary_search_steps': int(5 * scale),
            'max_linf': None,
            'initial_const': 1e-1,  # c = 0.1
            'random_start': True,
            'verbose': False,
        }
        for i in range(num_batches):
            begin, end = i * batch_size, (i + 1) * batch_size
            if mode == 1:
                x_adv[begin:end] = attack(x[begin:end], y[begin:end], 2,
                                          init_mode=1,
                                          init_mode_k=1,
                                          max_iterations=int(1000 * scale),
                                          learning_rate=1e-2,
                                          thres_steps=int(100 / scale),
                                          check_adv_steps=int(200 / scale),
                                          **sw_params)
            else:
                x_adv[begin:end] = attack(x[begin:end], y[begin:end], 2,
                                          init_mode=2,
                                          init_mode_k=gep['k'],
                                          max_iterations=int(2000 * scale),
                                          learning_rate=1e-1,
                                          thres_steps=int(50 / scale),
                                          check_adv_steps=int(50 / scale),
                                          **sw_params)
        return x_adv

    @staticmethod
    def get_precise_label(points, labels, input, k, num_classes):
        """
        Use this method to get the final prediction when `input` is close to or on
        multiple bisectors. Normal k-NN classifiers can be ambiguous in this case.
        Specifically, we split neighbors into two groups: (1) "sure" = unambiguous
        neighbors, well closer to input than the k-th neighbor, (2) "close" =
        ambiguous neighbors that are about the same distance away as the k-th
        neighbor. These "close" neighbors usually border one another including the
        k-th neighbor. The final prediction includes all labels that are possible
        given any combination of the neighbors.
        """
        TOL = 1e-6

        dist = np.sum((input - points) ** 2, 1)
        # Find distance to the kth neighbor
        k_dist = np.sort(dist)[k - 1]
        indices = np.where(dist - k_dist < TOL)[0]

        # Splitting neighbors into sure and close groups
        close_indices = np.where(np.abs(dist - k_dist) < TOL)[0]
        sure_indices = np.setdiff1d(indices, close_indices)
        close_labels = labels[close_indices]
        sure_labels = labels[sure_indices]
        close_counts = np.bincount(close_labels, minlength=num_classes)
        sure_counts = np.bincount(sure_labels, minlength=num_classes)

        num_to_fill = k - sure_counts.sum()
        # If number of sure counts is k, then we are done
        assert num_to_fill >= 0
        if num_to_fill == 0:
            max_count = sure_counts.max()
            return np.where(sure_counts == max_count)[0]

        y_pred = []
        for i in range(num_classes):
            # Fill class i as much as possible first
            num_fill = min(num_to_fill, close_counts[i])
            new_counts = deepcopy(sure_counts)
            new_counts[i] += num_fill
            close_counts_tmp = deepcopy(close_counts)
            # Fill the other classes in a way that benefits class i most
            while num_fill < num_to_fill:
                assert np.all(close_counts_tmp >= 0)
                # Get classes that can still be filled except for i
                ind = np.setdiff1d(np.where(close_counts_tmp > 0)[0], i)
                # Find class with the smallest count and add to it
                ind_to_fill = ind[new_counts[ind].argmin()]
                new_counts[ind_to_fill] += 1
                close_counts_tmp[ind_to_fill] -= 1
                num_fill += 1
            assert new_counts.sum() == k
            # Check if class i can be the prediction
            max_count = new_counts.max()
            if new_counts[i] == max_count:
                y_pred.append(i)
        return np.array(y_pred)

    @staticmethod
    def classify_correct(x_train, y_train, x_test, y_test, gc_params, num_classes):
        index = []

        for i in range(len(x_test)):
            label = Main.get_precise_label(
                x_train, y_train, x_test[i], gc_params['k'], num_classes)
            if y_test[i] in label and len(label) == 1:
                index.append(i)

        return index

    @staticmethod
    def attack_internal(glp, gep, sc):
        dist, adv_out, exit_code = [], [], []

        # Set logger
        logger_name = '%s_k%d_exp%d_gds' % (glp['dataset'], gep['k'], glp['exp'])
        log = get_logger(logger_name, level=glp['log_level'])
        log.setLevel(glp['log_level'])
        log.info('\n%s', pprint.pformat(glp))
        log.info('\n%s', pprint.pformat(gep))

        # Load data
        x_train, y_train, x_test, y_test = initialize_data(glp)
        x_train = x_train.astype(gep['dtype'])
        x_test = x_test.astype(gep['dtype'])

        num_test = glp['num_test']
        num_classes = len(np.unique(y_train))  # Get unique label number

        log.info('Training data shape: %s' % str(x_train.shape))
        log.info('Test data shape: %s' % str(x_test.shape))

        log.info('Gradient-based attack...')
        # Using KNN in DKNN, since it is implemented by PyTorch
        net_knn = KNNModel()
        knn = DKNNL2(net_knn,
                     torch.from_numpy(x_train), torch.from_numpy(y_train),
                     torch.from_numpy(x_test), torch.from_numpy(y_test),
                     ['identity'], k=gep['k'],
                     num_classes=num_classes,
                     device=gep['device'])

        attack = AttackModel(knn, gep, log)

        log.info('Finding correctly classified samples for attacking...')
        y_pred = knn.classify(torch.from_numpy(x_test[:num_test]))
        index = np.where(y_pred.argmax(1) == y_test[:num_test])[0]
        clean_acc = len(index) / num_test

        if num_test > len(index):
            num_test = len(index)

        log.info('Ready to attack...')
        start = time.time()

        log.info('Running GDS attack...')
        x_adv = Main.attack_in_batch(
            torch.from_numpy(x_test[index]).to(gep['device']),
            torch.from_numpy(y_test[index]).to(gep['device']),
            100, 1, gep, attack, sc=sc)

        log.info("Check which x_adv is attacked successfully...")
        index_correct = Main.classify_correct(
            x_train, y_train, x_adv.detach().cpu().numpy(), y_test[index],
            gep, num_classes)
        log.info('Success rate of attack: '
                 f'{(1 - len(index_correct) / num_test):.2f}')

        # runtime
        runtime = time.time() - start

        # exit_code
        exit_code = [1 if i not in index_correct else 0 for i in range(len(index))]

        # y_pred
        x_adv_ = np.array(x_adv.detach())
        for i in range(len(x_adv_)):
            y_pred = Main.get_precise_label(x_train, y_train, x_adv_[i], gep['k'], num_classes)

        # adv_out and dist
        adv_out = np.array(x_adv.detach())
        adv_out = [np.array(i) for i in adv_out]

        for i, z in zip(index, adv_out):
            dist.append(float(np.linalg.norm(x_test[i] - z)))

        return [dist, adv_out, exit_code, runtime, clean_acc]


    @staticmethod
    def attack(glp, gep, scale_, trials_):
        outputs = []

        counter = 0

        for i in range(trials_):
            output = None
            # global_parameters_['seed'] = np.random.randint(2147483647)

            output = Main.attack_internal(glp, gep, scale_)
            outputs = [glp['dataset'], output]

            counter += 1

        with open('./save/%s_geo_outputs_%d_gds_k%d.pkl' %
                  (glp['dataset'], glp['exp'], gep['k']), 'wb') as f:
            pickle.dump(outputs, f)



if __name__ == '__main__':
    global_parameters = {
        'exp': 1,
        'dataset_dir': './data/',
        'random': True,
        'seed': 1,

        # Do not use in our project
        'partial': False,
        'label_domain': (1, 7),

        'num_test': 20,  # The number of samples in test set you want to use

        'init_ub': True,  # Initialize upper bound for geo

        'log_level': logging.INFO,

        # Do not touch this
        'gaussian': {
            'dim': 20,
            'dist': 0.5,
            'sd': 1.,
            'num_points': 12500,
            'test_ratio': 0.2
        }
    }

    gep = {
        'k': 3,

        'dtype': np.float32,

        'device': 'cpu',

        'time_limit': 120,
    }

    v1 = Main()
    scale = 2
    trials = 1


    for k in [1, 3, 5, 7]:
        gep['k'] = k
        for dataset in ['australian',
                        'covtype',
                        'diabetes',
                        'fourclass',
                        'gaussian',
                        'yang-fmnist']:
            print("=================%s=================" % dataset)

            global_parameters['dataset'] = dataset
            v1.attack(global_parameters, gep, scale, trials)