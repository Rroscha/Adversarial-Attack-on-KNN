'''
 * @author Rroscha
 * @date 2022/12/11 21:55
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
from AttackModel.geo import AttackModel

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
        clean_acc = 0

        # Set logger
        logger_name = '%s_k%d_exp%d_geo' % (glp['dataset'], gep['k'], glp['exp'])
        log = get_logger(logger_name, level=glp['log_level'])
        log.setLevel(glp['log_level'])
        log.info('\n%s', pprint.pformat(glp))
        log.info('\n%s', pprint.pformat(gep))

        # Load data
        x_train, y_train, x_test, y_test = initialize_data(glp)
        x_train = x_train.astype(gep['dtype'])
        x_test  = x_test.astype(gep['dtype'])

        num_test = glp['num_test']
        num_classes = len(np.unique(y_train))  # Get unique label number

        log.info('Training data shape: %s' % str(x_train.shape))
        log.info('Test data shape: %s' % str(x_test.shape))

        # Set a tight upperbound, which accelerate attacking effectiveness
        log.info('Setting up a quick attack for computing loose upperbound...')

        # Using KNN in DKNN, since it is implemented by PyTorch
        net_knn = KNNModel()
        knn = DKNNL2(net_knn,
                     torch.from_numpy(x_train), torch.from_numpy(y_train),
                     torch.from_numpy(x_test), torch.from_numpy(y_test),
                     ['identity'], k=gep['k'],
                     num_classes=num_classes,
                     device=gep['device'])

        # Set gradient descent-version 2 attack
        attack = DKNNAttackV2(knn)

        log.info('Finding correctly classified samples for attacking...')
        y_pred = knn.classify(torch.from_numpy(x_test[:num_test]))
        index = np.where(y_pred.argmax(1) == y_test[:num_test])[0]
        clean_acc = len(index) / num_test

        if num_test > len(index):
            num_test = len(index)
        index = index[:num_test]

        log.info('Ready to attack...')
        start = time.time()
        if glp['init_ub']:
            log.info('Running GB-based version 2 attack...')
            x_adv = Main.attack_in_batch(
                torch.from_numpy(x_test[index]).to(gep['device']),
                torch.from_numpy(y_test[index]).to(gep['device']),
                100, 1, gep, attack, sc=sc)

            log.info("Check which x_adv is attacked successfully...")
            index_correct = Main.classify_correct(
                x_train, y_train, x_adv.detach().cpu().numpy(), y_test[index],
                gep, num_classes)

            log.info('Success rate of the heuristic attack (1): '
                     f'{(1 - len(index_correct) / num_test):.2f}')

            upperbound = np.linalg.norm(x_adv.detach().numpy() - x_test[index], 2, 1)
            upperbound[index_correct] = np.inf

            # Since GB-based version 2 attack cannot assure the 100% attack success rate, need a second round attack
            if len(index_correct) > 0:
                # mode 2 is to preset nearest adversarial sample as x_adv
                log.info('Running GB-based version 2 (mode 2) attack...')
                x_adv_2 = Main.attack_in_batch(
                            torch.from_numpy(x_test[index]).to(gep['device']),
                            torch.from_numpy(y_test[index]).to(gep['device']),
                            100, 2, gep, attack, sc=sc)

                log.info("Check whether x_adv_2 are all attacked successfully...")
                index_correct_2 = Main.classify_correct(
                    x_train, y_train, x_adv_2.detach().cpu().numpy(), y_test[index],
                    gep, num_classes)

                upperbound_2 = np.linalg.norm(x_adv_2.detach().numpy() - x_test[index], 2, 1)
                upperbound_2[index_correct_2] = np.inf

                index_2 = upperbound_2 < upperbound
                upperbound[index_2] = upperbound_2[index_2]
                x_adv[index_2] = x_adv_2[index_2]

            log.info(f'Founded upperbound: {upperbound}')


        log.info('Setting up Geo...')
        dim = x_train.shape[1]
        index_box = None
        if gep['index'] == 'flat':
            index_box = faiss.IndexFlatL2(dim)

        index_box.add(x_train)

        attack_geo = AttackModel(x_train, y_train, gep['k'], knn.indices[0], log,
                                 index_box=index_box)

        log.info('Start running Geo...')

        # Distance between x and z, the output x_adv, and whether success
        dist, adv_out, exit_code = [], [], []

        for i, idx in enumerate(index):
            log.info(f'# ==================== SAMPLE {i} =================== #')
            query_x     = x_test[idx].flatten().astype(gep['dtype'])
            truth_label = y_test[idx]

            if glp['init_ub']:
                gep['upperbound'] = upperbound[i]
            else:
                gep['upperbound'] = np.inf

            log.info(f'Upper bound: {gep["upperbound"]:.4f}')

            output_geo = attack_geo.attack(query_x, truth_label, gep, k=gep['k'])
            dist.append(output_geo[1])
            adv_out.append(output_geo[0])
            exit_code.append(output_geo[2])

        # Filter out failed adv for following evaluation
        dist_2 = [d for d in dist if d < np.inf]

        runtime = time.time() - start
        log.info(f'Total runtime: {runtime:.2f}s')
        log.info(f'mean: {np.mean(dist_2):.4f}, median: {np.median(dist_2):.4f}, all: {dist_2}')
        log.info(f'exit code: {exit_code}')

        pickle.dump([exit_code, dist, upperbound], open(f'save/{logger_name}.pth', 'wb'))
        log.info('Exit code: %d, %d, %d.' % (int(np.sum(0 == np.array(exit_code))),
                                             int(np.sum(1 == np.array(exit_code))),
                                             int(np.sum(2 == np.array(exit_code)))))
        if upperbound is not None:
            log.info(f'Initialized mean of upperbound: {np.mean(upperbound):.4f}')

        index_correct_3 = Main.classify_correct(x_train, y_train,
                                                np.stack(adv_out, axis=0), y_test[index],
                                                gep, num_classes)

        log.info(f'Failed sample number: {len(index_correct_3)}')
        log.info(f'Success sample number: {len(y_test[index]) - len(index_correct_3)}')
        log.info(f'Success rate: {(len(y_test[index]) - len(index_correct_3)) / len(y_test[index]):.4f}')

        # Closing log files
        handlers = log.handlers[:]
        for handler in handlers:
            handler.close()
            log.removeHandler(handler)

        return [dist, adv_out, exit_code, runtime, clean_acc]

    @staticmethod
    def attack(global_parameters_, geo_parameters_, scale_, trials_):
        outputs = []

        counter = 0

        for i in range(trials_):
            output = None
            # global_parameters_['seed'] = np.random.randint(2147483647)

            output = Main.attack_internal(global_parameters_, geo_parameters_, scale_)
            outputs = [global_parameters_['dataset'], output]

            counter += 1

        with open('./save/%s_geo_outputs_%d_geo_k%d.pkl' %
                  (global_parameters_['dataset'], global_parameters_['exp'], geo_parameters_['k']), 'wb') as f:
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

        'num_test': 200,  # The number of samples in test set you want to use

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

    geo_parameters = {
        'k': 3,

        # No use
        'method': 'gca',

        'dtype': np.float32,

        'device': 'cpu',

        'time_limit': 120,

        # We do not use 'all' option since its time comsuming
        'neighbor_method': 'm_nearest',

        'm': 20,  # Guided sample number

        'use_potential': False,

        'save_1nn_nb': False,

        'compute_dist_to_cell': True,

        'treat_facet_as_cell': False,

        'index': 'flat',

        # gca parameters
        'max_proj_iters': 2000,
        'max_proj_iters_verify': 10000,
        'early_stop': True,
        'check_obj_steps': 200,
        'upperbound': np.inf,
        'div_counter': 8,
        'div_ratio': 0.999,
        'div_step': 10,
        'tol': 1e-7,
    }

    v1 = Main()
    scale = 2
    trials = 1

    for k in [1, 3, 5, 7]:
        geo_parameters['k'] = k
        for dataset in ['australian',
                        'covtype',
                        'diabetes',
                        'fourclass',
                        'gaussian',
                        'yang-fmnist']:
            print("=================%s=================" % dataset)

            global_parameters['dataset'] = dataset
            v1.attack(global_parameters, geo_parameters, scale, trials)
