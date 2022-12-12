'''
 * @author Rroscha
 * @date 2022/12/12 12:01
'''

import time
from copy import deepcopy

import numpy as np
from joblib import Parallel, delayed

from lib.myskiplist import MySkipList
from lib.polytope_dual import gca


class AttackModel:
    EPS = 1e-9
    TOL = 1e-6

    FAIL    = 0  # Fail to find an optimal adversarial examples
    SUCCESS = 1  # Optimal adversarial examples found
    TIMELIM = 2  # Maximum time limit is reached
    MISCLF  = 3  # The query is already classified

    def __init__(self, x_train, y_train, k, index, log, index_box=None):
        self.points = x_train
        self.labels = y_train
        self.num_classes = len(np.unique(y_train))
        self.k = k
        self.log = log
        self.num_points, self.dim = x_train.shape

        # dictionary to store neighboring relationship of 1st-order cells
        self.nb_dict = {}

        # index for kNN search
        self.knn = index
        self.knn_box = index_box if index_box is not None else index

        # DEPRECATED: Compute class mean (only used with potential)
        self.class_means = np.zeros((self.num_classes, self.dim),
                                    dtype=x_train.dtype)
        for i in range(self.num_classes):
            self.class_means[i] = x_train[x_train == i].mean(0)

    def _get_hyperplane(self, point1, point2, w):
        point1 = self.points[point1]
        point2 = self.points[point2]
        midpoint = (point1 + point2) / 2

        # normal vector is normalized to have norm of 1
        diff = point2 - point1
        if point1.ndim == point2.ndim:
            w[:-1] = diff / np.maximum(np.linalg.norm(diff, 2), self.EPS)
            w[-1]  = - w[:-1] @ midpoint
        else:
            w[:, :-1] = diff / \
                np.maximum(np.linalg.norm(diff, 2, 1), self.EPS)[:, np.newaxis]
            w[:, -1]  = - (midpoint * w[:, :-1]).sum(1)
    @classmethod
    def _compute_lb_dist(cls, query_x, hyperplanes, return_proj=False):
        signed_dist = hyperplanes[:, :-1] @ query_x + hyperplanes[:, -1]
        dist = signed_dist

        if return_proj:
            # Move proj slightly inside the polytope
            # Get the projection point of hyperplane, xp = x - r * w/||w||.
            # Since hyperplanes is already normalized by  w/||w||, thus, xp = x - r * hyperplanes[:, :-1].
            # Lecture 5 page 20
            proj = query_x - (signed_dist + 1e-5).reshape(-1, 1) * hyperplanes[:, :-1]

            return dist, proj

        return dist

    def classify(self, cell):
        """Returns majority label of <cell>. Returns all labels that tie."""
        counts = np.bincount(self.labels[list(cell)])
        max_counts = counts.max()
        return np.where(counts == max_counts)[0]


    def compute_potential(self, query, label, proj, params):
        # DEPRECATED: potential doesn't work here
        potential = np.linalg.norm(query - proj)
        if params['use_potential']:
            raise NotImplementedError()
            class_dists = np.sum((self.class_means - proj) ** 2, 1)
            true_class_dist = class_dists[label]
            class_dists[label] += 1e9
            other_class = class_dists.argmin()
            alpha = 0.5 / (np.linalg.norm(self.class_means[label]
                                          - self.class_means[other_class]))
            potential += alpha * (true_class_dist
                                  - class_dists[other_class])
        return potential


    def dist_to_facet(self, cur_cell, idx, facet, query, label, A, AAT, b,
                      params, is_adv, ub=None, dist_to_cell=None):

        if dist_to_cell is None:
            dist_to_cell = params['compute_dist_to_cell']
        if ub is None:
            ub = params['upperbound']

        if not dist_to_cell:
            # Apply screening
            b_hat = A @ query - b
            if np.any(ub < np.maximum(b_hat, 0)):
                self.log.debug('Screened.')
                return None
            # Compute distance to facet
            proj = gca(query, A, AAT, b, params, idx_plane=idx, ub=ub)
        else:
            # Get facets of neighbor cell that we want to compute distance to
            neighbor = self._get_neighbor_from_facet(cur_cell, facet)
            _, nb_hplanes = self.get_neighbor_facet(neighbor, params)
            # TODO: handle box constraint
            nb_A = nb_hplanes[:, :-1]
            nb_b = - nb_hplanes[:, -1]
            # Apply screening
            b_hat = nb_A @ query - nb_b
            if np.any(ub < np.maximum(b_hat, 0)):
                self.log.debug('Screened.')
                return None
            nb_AAT = nb_A @ nb_A.T
            # Compute distance to cell
            proj = gca(query, nb_A, nb_AAT, nb_b, params, idx_plane=None, ub=ub)
        # Skip if solver exits with None, either facet is further
        # than ub or this is an invalid facet
        if proj is None:
            return None

        true_dist = self.compute_potential(query, label, proj, params)
        # Set lb bit to 0 since we compute true distance
        return (true_dist, (cur_cell, facet, is_adv, 0))

    @staticmethod
    def _check_labels_adv(label_list, label):
        return np.any(label_list != label)

    def _get_precise_label(self, inpt, k):
        """
        Use this method to classify when <inpt> is close to or on multiple
        bisectors. Normal knn can be ambiguous in this case.
        """
        dist = np.sum((inpt - self.points) ** 2, 1)
        k_dist = np.sort(dist)[k - 1]
        indices = np.where(dist - k_dist < self.TOL)[0]

        close_indices = np.where(np.abs(dist - k_dist) < self.TOL)[0]
        sure_indices = np.setdiff1d(indices, close_indices)
        close_labels = self.labels[close_indices]
        sure_labels = self.labels[sure_indices]
        close_counts = np.bincount(close_labels, minlength=self.num_classes)
        sure_counts = np.bincount(sure_labels, minlength=self.num_classes)

        num_to_fill = k - sure_counts.sum()
        # If number of sure counts is k, then we are done
        assert num_to_fill >= 0
        if num_to_fill == 0:
            max_count = sure_counts.max()
            return np.where(sure_counts == max_count)[0]

        y_pred = []
        for i in range(self.num_classes):
            num_fill = min(num_to_fill, close_counts[i])
            new_counts = deepcopy(sure_counts)
            new_counts[i] += num_fill
            close_counts_tmp = deepcopy(close_counts)
            # Fill the other classes in a way that benefits class i most
            while num_fill < num_to_fill:
                assert np.all(close_counts_tmp >= 0)
                # Get classes that can still be filled except for i
                ind = np.setdiff1d(np.where(close_counts_tmp > 0)[0], i)
                # Find class with the smallest count
                ind_to_fill = ind[new_counts[ind].argmin()]
                new_counts[ind_to_fill] += 1
                close_counts_tmp[ind_to_fill] -= 1
                num_fill += 1
            assert new_counts.sum() == k
            max_count = new_counts.max()
            if new_counts[i] == max_count:
                y_pred.append(i)

        return np.array(y_pred)

    @staticmethod
    def _get_neighbor_from_facet(current_cell, facet):
        # Find one of the side of facet equal to current cell and extract its contrast facet.
        neighbor = [facet[1] if x == facet[0] else x for x in current_cell]
        return tuple(sorted(neighbor))

    def exit_func(self, exit_code, query, ub_facet, start_main, visited_cells,
                  computed_cells, dist=None, proj=None):
        if exit_code == self.MISCLF:
            self.log.info('CODE 3: Query is already misclassified.')
            return query, 0, exit_code
        self.log.info('FINISHED: main loop time: %.2f' %
                      (time.time() - start_main))
        self.log.info('num cells visited: %d, num cells computed: %d' %
                      (len(visited_cells), len(computed_cells)))
        if exit_code == self.FAIL:
            self.log.info('CODE 0: No valid adv cell found.')
            self.log.info('returning the initial upperbound.')
            return query, ub_facet[0], exit_code
        if exit_code == self.SUCCESS:
            self.log.info(
                'CODE 1: Success. Optimal adv cell found! Dist: %.4f' % dist)
            return proj, dist, exit_code
        if exit_code == self.TIMELIM:
            self.log.info('CODE 2: Time limit. At least one adv cell found! ' +
                          'Dist: %.4f' % dist)
            return proj, dist, exit_code
        raise NotImplementedError('Unknown exit code!')

    def verify_adv_cell(self, query, label, cell, facet, k, params):
        adv_cell = self._get_neighbor_from_facet(cell, facet)
        ver_params = deepcopy(params)
        ver_params['neighbor_method'] = 'all'
        ver_params['save_1nn_nb'] = False
        # DEBUG: ver_params?
        _, hplanes = self.get_neighbor_facet(adv_cell, params, parallel=None)
        A = hplanes[:, :-1]
        b = - hplanes[:, -1]
        AAT = A @ A.T
        ver_params['max_proj_iters'] = params['max_proj_iters_verify']
        proj = gca(query, A, AAT, b, ver_params, idx_plane=None, ub=np.inf)
        if proj is None:
            return False, None

        final_label = self._get_precise_label(proj, k)
        is_adv = self._check_labels_adv(final_label, label)
        return is_adv, proj

    def get_neighbor_facet(self, cell, params, parallel=None):
        if params['neighbor_method'] == 'all':
            masks = [list(range(self.num_points)), ] * len(cell)
        elif params['neighbor_method'] == 'm_nearest':
            # Contain itself so m + 1
            _, masks = self.knn_box.search(self.points[list(cell)].reshape(len(cell), -1), params['m'] + 1)
            # Exclude the point itself from neighbor list
            masks = [mask[1:] for mask in masks]
        else:
            raise NotImplementedError('no specified approximate neighbors.')

        union_neighbors = set()

        for mask in masks:
            union_neighbors.update(mask)

        # Subtract elements in cell from union_nb
        union_neighbors = list(union_neighbors - set(cell))
        len_nb = len(union_neighbors)

        # Create a list of all possible hyperplanes
        hyperplanes = np.zeros((len_nb * len(cell), self.dim + 1), dtype=params['dtype'])

        facets = []
        for i, point in enumerate(cell):
            # Get hyperplanes' w and b, wx +b = 0 is the hyperplane
            self._get_hyperplane(point, union_neighbors, hyperplanes[i * len_nb:(i + 1) * len_nb])

            # Theorem 1
            for u in union_neighbors:
                facets.append((point, u))

        return facets, hyperplanes

    def attack(self, query_x, truth_label, gep, k=None):
        if k is None:
            k = self.k

        # Q stores (distance, (cell, facet, is_adv, is_lb)) of facet
        Q = MySkipList()

        # Set upperbound
        ub_facet = (gep['upperbound'], ([], (-1, -1), 0, 0))  # Except gep['upperbound'], the redundant is useless

        # Get current facet and put it into Q
        _, indices = self.knn.search(query_x.reshape(1, -1), k)
        current_cell = tuple(sorted(indices[0]))
        visited_cells = {current_cell}
        computed_cells = {current_cell}

        start = time.time()

        # https://arxiv.org/pdf/2011.09719.pdf  algorithm 1
        while True:
            # Whether out of time
            if time.time() - start > gep['time_limit']:
                self.log.info('Hit the time limit. Verifying seen adv cell...')
                while True:
                    if len(Q) > 0:
                        (dist, (cell, facet, is_adv, _)) = Q.popitem()
                    else:
                        # if Q is empty before finding any adversarial cell,
                        # return the failure code
                        return self.exit_func(self.FAIL, query_x, ub_facet,
                                              start, visited_cells,
                                              computed_cells)
                    if not is_adv:
                        continue

                    is_verified, proj = self.verify_adv_cell(query_x, truth_label, cell, facet, k, gep)
                    if is_verified:
                        dist = np.linalg.norm(query_x - proj)
                        return self.exit_func(
                            self.TIMELIM, query_x, ub_facet, start,
                            visited_cells, computed_cells, dist=dist, proj=proj)

            # Determine neighboring facets of current cell
            self.log.debug('Getting facets...')
            facets, hyperplanes = self.get_neighbor_facet(
                current_cell, gep,
                parallel=None)

            # extract A, AA^T, and b from hplanes
            if gep['compute_dist_to_cell']:
                A   = None
                b   = None
                AAT = None

            # Compute lower bound
            self.log.debug('Computing lb dist...')
            lb_dist, proj = self._compute_lb_dist(query_x, hyperplanes, return_proj=True)
            self.log.debug('done.')

            self.log.debug('Screening with lower bound distance to facets...')
            lb_dist = - lb_dist
            indices = lb_dist.argsort()

            # Ignore facets greater than ub_facet
            mask = lb_dist[indices] <= ub_facet[0] + self.TOL
            indices = indices[mask]

            self.log.debug('Determining adv/benign facets...')
            adv_indices, benign_indices = [], []
            adv_nb, benign_nb = [], []
            for idx in indices:
                # Find the neighboring cell
                neighbor = self._get_neighbor_from_facet(current_cell, facets[idx])

                # Skip facet if neighbor is already visited
                if neighbor in visited_cells:
                    continue

                if gep['compute_dist_to_cell'] and neighbor in computed_cells:
                    continue

                # Check label of neighbor if adversarial
                neighbor_label = self.classify(neighbor)

                if self._check_labels_adv(neighbor_label, truth_label):
                    adv_indices.append(idx)
                    adv_nb.append(neighbor)
                else:
                    benign_indices.append(idx)
                    benign_nb.append(neighbor)

            self.log.debug(
                f'After lb screening | num adv facets: {len(adv_indices)}, '
                f'num benign facets: {len(benign_indices)}')

            # Set upperbound
            # For adv (first)
            if len(adv_indices) > 0:
                self.log.debug('Computing distance to adv facets to find upper bound...')
                _, proj_neighbours = self.knn.search(proj[adv_indices], k)
                self.log.debug('k-NN search done.')

                for i, (idx, neighbor) in enumerate(zip(adv_indices, adv_nb)):
                    if not gep['compute_dist_to_cell'] and set(proj_neighbours[i]) == set(neighbor):
                        print("I have no idea!")
                    else:
                        if gep['neighbor_method'] == 'all':
                            facet = self.dist_to_facet(
                                current_cell, idx, facets[idx], query_x, truth_label, A, AAT,
                                b, gep, 1)
                        elif gep['neighbor_method'] == 'm_nearest':
                            is_verified, x_adv = self.verify_adv_cell(query_x, truth_label, current_cell,
                                                                      facets[idx], k, gep)

                            if is_verified:
                                dist = np.linalg.norm(query_x - x_adv)
                                facet = (dist, (current_cell, facets[idx], 1, 0))
                            else:
                                facet = None

                    self.log.debug((neighbor, facet))
                    computed_cells.add(tuple(neighbor))

                    # Skip if facet is invalid or further than the upper bound
                    if facet is None or facet[0] > ub_facet[0] + self.TOL:
                        continue

                    Q.insert(facet[0], facet[1])

                    # Set new upper bound of adversarial distance
                    ub_facet = facet
                    gep['upperbound'] = facet[0]
                    self.log.debug(f'>>>>>>> new ub_facet: {facet[0]:.4f} <<<<<<<')

            # For benign
            # Since we may have new ub, thus refine
            self.log.debug('Second screening on benign facets...')
            mask = np.where(lb_dist[benign_indices] <= ub_facet[0] + self.TOL)[0]
            benign_indices = np.array(benign_indices)[mask]
            benign_nb = np.array(benign_nb)[mask]

            if len(benign_indices) > 0:
                self.log.debug(f'Computing distance to benign facets...')
                _, proj_neighbours = self.knn.search(proj[benign_indices], k)
                self.log.debug('k-NN search done.')

                for i, (idx, neighbor) in enumerate(zip(benign_indices, benign_nb)):
                    if not gep['compute_dist_to_cell'] and set(proj_neighbours[i]) == set(neighbor):
                        print("I have no idea!")
                    else:
                        facet = self.dist_to_facet(
                            current_cell, idx, facets[idx], query_x, truth_label, A,
                            AAT, b, gep, 0)

                    self.log.debug((neighbor, facet))
                    computed_cells.add(tuple(neighbor))

                    if facet is None or facet[0] > ub_facet[0] + self.TOL:
                        continue

                    Q.insert(facet[0], facet[1])

            # deleteMin
            while True:
                if len(Q) == 0:
                    self.log.info('PQ is empty. No facet to pop.')
                    return self.exit_func(
                        self.FAIL, query_x, ub_facet, start, visited_cells,
                        computed_cells)

                dist, (cell, facet, is_adv, _) = Q.popitem()
                # The current cell from facet
                neighbor = self._get_neighbor_from_facet(cell, facet)

                if neighbor not in visited_cells:
                    visited_cells.add(neighbor)

                    # Only return facet that smaller than ub
                    if dist <= ub_facet[0] + self.TOL:
                        break

            if is_adv:
                is_verified, proj = self.verify_adv_cell(
                    query_x, truth_label, cell, facet, k, gep)

                return self.exit_func(
                    self.SUCCESS, query_x, ub_facet, start,
                    visited_cells, computed_cells, dist=dist, proj=proj)

            current_cell = neighbor


















