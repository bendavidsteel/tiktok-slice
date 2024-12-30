from __future__ import division
import numpy as np
import collections

__author__ = 'george paraskevopoulos'


# the implementation and the symbols are based on
# [Gale & Sampson, 1995]
class SimpleGoodTuring():
    def __init__(self, species, max_count):
        self._sort_dict = lambda my_dict: \
            collections.OrderedDict(sorted(my_dict.items()))
        self.species, self.max_count = species, max_count
        self.r, self.n, self.Z, self.S, self.P0, self.sgt_probs = \
            None, None, None, None, None, None

    def count_of_counts(self, species, max_count):
        """
        compute the count of counts for the
        different individuals in the species
        :param species: dict, initial measurement
        :param max_count: int, most popular element count
        :return: (r, n) = (np.array, np.array):
                #individuals in species, count of species with r individuals
        """
        N_c = dict()
        for r in range(1, max_count + 1):
            n = np.sum(np.fromiter((1 for v in species.values() if v == r), np.int32))
            if n != 0:
                N_c[r] = n
        sort_Nc = self._sort_dict(N_c)
        r = np.fromiter(sort_Nc.keys(), dtype=np.int32)
        n = np.fromiter(sort_Nc.values(), dtype=np.int32)
        return r, n

    def Z_smoothing(self, r, n):
        """
        Z smooths the count of counts
        by "interpolating" each element with
        the previous and the next value
        :param r: numpy.array, number of individuals of certain species
        :param n: numpy.array, number of different species with r individuals
        :return: np.array, smoothed count of counts
        """
        Z = dict()
        Z[r[0]] = 2 * n[0] / r[1]
        Z[r[-1]] = n[-1] / (r[-1] - r[-2])
        for idx in range(1, len(r) - 1):
            j, nj = r[idx], n[idx]
            i, k = r[idx - 1], r[idx + 1]
            Z[j] = 2 * nj / (k - i)
        sort_Z = self._sort_dict(Z)
        Z_arr = np.fromiter(sort_Z.values(), dtype=np.float64)
        return Z_arr

    def linregres_coeffs(self, r, Z):
        """
        Computes best fit line a+b*log(r) to the pairs log(r) - log(Z)
        :param r: numpy.array
        :param Z: numpy.array, smoothed count of counts
        :return: (a, b, S) = (int, int, numpy.array):
                cross of y axis, slope of line, antilog(a+b*log(r))
        """
        logr = np.log10(r)
        logZ = np.log10(Z)
        A = np.vstack([logr, np.ones(len(logr))]).T
        b, a = np.linalg.lstsq(A, logZ)[0]
        return a, b

    def compute_S(self, a, b, r):
        return 10 ** (a + b * np.log(r))

    def smooth_counts(self, r, n, S):
        """
        Compute smoothed (discounted) count values r*
        :param r: numpy.array
        :param n: numpy.array
        :param S: numpy.array
        :return: numpy.array, r*
        """
        r_star = np.array(r, dtype=float)
        cease_x = False
        for i in np.arange(len(r)):
            y = (r[i] + 1) * S[r[i]] / S[r[i] - 1]  # zero index base in python
            print(f"Sr+1 = {str(S[r[i]])}")
            print(f"Sr = {str(S[r[i] - 1])}")
            x = 0
            t = float("inf")
            if i + 1 >= len(r) or r[i + 1] != r[i] + 1:
                cease_x = True
            if not cease_x:
                x = (r[i] + 1) * n[i + 1] / n[i]
                t = 1.96 * np.sqrt((r[i] + 1) * (r[i] + 1) * n[i + 1] *
                                   (1 + n[i + 1] / n[i]) / (n[i] * n[i]))
            if np.abs(x - y) <= t:
                cease_x = True
                r_star[i] = y
            else:
                r_star[i] = x
        return r_star

    def sgt_discount(self, r, r_star, n):
        """
        Compute P0 and Good-Turing probabilities discount
        :param r: numpy.array
        :param r_star: numpy.array, smoothed r
        :param n: numpy.array
        :param S: numpy.array
        :return: (P0, sgt_probs) = (float, numpy.array):
                the probability of an unseen species, sgt probabilities discount
        """
        P0 = r[0] / np.inner(r, n)
        sgt_probs = (1 - P0) * r_star / np.inner(r_star, n)
        return P0, sgt_probs

    def species_probs(self, species, r, P0, sgt_probs, species_pool=[]):
        """
        :param species: dict
        :param P0: float
        :param sgt_probs: numpy.array
        :param species_pool: list, all the possible species if known a priori
        :return: dict, the probability of any seen species
                if species_pool is specified the probability
                of the unseen species is also calculated
        """
        species_sgt = dict()
        if len(species_pool) == 0:
            species_pool = species.keys()
        species_seen = species.keys()
        species_unseen = list(set(species_seen) ^ set(species_pool))
        for s in species_seen:
            idx = -1
            try:
                idx = np.where(r == species[s])
            except KeyError:
                print("Value not found in r table.\n")
                print("Maybe used different species for computing sgt probs?\n")
            species_sgt[s] = sgt_probs[idx][0]

        total_unseen = len(species_unseen)
        for s in species_unseen:
            species_sgt[s] = P0 / total_unseen
        return species_sgt

    def run_sgt(self, species_pool=[]):
        self.r, self.n = self.count_of_counts(self.species, self.max_count)
        self.Z = self.Z_smoothing(self.r, self.n)
        a, b = self.linregres_coeffs(self.r, self.Z)
        self.S = self.compute_S(a, b, np.arange(1, self.max_count + 2))
        r_star = self.smooth_counts(self.r, self.n, self.S)
        self.P0, self.sgt_probs = self.sgt_discount(self.r, r_star, self.n)
        species_sgt = self.species_probs(self.species,
                                         self.r, self.P0,
                                         self.sgt_probs,
                                         species_pool)
        return species_sgt
