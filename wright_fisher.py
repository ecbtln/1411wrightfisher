__author__ = 'eblubin@mit.edu, nanaya@mit.edu'
import math
import numpy as np
from random import choice
from inspect import isfunction
from matplotlib import pyplot as plt
from UserDict import IterableUserDict
import random
import operator
import heapq
from progressbar import AnimatedProgressBar


# This is used to help debug the code in case of unexpected output. This will start the simulation at a particular
# state (a tuple of the signals_sent, and the receiver strategies), where each is a list of dictionaries of the
# appropriate length.
DEBUG_STATE = None

# The precision of the decimal comparison operations this should not need any changing
DECIMAL_PRECISION = 5

# Colors used to plot the senders and receivers
GRAPH_COLORS = 'mcrgbyk'


class SparseDictionary(IterableUserDict):
    """
    A helper dictionary that helps minimize the overhead of storing continuous actions. Instead of storing keys
    for every possible strategy, we make use of the fact that keys will be queried in order and that this dictionary
    will only be used to store cumulative frequencies.
    """
    def __init__(self, asc=True, default=0.0, *args, **kwargs):
        """
        Initialize the sparse SparseDictionary
        :param asc: whether the dictionary will be queried in ascending or descending order. Ascending corresponds
        to sender payoffs where we accumulate upwards, and descending corresponds to receiver payoffs where we are
        accumulating downwards
        :param default: The default value to return if the key does not have a value associated with it
        """
        IterableUserDict.__init__(self, *args, **kwargs)
        self.default = default
        if asc:
            self.cmp = operator.lt
        else:
            self.cmp = operator.gt
        self.history = []
        self.last_key, self.last_value = None, None

    def __getitem__(self, item):
        try:
            out = IterableUserDict.__getitem__(self, item)
            self.last_key = item
            self.last_value = out
            return out
        except KeyError as e:
            if self.last_key is None or self.cmp(item, self.last_key):
                return self.default
            else:
                return self.last_value


class WrightFisher(object):
    """
    A robust Wright-Fisher simulator of the costly signaling model, that allows for a variety of sender/receiver
    modifications and combinations and options for parameters.
    """

    def __init__(self, wages=(5,), sender_dist=(2.0/3.0, 1.0/3.0), w=0.15, u=0.02, receiver_prop=1.0/2.0, cost_fns = (lambda x: x * 3, lambda x: x), signals=(0, 1, 2, 3), receiver_dist = (1.0,), receiver_payoffs=((0, 10),), pop_size=100, fitness_func = lambda p, w: math.e**(p*w), animated_progress=True):
        """
        Construct a WrightFisher simulator with the desired parameters to be simulated one or more times.

        :param wages: a list of wages that receiver i needs to pay any sender whom it accepts.
        :param sender_dist: a probability distribution identifying how the senders will be divided by sender type.
        The sum of this must be 1, and this will also specify the number of types of senders there are
        :param w: the selection strength associated with the simulation
        :param u: the mutation rate, the probability that a given individual does not keep the same strategy but instead
        randomly chooses a new strategy
        :param receiver_prop: the proportion of the pop_size that wll be devoted to receivers, (1 - receiver_prop) will
        be devoted to senders.
        :param cost_fns: The cost functions for each type of sender, which can be passed in as callables or dictionaries
        mapping a signal to its cost
        :param signals: a list of all possible signals that can be sent
        :param receiver_dist: the distribute of proportion of receivers to each possible receiver type.
        :param receiver_payoffs: a list of payoffs that the receiver of type i receives for accepting a sender of type j
        :param pop_size: the population size used for the simulations, note this this should be sufficiently large
        relative to the number of possible signals
        :param fitness_func: a function that takes as arguments a payoff and selection strength and outputs fitness
        :param animated_progress: whether or not to display an animated progress bar while performing the simulation
        """

        # Verify the correctness and compatibility of the parameters
        assert math.fsum(sender_dist) == 1.0, "the sender distribution must be a valid probability distribution"
        assert math.fsum(receiver_dist) == 1.0, "the receiver distribution must be a valid probability distribution"
        assert len(sender_dist) == len(cost_fns), "the number of elements in the sender distribution must be equal to the number of elements in the cost functions"
        for x in receiver_payoffs:
            assert len(x) == len(sender_dist), "the number of elements in each of the receiver payoffs must be equal to the number of senders"
        assert len(receiver_dist) == len(receiver_payoffs) == len(wages), "the number of of elements in the receiver distribution, the receiver's payoffs, and the number of wages must all equal the number of total receiver types"
        assert len(sender_dist) > 1, "this model only makes sense with more than one type of sender"
        assert len(receiver_dist) > 0, "this model only makes sense with a nonzero number of senders"
        assert isinstance(pop_size, int), "the population size must be an integer, not something else"
        assert len(signals) == len(set(signals)), "the list of signals should not have any repeated elements"

        self.animated_progress = animated_progress

        self.wages = wages # benefit for being accepted by a given receiver
        self.sender_dist = sender_dist
        self.receiver_dist = receiver_dist
        self.n_types_of_senders = len(sender_dist)
        self.n_types_of_receivers = len(receiver_dist)
        self.w = w
        self.u = u
        self.num_signals = len(signals)
        self.signals = signals
        cost_functions_by_index = []


        # cost_fns can be inputted as either arrays (corresponding to the signals), or functions (mapping signal to cost)
        # we want to map them to arrays before we begin
        for f in cost_fns:

            if isinstance(f, (tuple, list)):
                assert len(f) == self.num_signals, "the list of payoffs for a given sender must be equal to the number of signals"
                cost_functions_by_index.append(f)
            else:
                assert isfunction(f)
                x = [f(s) for s in self.signals]
                cost_functions_by_index.append(x)

        self.cost_fns_by_signal_index = cost_functions_by_index # for each sender, a lookup table mapping the signal's index (in the signals array) to its cost
        # for convenience, we also want to make a direct mapping of all signals to their costs
        self.cost_fns = [{signals[i]:x[i] for i, s in enumerate(signals)} for x in cost_functions_by_index]
        self.signals = signals
        self.receiver_payoffs = receiver_payoffs
        self.n_types_of_receivers = len(receiver_dist)
        self.fitness_func = lambda p: fitness_func(p, w)


        assert pop_size is not None

        self.num_senders = [pop_size * x * (1 - receiver_prop) for x in sender_dist]
        total_receivers = pop_size * receiver_prop
        self.num_receivers = [total_receivers * x for x in receiver_dist]
        self.pop_size = pop_size


        self.num_senders = self._round_individuals(self.num_senders)
        self.num_receivers = self._round_individuals(self.num_receivers)
        self.index_of_signal = {s:i for i, s in enumerate(self._possible_receiver_strategies())}




    def _round_given_type(self, unrounded_dict, desired_total):
        """
        Converts a given sender or receiver's distribution, given as a dictionary, and scales it proportionally to add
        to the desired_total
        :param unrounded_dict: a weighted distribution of the number of senders and receivers sending each signal
        :param desired_total: the total to which the aggregate sum should be scaled
        """
        unrounded_total = sum(unrounded_dict[k] for k in unrounded_dict)
        total = int(round(unrounded_total, DECIMAL_PRECISION))
        assert total == desired_total

        int_nums = {k:int(unrounded_dict[k]) for k in unrounded_dict}

        diff = total - sum(int_nums[k] for k in int_nums)
        if diff > 0:
            thresh = [((int_nums[k] - unrounded_dict[k]), k) for k in int_nums]
            heapq.heapify(thresh)
            while diff > 0:
                v, i = heapq.heappop(thresh)
                int_nums[i] += 1
                diff -= 1

        assert sum(int_nums[k] for k in int_nums) == total

        return int_nums

    def _round_individuals(self, unrounded_frequencies):
        """
        Due to integer cutoffs, the number of senders and receivers might not be consistent. This take the integer part of each
        of the inputs and then assign the remaining few leftovers (so that the sum is the sum of the original floats)
        in a way such that the numbers with higher decimal parts will get the extra int before those with lower.
        """
        unrounded_total = math.fsum(unrounded_frequencies)
        total = int(round(unrounded_total, DECIMAL_PRECISION))

        int_num_senders = [int(x) for x in unrounded_frequencies]

        diff = total - sum(int_num_senders)
        if diff > 0:
            # note the difference needs to be negative, because heapq's only implement a minimum priority queue but we want max priority queue
            thresh = [((x - y), i) for i, (x, y) in enumerate(zip(int_num_senders, unrounded_frequencies))]
            heapq.heapify(thresh)
            while diff > 0:
                v, i = heapq.heappop(thresh)
                int_num_senders[i] += 1
                diff -= 1
        assert sum(int_num_senders) == total, "the total number of individuals after rounding must be the same as before rounding"

        return int_num_senders






    def _normalize_to_pop_size(self, senders, receivers):
        """ Takes in a list of distributions of senders and receivers and rounds each distribution of each type such that
        each type is scaled back to the appropriate total (since each type's population remains constant

        :param senders: the list of sender proportions
        :param receivers: the list of receiver proportions

        :return sender, receivers: a tuple of the scaled versions of the inputs
        """

        # to normalize, the sum at index i of senders should correspond to self.sender_dist at index i
        total_senders = [sum(d[k] for k in d) for d in senders]
        total_receivers = [sum(d[k] for k in d) for d in receivers]

        signals_sent = [{k:y[k] * N / total for k in y} for y, N, total in zip(senders, self.num_senders, total_senders)]
        receiver_strats = [{k:y[k] * N / total for k in y} for y, N, total in zip(receivers, self.num_receivers, total_receivers)]

        for i in xrange(self.n_types_of_senders):
            signals = signals_sent[i]
            signals_sent[i] = self._round_given_type(signals, self.num_senders[i])
        assert sum(sum(x[k] for k in x) for x in signals_sent) == sum(self.num_senders)

        for i in xrange(self.n_types_of_receivers):
            signals = receiver_strats[i]
            receiver_strats[i] = self._round_given_type(signals, self.num_receivers[i])
        assert sum(sum(x[k] for k in x) for x in receiver_strats) == sum(self.num_receivers)


        return signals_sent, receiver_strats


    def _compute_avg_cost(self, signals_by_sender_type):
        """
        :param signals_by_sender_type:  an array of senders, and each sender has a dictionary mapping a signal sent and the proportion of the population sending that signal.
        :Returns: the average signal sent by each sender type, as an array
        """

        out = []
        for f, signals in zip(self.cost_fns, signals_by_sender_type):
            sum_n = 0
            sum_v = 0
            for k in signals:
                sum_n += signals[k]
                sum_v += signals[k] * f[k]

            out.append(float(sum_v) / sum_n)

        return out

    def _compute_acceptance_frequencies(self, receiver_strategies):
        """
        :returns: an array of dictionaries mapping a key (the signal sent) to a value (the proportion of receivers accepting
        that signal) for every type of receiver
        """
        overall_out = []
        for z in receiver_strategies:
            out = {}

            def increment(k, v):
                out[k] = out.get(k, 0) + v

            for k in z:
                increment(k, z[k])

            signals = sorted(list(out.keys()))

            # make the frequency distribution into cumulative sums
            for i in xrange(len(signals) - 1):
                out[signals[i+1]] += out[signals[i]]

            frequency_accepted = SparseDictionary()
            for x in signals:
                frequency_accepted[x] = float(out[x])/out[signals[-1]]

            overall_out.append(frequency_accepted)

        return overall_out


    def _compute_type_frequencies(self, signals_sent_by_sender):
        """
        :returns: a dictionary mapping a key (the signal accepted), to an array, where each value at index i is the
        likelihood of having accepted a sender with that type
        """

        out = {}


        sums = {}
        def increment(x, s_index, val):
            sums[x] = sums.get(x, 0) + val
            likelihood = out.get(x, None)
            if likelihood is None:
                out[x] = np.zeros(self.n_types_of_senders)
                likelihood = out[x]
            likelihood[s_index] += val

        for s_index, sender in enumerate(signals_sent_by_sender):
            for x in sender:
                increment(x, s_index, sender[x])

        signals = sorted(list(out.keys()))
        # we go in opposite order as above because we are now change the receiver signal chosen, so lower means more, not
        # less, will be accepted
        for i in reversed(xrange(1, len(signals))):
            out[signals[i-1]] += out[signals[i]] # numpy element-wise addition

        total = sum(out[signals[0]])
        retvalue = SparseDictionary(asc=False, default=[0]*self.n_types_of_senders)
        for s in signals:
            retvalue[s] = out[s]
        return retvalue

    def _mean_of_frequency_table(self, freq_table):
        """ Compute the mean of a frequency table, which is a dictionary mapping values to their frequencies """
        s = 0
        tv = 0
        for k in freq_table:
            num = freq_table[k]
            s += num
            tv += k * num

        return float(tv)/s


    def _possible_receiver_strategies(self):
       return self.signals

    def simulate(self, num_gens=1000, show_signals_graph=True):
        """
        Performs a simulation on the given WrightFisher simulation object to a desired number of generations and
        defaulting to showing both the average cost of each sender type as well as the average signals of each sender
        and receiver type
        :param num_gens: the number of iterations to run the simulation for
        :param show_signals_graph: whether or not to show the supplemental graph
        """

        # if the DEBUG flag is turned on
        if DEBUG_STATE is not None:
            signals_sent, receiver_strats = DEBUG_STATE
        else:
            # initialize the state of the world to same random state, given the restrictions on the counts for the number of each player population


            # for each type of sender, randomly initialize a signal for each sender and store them as a frequency table
            signals_sent = []
            for x in self.num_senders:
                sender_freqs = {}
                for i in xrange(x):
                    c = choice(self.signals)
                    sender_freqs[c] = sender_freqs.get(c, 0) + 1
                signals_sent.append(sender_freqs)

            # for each receiver, randomly initialize a strategy based on the existing signals (plus a reject all)
            possible_receiver_strats = self._possible_receiver_strategies()
            receiver_strats = []
            for x in self.num_receivers:
                receiver_freqs = {}
                for i in xrange(x):
                    c = choice(possible_receiver_strats)
                    receiver_freqs[c] = receiver_freqs.get(c, 0) + 1
                receiver_strats.append(receiver_freqs)

        avg_cost_signals_sent = np.zeros((num_gens, self.n_types_of_senders))
        avg_signals_sent = np.zeros((num_gens, self.n_types_of_senders))
        avg_signals_sent[0, :] = [self._mean_of_frequency_table(x) for x in signals_sent]
        avg_cost_signals_sent[0, :] = self._compute_avg_cost(signals_sent)

        avg_signals_accepted = np.zeros((num_gens, self.n_types_of_receivers))
        avg_signals_accepted[0, :] = [self._mean_of_frequency_table(x) for x in receiver_strats]

        if self.animated_progress:
            # start the animated progress bar, if the bool is enabled
            progress_bar = AnimatedProgressBar(end=num_gens, width=80)
            print progress_bar,

        # Iterate through all the generations
        for t in xrange(num_gens - 1):
            # determine payoffs of each player

            # 1. for each type of receiver and for each strategy, determine proportion of receivers
            # accepting that strategy
            acceptance_ratios = self._compute_acceptance_frequencies(receiver_strats)

            # 2. for each type of sender, compute payoff for each possible signal
            sender_payoffs = [[sum(acceptance_ratios[r_i][s]*w - f[s] for r_i, w in enumerate(self.wages)) for s in self.signals] for f in self.cost_fns]

            # 3. compute payoffs for each possible receiver strategy for each possible receiver
            sender_likelihoods = self._compute_type_frequencies(signals_sent)
            receiver_payoffs = [[sum(sender_likelihoods[x][i]* (r_payoffs[i] - w) for i in reversed(xrange(self.n_types_of_senders))) / (self.pop_size / 5) for x in self._possible_receiver_strategies()] for w, r_payoffs in zip(self.wages, self.receiver_payoffs)]


            # compute fitnesses
            # this is a lookup table, where for each type of sender, we have the function for each possible strategy
            f_senders = [[self.fitness_func(p) for p in x] for x in sender_payoffs]
            f_receivers = [[self.fitness_func(p) for p in x] for x in receiver_payoffs]


            # generate frequencies for next generation, with some mutation rate ,u
            # we use a slightly different strategy than that included in the problem set. Instead of using a random
            # number generate to index into the cumulative distribution of fitnesses of individuals, we instead allocate
            # the exact fitness (as a decimal) for number of people in the ensuing population, and then normalize over
            # the sum of these fitnesses. This strategy seems to be more effective, as it reduces the random noise that
            # was present in the simulations for the problem set.
            new_signals_sent = []
            for i, signals_sent_by_sender in enumerate(signals_sent):
                new_freqs = {}
                fitnesses = f_senders[i]

                for signal in signals_sent_by_sender:
                    num = signals_sent_by_sender[signal]
                    for j in xrange(num):

                        if random.random() < self.u:
                            cur_signal = choice(self.signals)
                        else:
                            cur_signal = signal

                        idx = self.index_of_signal[cur_signal]
                        assert cur_signal == self.signals[idx] # make sure the lookup table is correct

                        f = fitnesses[idx]
                        old = new_freqs.get(cur_signal, 0)
                        new_freqs[cur_signal] = old + f
                        assert new_freqs[cur_signal] > old # make sure no overflow


                new_signals_sent.append(new_freqs)
            signals_sent = new_signals_sent

            #needs to repeat for all types of senders
            new_signals_received = []
            for i, signals_sent_by_receiver in enumerate(receiver_strats):
                new_freqs = {}
                fitnesses = f_receivers[i]

                for signal in signals_sent_by_receiver:
                    num = signals_sent_by_receiver[signal]
                    for j in xrange(num):
                        if random.random() < self.u:
                            cur_signal = choice(self.signals)
                        else:
                            cur_signal = signal

                        idx = self.index_of_signal[cur_signal]
                        assert cur_signal == self.signals[idx]
                        f = fitnesses[idx]

                        old = new_freqs.get(cur_signal, 0)
                        new_freqs[cur_signal] = old + f
                        assert new_freqs[cur_signal] > old # make sure no overflow

                new_signals_received.append(new_freqs)

            receiver_strats = new_signals_received

            # now we need to normalize new_signals and receiver_strats back down to their original population sizes
            signals_sent, receiver_strats = self._normalize_to_pop_size(signals_sent, receiver_strats)

            # We now want to update our running totals
            avg_signals_sent[t + 1, :] = [self._mean_of_frequency_table(x) for x in signals_sent]
            avg_cost_signals_sent[t + 1, :] = self._compute_avg_cost(signals_sent)
            avg_signals_accepted[t + 1, :] = [self._mean_of_frequency_table(x) for x in receiver_strats]

            if self.animated_progress:
                # print the progress bar, if it is enabled
                print '\r',
                print progress_bar + 1,

        # plot the results
        self._plot_results(avg_signals_sent, avg_cost_signals_sent, avg_signals_accepted, num_gens, show_signals_graph=show_signals_graph)

    def _plot_results(self, avg_signals_sent, avg_costs, avg_accepted, t, show_signals_graph=False):
        colors = GRAPH_COLORS
        x_axis = range(t)
        if show_signals_graph:
            plt.figure(1)
            plt.subplot(211)
        for sender_type_idx in xrange(self.n_types_of_senders):
            plt.plot(x_axis, avg_costs[: t, sender_type_idx], colors[sender_type_idx], label='S_%d' % sender_type_idx)
        if not show_signals_graph:
            plt.legend(borderaxespad=0, bbox_to_anchor=(1.01, 1), loc=2)
            plt.xlabel('Generation')
        plt.title('Costly Signaling in Wright Fisher')
        plt.ylabel('Average cost of signal')
        plt.ylim(self.signals[0], np.max(avg_costs))


        # show supplemental graph to help interpret results, this one will just show the signal sent and received by
        # all parties over time
        if show_signals_graph:
            plt.subplot(212)
            for sender_type_idx in xrange(self.n_types_of_senders):
                plt.plot(x_axis, avg_signals_sent[: t, sender_type_idx], colors[sender_type_idx], label='S_%d' % sender_type_idx)
            for receiver_type_idx in xrange(self.n_types_of_receivers):
                plt.plot(x_axis, avg_accepted[: t, receiver_type_idx], colors[self.n_types_of_senders + receiver_type_idx], label='R_%d' % receiver_type_idx)


            plt.legend(loc=3, borderaxespad=0, ncol=self.n_types_of_senders + self.n_types_of_receivers, mode="expand", bbox_to_anchor=(0., -.22, 1., .102))

            plt.ylabel('Average signal')
            plt.ylim(self.signals[0], self.signals[-1])
        plt.show()



if __name__ == '__main__':
    w = WrightFisher(pop_size=100, signals=(0, 1, 2, 4))
    w.simulate(num_gens=10000)
