import os
import numpy as np
import time
import random
import copy
import math
import datetime
import bisect
import itertools
from multiprocessing import Manager, Process, cpu_count

# Manages a shared pool of elite solutions for parallel processing.
class SolutionPool():
    def __init__(self, size, pool, best_pair, lock=None, print=False, Best=None, env=None):
        """
        Initializes the SolutionPool.
        Args:
            size (int): The maximum size of the pool.
            pool (Manager.list): A multiprocessing list to store solutions.
            best_pair (Manager.list): A multiprocessing list to track the best solution found [fitness, keys, time].
            lock (Manager.Lock): A lock for safe concurrent access.
            print (bool): Flag to enable/disable printing new best solutions.
            Best (float): The known best possible value for the instance, for GAP calculation.
            env (object): The problem environment instance.
        """
        self.size = size
        self.pool = pool
        self.best_pair = best_pair
        self.lock = lock
        self.start_time = time.time()
        self.print = print
        self.best_possible = Best
        self.env = env

    def insert(self, entry_tuple, metaheuristic_name, tag):
        """
        Inserts a new solution into the pool in a thread-safe manner.
        It keeps the pool sorted by fitness and manages the overall best solution found.
        """
        fitness, keys = entry_tuple
        with self.lock:
            # Update the globally best solution if the new one is better.
            if fitness < self.best_pair[0]:
                self.best_pair[0] = fitness
                self.best_pair[1] = list(keys)
                self.best_pair[2] = round(time.time() - self.start_time, 2)

                # Print information about the new best solution found.
                if self.print:
                    gap_info = ""
                    if self.best_possible is not None:
                        gap = ((fitness - self.best_possible) / self.best_possible) * 100
                        gap_info = f" - BEST: {self.best_possible} - GAP: {gap:.2f}%"
                    
                    print(f"\n{metaheuristic_name} NEW BEST: {fitness}{gap_info} - Time: {self.best_pair[2]}s - Pool Size: {len(self.pool)}")

            # Insert the new solution while maintaining the sorted order and size limit.
            bisect.insort(self.pool, entry_tuple)
            if len(self.pool) > self.size:
                self.pool.pop()

# Main class for the Random-Key Optimizer framework.
class RKO():
    def __init__(self, env, print_best=False, save_directory=None):
        """
        Initializes the RKO solver.
        Args:
            env (object): The problem-specific environment object. This is the crucial part for adaptation.
            print_best (bool): Flag to enable printing updates for new best solutions.
            save_directory (str): Path to a file for saving run results.
        """
        self.env = env
        self.__MAX_KEYS = self.env.tam_solution
        self.LS_type = self.env.LS_type
        self.start_time = time.time()
        self.max_time = self.env.max_time
        self.rate = 1
        self.print_best = print_best
        self.save_directory = save_directory
        self.q_managers = {}

    def _setup_parameters(self, metaheuristic_name, params_config):
        """
        Sets up parameters for a metaheuristic, deciding between static (offline)
        and dynamic (online with Q-Learning) configuration.
        """
        is_online = any(len(v) > 1 for v in params_config.values())

        if is_online:
            if metaheuristic_name not in self.q_managers:
                self.q_managers[metaheuristic_name] = QLearningManager(
                    parameters_config=params_config,
                    max_time=self.max_time,
                    metaheuristic_name=metaheuristic_name,
                    save_report=self.env.save_q_learning_report
                )
            q_manager = self.q_managers[metaheuristic_name]
            initial_params = q_manager.get_current_parameters()
            return q_manager, initial_params
        else:
            static_params = {k: v[0] for k, v in params_config.items()}
            return None, static_params

    # Generates a vector of random keys of size __MAX_KEYS.
    def random_keys(self):
        return np.random.random(self.__MAX_KEYS)

    # Applies perturbations to a key vector to escape local optima.
    def shaking(self, keys, beta_min, beta_max):
        beta = random.uniform(beta_min, beta_max)
        new_keys = copy.deepcopy(keys)
        
        num_perturbations = max(1, int(self.__MAX_KEYS * beta))
        for _ in range(num_perturbations):
            move_type = random.choice(['Swap', 'SwapN', 'Invert', 'Random'])
            
            if move_type == 'Swap':
                idx1, idx2 = random.sample(range(self.__MAX_KEYS), 2)
                new_keys[idx1], new_keys[idx2] = new_keys[idx2], new_keys[idx1]
            
            elif move_type == 'SwapN':
                idx = random.randint(0, self.__MAX_KEYS - 1)
                if idx == 0:
                    idx2 = 1
                elif idx == self.__MAX_KEYS - 1:
                    idx2 = idx - 1
                else:
                    idx2 = random.choice([idx - 1, idx + 1])
                new_keys[idx], new_keys[idx2] = new_keys[idx2], new_keys[idx]
                               
            elif move_type == 'Invert':
                idx = random.randint(0, self.__MAX_KEYS - 1)
                new_keys[idx] = 1.0 - new_keys[idx]
                            
            elif move_type == 'Random':
                idx = random.randint(0, self.__MAX_KEYS - 1)
                new_keys[idx] = random.random()
        
        return new_keys
    
    # A local search heuristic that improves a solution by swapping pairs of keys.
    def SwapLS(self, keys, metaheuristic_name="SwapLS"):
        if self.LS_type == 'Best':
            swap_order = list(range(int(self.rate * self.__MAX_KEYS)))
            random.shuffle(swap_order)
            
            best_keys = copy.deepcopy(keys)
            best_cost = self.env.cost(self.env.decoder(best_keys))
            
            for idx1 in swap_order:
                for idx2 in reversed(swap_order):
                    if self.stop_condition(best_cost, metaheuristic_name, -1):
                        return best_keys

                    new_keys = copy.deepcopy(best_keys)
                    new_keys[idx1], new_keys[idx2] = new_keys[idx2], new_keys[idx1]
                    new_cost = self.env.cost(self.env.decoder(new_keys))
                    
                    if new_cost < best_cost:
                        best_keys = new_keys
                        best_cost = new_cost
            return best_keys
        
        elif self.LS_type == 'First':
            swap_order = list(range(int(self.rate * self.__MAX_KEYS)))
            random.shuffle(swap_order)
            
            best_keys = copy.deepcopy(keys)
            best_cost = self.env.cost(self.env.decoder(best_keys))
            
            for idx1 in swap_order:
                for idx2 in reversed(swap_order):
                    if self.stop_condition(best_cost, metaheuristic_name, -1):
                        return best_keys
                        
                    new_keys = copy.deepcopy(best_keys)
                    new_keys[idx1], new_keys[idx2] = new_keys[idx2], new_keys[idx1]
                    new_cost = self.env.cost(self.env.decoder(new_keys))
                    
                    if new_cost < best_cost:
                        return new_keys
            return best_keys
    
    # A local search that adjusts key values based on the Farey sequence.
    def FareyLS(self, keys, metaheuristic_name="FareyLS"):
        Farey_Sequence = [0.00, 0.142857, 0.166667, 0.20, 0.25, 0.285714, 0.333333, 0.40, 0.428571, 0.50, 
                                 0.571429, 0.60, 0.666667, 0.714286, 0.75, 0.80, 0.833333, 0.857143, 1.0]
        
        if self.LS_type == 'Best':
            swap_order = list(range(int(self.rate * self.__MAX_KEYS)))
            random.shuffle(swap_order)
            
            best_keys = copy.deepcopy(keys)
            best_cost = self.env.cost(self.env.decoder(best_keys))
            
            for idx in swap_order:
                for i in range(len(Farey_Sequence) - 1):
                    if self.stop_condition(best_cost, metaheuristic_name, -1):
                        return best_keys

                    new_keys = copy.deepcopy(best_keys)
                    new_keys[idx] = random.uniform(Farey_Sequence[i], Farey_Sequence[i+1])
                    new_cost = self.env.cost(self.env.decoder(new_keys))
                    
                    if new_cost < best_cost:
                        best_keys = new_keys
                        best_cost = new_cost
 
            return best_keys
            
        elif self.LS_type == 'First':
            swap_order = list(range(int(self.rate * self.__MAX_KEYS)))
            random.shuffle(swap_order)
            
            best_keys = copy.deepcopy(keys)
            best_cost = self.env.cost(self.env.decoder(best_keys))
            
            for idx in swap_order:
                for i in range(len(Farey_Sequence) - 1):
                    if self.stop_condition(best_cost, metaheuristic_name, -1):
                        return best_keys
                        
                    new_keys = copy.deepcopy(best_keys)
                    new_keys[idx] = random.uniform(Farey_Sequence[i], Farey_Sequence[i+1])
                    new_cost = self.env.cost(self.env.decoder(new_keys))
                    
                    if new_cost < best_cost:
                        return new_keys
                        
            return best_keys
    
    # A local search that improves a solution by inverting key values (k -> 1-k).
    def InvertLS(self, keys, metaheuristic_name="InvertLS"):
        if self.LS_type == 'Best':
            swap_order = list(range(int(self.__MAX_KEYS)))
            random.shuffle(swap_order)
            
            best_keys = copy.deepcopy(keys)
            best_cost = self.env.cost(self.env.decoder(best_keys))
            
            blocks = []
            while swap_order:
                block = swap_order[:int(self.rate * self.__MAX_KEYS)]
                swap_order = swap_order[int(self.rate * self.__MAX_KEYS):]
                blocks.append(block)

            for block in blocks:
                if self.stop_condition(best_cost, metaheuristic_name, -1):
                    return best_keys

                new_keys = copy.deepcopy(best_keys)
                for idx in block:
                    new_keys[idx] = 1 - new_keys[idx]
                
                new_cost = self.env.cost(self.env.decoder(new_keys))
                
                if new_cost < best_cost:
                    best_keys = new_keys
                    best_cost = new_cost
            
            return best_keys
        
        elif self.LS_type == 'First':
            swap_order = list(range(int(self.rate * self.__MAX_KEYS)))
            random.shuffle(swap_order)
            
            best_keys = copy.deepcopy(keys)
            best_cost = self.env.cost(self.env.decoder(best_keys))
            
            for idx in swap_order:
                if self.stop_condition(best_cost, metaheuristic_name, -1):
                    return best_keys
                        
                new_keys = copy.deepcopy(best_keys)
                new_keys[idx] = 1 - new_keys[idx]
                new_cost = self.env.cost(self.env.decoder(new_keys))
                    
                if new_cost < best_cost:
                    return new_keys
                
            return best_keys
    
    # Crossover operator to combine two parent key vectors.
    def Blending(self, keys1, keys2, factor):
        new_keys = np.zeros(self.__MAX_KEYS)
        
        for i in range(self.__MAX_KEYS):
            if random.random() < 0.02:
                new_keys[i] = random.random()
            else:
                if random.random() < 0.5:
                    new_keys[i] = keys1[i]
                else:
                    if factor == -1:
                        new_keys[i] = max(0.0, min(1.0 - keys2[i], 0.9999999))
                    else:
                        new_keys[i] = keys2[i]
        
        return new_keys
    
    # An advanced local search based on the Nelder-Mead simplex method.
    def NelderMeadSearch(self, keys, pool = None, metaheuristic_name="NelderMeadSearch"):
        improved = 0
        improvedX1 = 0
        keys_origem = copy.deepcopy(keys)
        
        x1 = copy.deepcopy(keys)
        
        if pool is None or len(pool.pool) < 2:
            x2 = self.random_keys()
            x3 = self.random_keys()
        else:
            x2, x3 = random.sample(list(pool.pool), 2)
            x2, x3 = x2[1], x3[1] # Extract keys from tuple
        
        fit1 = self.env.cost(self.env.decoder(x1))
        fit2 = self.env.cost(self.env.decoder(x2))
        fit3 = self.env.cost(self.env.decoder(x3))
        
        # Sort points by fitness
        points = sorted([(fit1, x1), (fit2, x2), (fit3, x3)], key=lambda p: p[0])
        fit1, x1 = points[0]
        fit2, x2 = points[1]
        fit3, x3 = points[2]
        
        xBest, fitBest = copy.deepcopy(x1), fit1
        
        x0 = self.Blending(x1, x2, 1)
        fit0 = self.env.cost(self.env.decoder(x0))
        if fit0 < fitBest:
            xBest, fitBest, improved = copy.deepcopy(x0), fit0, 1
            
        iter_count = 1
        max_iter = int(self.__MAX_KEYS * math.exp(-2))
        
        while iter_count <= (max_iter * self.rate):
            if self.stop_condition(fitBest, metaheuristic_name, -1):
                return xBest
            
            shrink = 0
            
            x_r = self.Blending(x0, x3, -1)
            fit_r = self.env.cost(self.env.decoder(x_r))
            if fit_r < fitBest:
                xBest, fitBest, improved, improvedX1 = copy.deepcopy(x_r), fit_r, 1, 1

            if fit_r < fit1:
                x_e = self.Blending(x_r, x0, -1)
                fit_e = self.env.cost(self.env.decoder(x_e))
                if fit_e < fitBest:
                    xBest, fitBest, improved, improvedX1 = copy.deepcopy(x_e), fit_e, 1, 1
                
                if fit_e < fit_r:
                    x3, fit3 = copy.deepcopy(x_e), fit_e
                else:
                    x3, fit3 = copy.deepcopy(x_r), fit_r
            
            elif fit_r < fit2:
                x3, fit3 = copy.deepcopy(x_r), fit_r
            
            else:
                if fit_r < fit3:
                    x_c = self.Blending(x_r, x0, 1)
                    fit_c = self.env.cost(self.env.decoder(x_c))
                    if fit_c < fitBest:
                        xBest, fitBest, improved, improvedX1 = copy.deepcopy(x_c), fit_c, 1, 1
                    
                    if fit_c < fit_r:
                        x3, fit3 = copy.deepcopy(x_c), fit_c
                    else:
                        shrink = 1
                else:
                    x_c = self.Blending(x0, x3, 1)
                    fit_c = self.env.cost(self.env.decoder(x_c))
                    if fit_c < fitBest:
                        xBest, fitBest, improved, improvedX1 = copy.deepcopy(x_c), fit_c, 1, 1
                    
                    if fit_c < fit3:
                        x3, fit3 = copy.deepcopy(x_c), fit_c
                    else:
                        shrink = 1
            
            if shrink:
                x2 = self.Blending(x1, x2, 1)
                fit2 = self.env.cost(self.env.decoder(x2))
                if fit2 < fitBest:
                    xBest, fitBest, improved, improvedX1 = copy.deepcopy(x2), fit2, 1, 1

                x3 = self.Blending(x1, x3, 1)
                fit3 = self.env.cost(self.env.decoder(x3))
                if fit3 < fitBest:
                    xBest, fitBest, improved, improvedX1 = copy.deepcopy(x3), fit3, 1, 1
            
            points = sorted([(fit1, x1), (fit2, x2), (fit3, x3)], key=lambda p: p[0])
            fit1, x1 = points[0]
            fit2, x2 = points[1]
            fit3, x3 = points[2]
            
            x0 = self.Blending(x1, x2, 1)
            fit0 = self.env.cost(self.env.decoder(x0))
            if fit0 < fitBest:
                xBest, fitBest, improved, improvedX1 = copy.deepcopy(x0), fit0, 1, 1
            
            if improved == 1:
                improved = 0
                iter_count = 0
            else:
                iter_count += 1
        
        if improvedX1 == 1:
            return xBest
        else:
            return keys_origem
    
    # Systematically explores different local search heuristics in a random order.
    def RVND(self, keys, pool=None, metaheuristic_name="RVND"):
        best_keys = copy.deepcopy(keys)
        best_cost = self.env.cost(self.env.decoder(best_keys))

        neighborhoods = ['SwapLS', 'NelderMeadSearch', 'FareyLS', 'InvertLS']
        not_used_nb = copy.deepcopy(neighborhoods)
        
        while not_used_nb:
            current_neighborhood = random.choice(not_used_nb)
            
            if current_neighborhood == 'SwapLS':
                new_keys = self.SwapLS(best_keys, metaheuristic_name=metaheuristic_name)
            elif current_neighborhood == 'NelderMeadSearch':
                new_keys = self.NelderMeadSearch(best_keys, pool, metaheuristic_name=metaheuristic_name)
            elif current_neighborhood == 'FareyLS':
                new_keys = self.FareyLS(best_keys, metaheuristic_name=metaheuristic_name)
            elif current_neighborhood == 'InvertLS':
                new_keys = self.InvertLS(best_keys, metaheuristic_name=metaheuristic_name)
                
            new_cost = self.env.cost(self.env.decoder(new_keys))
            
            if new_cost < best_cost:
                best_keys = new_keys
                best_cost = new_cost
                not_used_nb = copy.deepcopy(neighborhoods)
                
                if pool is not None:
                    pool.insert((best_cost, list(best_keys)), metaheuristic_name, -1)
            else:
                not_used_nb.remove(current_neighborhood)
            
            if self.stop_condition(best_cost, metaheuristic_name, -1):
                return best_keys
        
        return best_keys
    
    # A simple metaheuristic that applies local search to multiple random solutions.
    def MultiStart(self, tag, pool):
        metaheuristic_name = f"MS {tag}"
        start_time = time.time()
        tempo_max = self.max_time
        
        keys = self.random_keys()
        best_keys = keys
        solution = self.env.decoder(keys)
        best_cost = self.env.cost(solution)
        
        pool.insert((best_cost, list(best_keys)), metaheuristic_name, tag)
        if self.stop_condition(best_cost, metaheuristic_name, tag, pool=pool):
            return [], best_keys, best_cost
            
        while time.time() - start_time < tempo_max:
            if len(pool.pool) > 0:
                k1 = random.sample(list(pool.pool), 1)[0][1]
            else:
                k1 = self.random_keys()
            
            new_keys = self.shaking(k1, 0.1, 0.3)
            new_keys = self.RVND(metaheuristic_name=metaheuristic_name, pool=pool, keys=new_keys)
            
            new_solution = self.env.decoder(new_keys)
            new_cost = self.env.cost(new_solution)
            
            if new_cost < best_cost:
                best_keys = new_keys
                best_cost = new_cost
                pool.insert((best_cost, list(best_keys)), metaheuristic_name, tag)
            
            if self.stop_condition(best_cost, metaheuristic_name, tag, pool=pool):
                return [], best_keys, best_cost

        final_cost_solution = self.env.decoder(best_keys)
        final_cost_value = self.env.cost(final_cost_solution, True)
        
        return [], best_keys, final_cost_value

    # Implements the Simulated Annealing metaheuristic.
    def SimulatedAnnealing(self, tag, pool):
        metaheuristic_name = f"SA {tag}"
        tempo_max = self.max_time

        q_manager, params = self._setup_parameters(metaheuristic_name, self.env.SA_parameters)
        
        SAmax = params['SAmax']
        Temperatura = params['T0']
        alpha = params['alphaSA']
        beta_min = params['betaMin']
        beta_max = params['betaMax']

        start_time = time.time()
        
        keys = self.random_keys()
        s = keys
        best_keys = keys

        cost = self.env.cost(self.env.decoder(s))
        s_cost = cost
        best_cost = cost
        
        pool.insert((best_cost, list(best_keys)), metaheuristic_name, tag)
        if self.stop_condition(best_cost, metaheuristic_name, tag, pool=pool):
            return [], best_keys, best_cost
        
        while time.time() - start_time < tempo_max:
            if q_manager is not None:
                current_time = time.time() - self.start_time
                new_params = q_manager.select_action(current_time)
                SAmax, Temperatura, alpha, beta_min, beta_max = [new_params[k] for k in ['SAmax', 'T0', 'alphaSA', 'betaMin', 'betaMax']]

            T = Temperatura
            best_cost_in_cycle = best_cost
            improvement_flag = 0

            while T > 0.0001 and (time.time() - start_time < tempo_max):
                iter_at_temp = 0
                best_ofv_in_temp = float('inf')

                while iter_at_temp < SAmax:
                    iter_at_temp += 1
                    
                    new_keys = self.shaking(s, beta_min, beta_max)
                    new_cost = self.env.cost(self.env.decoder(new_keys))
                    
                    if new_cost < best_ofv_in_temp:
                        best_ofv_in_temp = new_cost

                    delta = new_cost - s_cost
                    
                    if delta < 0:
                        s, s_cost = new_keys, new_cost
                        if s_cost < best_cost:
                            best_cost, best_keys, improvement_flag = s_cost, s, 1
                            pool.insert((best_cost, list(best_keys)), metaheuristic_name, tag)
                            if self.stop_condition(best_cost, metaheuristic_name, tag, pool=pool):
                                return [], best_keys, best_cost
                    else:
                        if random.random() < math.exp(-delta / T):
                            s, s_cost = new_keys, new_cost

                s_cost = self.env.cost(self.env.decoder(s))
                if s_cost < best_cost:
                    best_cost, best_keys, improvement_flag = s_cost, s, 1
                    pool.insert((best_cost, list(best_keys)), metaheuristic_name, tag)
                    if self.stop_condition(best_cost, metaheuristic_name, tag, pool=pool):
                        return [], best_keys, best_cost

                T *= alpha
            
            if q_manager:
                reward = 1.0 if improvement_flag else (best_cost_in_cycle - best_ofv_in_temp) / best_cost_in_cycle if best_cost_in_cycle > 0 else 0
                q_manager.update_q_value(reward, time.time() - self.start_time)

        final_cost_solution = self.env.decoder(best_keys)
        final_cost_value = self.env.cost(final_cost_solution)
        
        return [], best_keys, final_cost_value

    # Implements the Variable Neighborhood Search metaheuristic.
    def VNS(self, limit_time, tag, pool):
        metaheuristic_name = f"VNS {tag}"
        
        q_manager, params = self._setup_parameters(metaheuristic_name, self.env.VNS_parameters)
        
        k_max = params['kMax']
        beta_min = params['betaMin']
        
        start_time = time.time()
        
        keys = self.random_keys()
        keys = self.RVND(metaheuristic_name=metaheuristic_name, pool=pool, keys=keys)
        current_s = keys
        current_cost = self.env.cost(self.env.decoder(keys))
        best_cost = current_cost
        best_keys = keys

        pool.insert((best_cost, list(best_keys)), metaheuristic_name, tag)
        if self.stop_condition(best_cost, metaheuristic_name, tag, pool=pool):
            return [], best_keys, best_cost

        while time.time() - start_time < limit_time:
            if q_manager:
                current_time = time.time() - start_time
                new_params = q_manager.select_action(current_time)
                k_max = new_params['kMax']
                beta_min = new_params['betaMin']
            
            k = 1
            improvement_flag_global = 0
            cost_before_vns_cycle = current_cost

            while k <= k_max:
                if self.stop_condition(best_cost, metaheuristic_name, tag, pool=pool):
                    return [], best_keys, best_cost

                s1 = self.shaking(current_s, k * beta_min, (k + 1) * beta_min)
                s2 = self.RVND(metaheuristic_name=metaheuristic_name, pool=pool, keys=s1)
                cost = self.env.cost(self.env.decoder(s2))

                if cost < current_cost:
                    current_s, current_cost, k = s2, cost, 1
                    if cost < best_cost:
                        best_cost, best_keys, improvement_flag_global = cost, s2, 1
                        pool.insert((best_cost, list(best_keys)), metaheuristic_name, tag)
                else:
                    k += 1
            
            if q_manager:
                reward = 1.0 if improvement_flag_global else (cost_before_vns_cycle - current_cost) / cost_before_vns_cycle if cost_before_vns_cycle > 0 else 0
                q_manager.update_q_value(reward, time.time() - start_time)

        final_cost_solution = self.env.decoder(best_keys)
        final_cost_value = self.env.cost(final_cost_solution)
        
        return [], best_keys, final_cost_value

    # Implements the Large Neighborhood Search metaheuristic.
    def LNS(self, limit_time, tag, pool):
        metaheuristic_name = f"LNS {tag}"
        
        q_manager, params = self._setup_parameters(metaheuristic_name, self.env.LNS_parameters)
        
        beta_min = params['betaMin']
        beta_max = params['betaMax']
        T0 = params['TO']
        alphaLNS = params['alphaLNS']

        Farey_Sequence = [0.00, 0.142857, 0.166667, 0.20, 0.25, 0.285714, 0.333333, 0.40, 0.428571, 0.50, 
                                 0.571429, 0.60, 0.666667, 0.714286, 0.75, 0.80, 0.833333, 0.857143, 1.0]
        
        start_time = time.time()
        
        s = self.random_keys()
        s_cost = self.env.cost(self.env.decoder(s))
        best_keys = s
        best_cost = s_cost
        
        pool.insert((best_cost, list(best_keys)), metaheuristic_name, tag)
        if self.stop_condition(best_cost, metaheuristic_name, tag, pool=pool):
            return [], best_keys, best_cost
        
        reanneling = False

        while time.time() - start_time < limit_time:
            if q_manager:
                current_time = time.time() - start_time
                new_params = q_manager.select_action(current_time)
                beta_min, beta_max, T0, alphaLNS = [new_params[k] for k in ['betaMin', 'betaMax', 'TO', 'alphaLNS']]
            
            improvement_flag_cycle = 0
            cost_before_cycle = best_cost

            T = T0 if not reanneling else T0 * 0.3
            
            while T > 0.01 and (time.time() - start_time < limit_time):
                s_line = copy.deepcopy(s)
                
                intensity = int(random.uniform(beta_min * self.__MAX_KEYS, beta_max * self.__MAX_KEYS))
                RKorder = list(range(self.__MAX_KEYS))
                random.shuffle(RKorder)
                
                for k in range(intensity):
                    pos = RKorder[k]
                    rkBestCost = float('inf')
                    best_rk_val = s_line[pos]

                    for j in range(len(Farey_Sequence) - 1):
                        if self.stop_condition(best_cost, metaheuristic_name, tag, pool=pool): return [], best_keys, best_cost
                        
                        s_line[pos] = random.uniform(Farey_Sequence[j], Farey_Sequence[j+1])
                        new_cost = self.env.cost(self.env.decoder(s_line))
                        
                        if new_cost < rkBestCost:
                            rkBestCost = new_cost
                            best_rk_val = s_line[pos]
                    
                    s_line[pos] = best_rk_val
                
                s_best_line = self.NelderMeadSearch(keys=s_line, pool=pool)
                best_cost_s1 = self.env.cost(self.env.decoder(s_best_line))
                
                delta = best_cost_s1 - s_cost
                if delta <= 0:
                    s, s_cost = s_best_line, best_cost_s1
                    if s_cost < best_cost:
                        best_cost, best_keys, improvement_flag_cycle = s_cost, s, 1
                        pool.insert((best_cost, list(best_keys)), metaheuristic_name, tag)
                else:
                    if random.random() < math.exp(-delta / T):
                        s, s_cost = s_best_line, best_cost_s1
                
                T *= alphaLNS

            reanneling = True

            if q_manager:
                reward = 1.0 if improvement_flag_cycle else (cost_before_cycle - best_cost) / cost_before_cycle if cost_before_cycle > 0 else 0
                q_manager.update_q_value(reward, time.time() - start_time)
            
            if self.stop_condition(best_cost, metaheuristic_name, tag, pool=pool):
                return [], best_keys, best_cost

        final_cost_solution = self.env.decoder(best_keys)
        final_cost_value = self.env.cost(final_cost_solution)
        
        return [], best_keys, final_cost_value

    # Implements the Particle Swarm Optimization metaheuristic.
    def PSO(self, tag, pool):
        metaheuristic_name = f"PSO {tag}"
        limit_time = self.max_time

        q_manager, params = self._setup_parameters(metaheuristic_name, self.env.PSO_parameters)
        
        Psize = params['PSize']
        c1 = params['c1']
        c2 = params['c2']
        w = params['w']

        start_time = time.time()
        
        X = [self.random_keys() for _ in range(Psize)]
        Pbest = [None] * Psize
        V = [np.random.random(self.__MAX_KEYS) for _ in range(Psize)]
        
        Gbest_keys = None
        Gbest_cost = float('inf')

        for i in range(Psize):
            cost_x = self.env.cost(self.env.decoder(X[i]))
            Pbest[i] = (cost_x, X[i])
            if cost_x < Gbest_cost:
                Gbest_cost, Gbest_keys = cost_x, X[i]

        pool.insert((Gbest_cost, list(Gbest_keys)), metaheuristic_name, tag)
        if self.stop_condition(Gbest_cost, metaheuristic_name, tag, pool=pool):
            return [], Gbest_keys, Gbest_cost
        
        while time.time() - start_time < limit_time:
            if q_manager:
                current_time = time.time() - self.start_time
                new_params = q_manager.select_action(current_time)
                
                new_Psize = new_params['PSize']
                c1, c2, w = [new_params[k] for k in ['c1', 'c2', 'w']]

                if new_Psize != Psize:
                    if new_Psize < Psize:
                        sorted_indices = sorted(range(Psize), key=lambda i: Pbest[i][0])
                        X, Pbest, V = [[X[i], Pbest[i], V[i]][j] for i in sorted_indices[:new_Psize] for j in range(3)]
                    else:
                        num_new = new_Psize - Psize
                        for _ in range(num_new):
                            new_keys = self.random_keys()
                            new_cost = self.env.cost(self.env.decoder(new_keys))
                            X.append(new_keys)
                            Pbest.append((new_cost, new_keys))
                            V.append(np.random.random(self.__MAX_KEYS))
                            if new_cost < Gbest_cost:
                                Gbest_cost, Gbest_keys = new_cost, new_keys
                    Psize = new_Psize
            
            best_cost_in_generation = float('inf')
            improvement_flag = 0

            for i in range(Psize):
                if self.stop_condition(Gbest_cost, metaheuristic_name, tag, pool=pool):
                    return [], Gbest_keys, Gbest_cost
                
                r1, r2 = random.random(), random.random()
                
                V[i] = w * V[i] + c1 * r1 * (Pbest[i][1] - X[i]) + c2 * r2 * (Gbest_keys - X[i])
                
                old_keys = X[i]
                X[i] = X[i] + V[i]
                
                for j in range(self.__MAX_KEYS):
                    if not (0.0 <= X[i][j] < 1.0):
                        X[i][j], V[i][j] = old_keys[j], 0.0

                cost_x = self.env.cost(self.env.decoder(X[i]))
                
                if cost_x < Pbest[i][0]:
                    Pbest[i] = (cost_x, X[i])
                
                if cost_x < Gbest_cost:
                    Gbest_cost, Gbest_keys, improvement_flag = cost_x, X[i], 1
                    pool.insert((Gbest_cost, list(Gbest_keys)), metaheuristic_name, tag)
                
                if cost_x < best_cost_in_generation:
                    best_cost_in_generation = cost_x

            if Psize > 0:
                chosen_pbest_index = random.randint(0, Psize - 1)
                improved_keys = self.NelderMeadSearch(keys=Pbest[chosen_pbest_index][1], pool=pool)
                improved_cost = self.env.cost(self.env.decoder(improved_keys))
                
                if improved_cost < best_cost_in_generation:
                    best_cost_in_generation = improved_cost

                if improved_cost < Gbest_cost:
                    Gbest_keys, Gbest_cost, improvement_flag = improved_keys, improved_cost, 1
                    pool.insert((Gbest_cost, list(Gbest_keys)), metaheuristic_name, tag)

            if q_manager:
                reward = 1.0 if improvement_flag else (Gbest_cost - best_cost_in_generation) / Gbest_cost if Gbest_cost > 0 else 0
                q_manager.update_q_value(reward, time.time() - start_time)
        
        return [], Gbest_keys, Gbest_cost

    # Implements the Biased Random-Key Genetic Algorithm.
    def BRKGA(self, tag, pool):
        metaheuristic_name = f"BRKGA {tag}"
        limit_time = self.max_time
        generation = 0
        half_time_restart_done = False

        q_manager, params = self._setup_parameters(metaheuristic_name, self.env.BRKGA_parameters)
        pop_size = params['p']
        elite_pop = params['pe']
        chance_elite = params['rhoe']
        tam_elite = max(1, int(pop_size * elite_pop))

        population = [self.random_keys() for _ in range(pop_size)]
        best_keys_overall = None
        best_fitness_overall = float('inf')

        start_time = time.time()

        while time.time() - start_time < limit_time:
            resize_pending = None

            if q_manager:
                elapsed = time.time() - start_time
                new_params = q_manager.select_action(elapsed)

                elite_pop = new_params.get('pe', elite_pop)
                chance_elite = new_params.get('rhoe', chance_elite)
                tam_elite = max(1, int(pop_size * elite_pop)) if elite_pop > 0 else 0

                new_pop_size = new_params.get('p', pop_size)
                if new_pop_size != pop_size:
                    resize_pending = new_pop_size

            if not half_time_restart_done and (time.time() - start_time) > (limit_time / 2):
                population = [self.random_keys() for _ in range(pop_size)]
                half_time_restart_done = True

            generation += 1

            evaluated_population = []
            best_fitness_in_generation = float('inf')
            improvement_flag = 0

            for key in population:
                sol = self.env.decoder(key)
                fitness = self.env.cost(sol)
                evaluated_population.append((key, sol, fitness))

                if fitness < best_fitness_in_generation:
                    best_fitness_in_generation = fitness

                if fitness < best_fitness_overall:
                    best_fitness_overall, best_keys_overall, improvement_flag = fitness, key, 1
                    pool.insert((best_fitness_overall, list(best_keys_overall)), metaheuristic_name, tag)

                if self.stop_condition(best_fitness_overall, metaheuristic_name, tag):
                    return [], best_keys_overall, fitness

            if resize_pending is not None:
                if resize_pending < pop_size:
                    evaluated_population.sort(key=lambda x: x[2])
                    population = [ind[0] for ind in evaluated_population[:resize_pending]]
                else:
                    num_new = resize_pending - pop_size
                    population.extend([self.random_keys() for _ in range(num_new)])
                pop_size = resize_pending
                tam_elite = max(1, int(pop_size * elite_pop)) if elite_pop > 0 else 0
                resize_pending = None

            evaluated_population.sort(key=lambda x: x[2])
            elite_keys = [item[0] for item in evaluated_population[:tam_elite]] if tam_elite > 0 else []

            new_population = elite_keys[:1] if elite_keys else []

            while len(new_population) < pop_size:
                parent1 = random.choice(population)
                parent2 = random.choice(elite_keys) if len(elite_keys) > 0 and random.random() < 0.5 and len(pool.pool) > 0 else random.choice(population)

                child = np.zeros(self.__MAX_KEYS)
                for i in range(len(child)):
                    child[i] = parent2[i] if random.random() < chance_elite else parent1[i]

                for idx in range(len(child)):
                    if random.random() < 0.05:
                        child[idx] = random.random()

                new_population.append(child)

            population = new_population[:pop_size]

            if q_manager:
                elapsed = time.time() - start_time
                reward = 1.0 + (1.0 / pop_size) if improvement_flag else ((best_fitness_overall - best_fitness_in_generation) / best_fitness_overall if best_fitness_overall not in [0, float('inf')] else 0.0)
                q_manager.update_q_value(reward, elapsed)

        final_cost_solution = self.env.decoder(best_keys_overall)
        final_cost_value = self.env.cost(final_cost_solution)
        return [], best_keys_overall, best_fitness_overall
    
    # Implements the Iterated Local Search metaheuristic.
    def ILS(self, limit_time, tag, pool):
        metaheuristic_name = f"ILS {tag}"
        
        q_manager, params = self._setup_parameters(metaheuristic_name, self.env.ILS_parameters)
        
        beta_min = params['betaMin']
        beta_max = params['betaMax']
        
        start_time = time.time()
        
        keys = self.random_keys()
        keys = self.RVND(metaheuristic_name=metaheuristic_name, pool=pool, keys=keys)
        best_keys = keys
        best_cost = self.env.cost(self.env.decoder(keys))
        
        pool.insert((best_cost, list(best_keys)), metaheuristic_name, tag)
        if self.stop_condition(best_cost, metaheuristic_name, tag, pool = pool):
            return [], best_keys, best_cost

        while time.time() - start_time < limit_time:
            if q_manager:
                current_time = time.time() - self.start_time
                new_params = q_manager.select_action(current_time)
                beta_min = new_params['betaMin']
                beta_max = new_params['betaMax']
            
            cost_before_iteration = best_cost
            improvement_flag = 0
            
            s1 = self.shaking(best_keys, beta_min, beta_max)
            s2 = self.RVND(metaheuristic_name=metaheuristic_name, pool=pool, keys=s1)
            
            sol2 = self.env.decoder(s2)
            cost = self.env.cost(sol2)

            if cost < best_cost:
                improvement_flag = 1
            
            if cost <= best_cost:
                best_cost, best_keys = cost, s2
                pool.insert((best_cost, list(best_keys)), metaheuristic_name, tag)

                if self.stop_condition(best_cost, metaheuristic_name, tag, pool = pool):
                    return [], best_keys, best_cost
            
            if q_manager:
                reward = 1.0 if improvement_flag else (cost_before_iteration - cost) / cost_before_iteration if cost_before_iteration > 0 else 0
                q_manager.update_q_value(reward, time.time() - self.start_time)

        final_cost_solution = self.env.decoder(best_keys)
        final_cost_value = self.env.cost(final_cost_solution)
        
        return [], best_keys, final_cost_value
    
    # Implements the Genetic Algorithm metaheuristic.
    def GA(self, tag, pool):
        metaheuristic_name = f"GA {tag}"
        limit_time = self.max_time
        
        q_manager, params = self._setup_parameters(metaheuristic_name, self.env.GA_parameters)
        
        pop_size = params['sizePop']
        prob_cros = params['probCros']
        prob_mut = params['probMut']

        population = []
        best_keys_overall = None
        best_fitness_overall = float('inf')

        for _ in range(pop_size):
            keys = self.random_keys()
            cost = self.env.cost(self.env.decoder(keys))
            population.append({'keys': keys, 'cost': cost})
            if cost < best_fitness_overall:
                best_fitness_overall, best_keys_overall = cost, keys

        pool.insert((best_fitness_overall, list(best_keys_overall)), metaheuristic_name, tag)
        if self.stop_condition(best_fitness_overall, metaheuristic_name, tag, pool=pool):
            return [], best_keys_overall, best_fitness_overall
        
        start_time = time.time()
        num_generations = 0

        while time.time() - start_time < limit_time:
            num_generations += 1
            
            if q_manager:
                current_time = time.time() - self.start_time
                new_params = q_manager.select_action(current_time)
                
                new_pop_size = new_params['sizePop']
                prob_cros = new_params['probCros']
                prob_mut = new_params['probMut']
                
                if new_pop_size != pop_size:
                    population.sort(key=lambda p: p['cost'])
                    if new_pop_size < pop_size:
                        population = population[:new_pop_size]
                    else:
                        num_new = new_pop_size - pop_size
                        for _ in range(num_new):
                            new_keys = self.random_keys()
                            new_cost = self.env.cost(self.env.decoder(new_keys))
                            population.append({'keys': new_keys, 'cost': new_cost})
                    pop_size = new_pop_size

            parents = [min(random.sample(population, 3), key=lambda p: p['cost']) for _ in range(pop_size)]
            
            new_population_data = []
            best_of_current_gen_cost = float('inf')
            improvement_flag = 0

            for i in range(0, pop_size, 2):
                if i + 1 >= len(parents):
                    new_population_data.append(parents[i])
                    continue

                parent1_keys, parent2_keys = parents[i]['keys'], parents[i+1]['keys']
                child1_keys, child2_keys = copy.deepcopy(parent1_keys), copy.deepcopy(parent2_keys)

                if random.random() < prob_cros:
                    for j in range(self.__MAX_KEYS):
                        if random.random() < 0.5:
                            child1_keys[j], child2_keys[j] = child2_keys[j], child1_keys[j]
                    
                    for j in range(self.__MAX_KEYS):
                        if random.random() <= prob_mut: child1_keys[j] = random.random()
                        if random.random() <= prob_mut: child2_keys[j] = random.random()
                
                cost1 = self.env.cost(self.env.decoder(child1_keys))
                cost2 = self.env.cost(self.env.decoder(child2_keys))
                
                new_population_data.extend([{'keys': child1_keys, 'cost': cost1}, {'keys': child2_keys, 'cost': cost2}])
                
                if cost1 < best_of_current_gen_cost: best_of_current_gen_cost = cost1
                if cost2 < best_of_current_gen_cost: best_of_current_gen_cost = cost2

            if new_population_data:
                new_population_data.sort(key=lambda p: p['cost'])
                best_new_individual = new_population_data[0]
                
                improved_keys = self.RVND(keys=best_new_individual['keys'], pool=pool, metaheuristic_name=metaheuristic_name)
                improved_cost = self.env.cost(self.env.decoder(improved_keys))
                
                if improved_cost < best_new_individual['cost']:
                    best_new_individual['keys'], best_new_individual['cost'] = improved_keys, improved_cost
                    if improved_cost < best_of_current_gen_cost:
                        best_of_current_gen_cost = improved_cost
            
            population.sort(key=lambda p: p['cost'])
            if new_population_data and population[0]['cost'] < new_population_data[0]['cost']:
                new_population_data[-1] = population[0]
            
            population = new_population_data
            
            current_best_gen_individual = min(population, key=lambda p: p['cost'])
            if current_best_gen_individual['cost'] < best_fitness_overall:
                best_fitness_overall, best_keys_overall, improvement_flag = current_best_gen_individual['cost'], current_best_gen_individual['keys'], 1
                pool.insert((best_fitness_overall, list(best_keys_overall)), metaheuristic_name, tag)

                if self.stop_condition(best_fitness_overall, metaheuristic_name, tag, pool = pool):
                    return [], best_keys_overall, best_fitness_overall

            if q_manager:
                reward = 1.0 if improvement_flag else (best_fitness_overall - best_of_current_gen_cost) / best_fitness_overall if best_fitness_overall > 0 else 0
                q_manager.update_q_value(reward, time.time() - start_time)
        
        final_cost_solution = self.env.decoder(best_keys_overall)
        final_cost_value = self.env.cost(final_cost_solution)
        
        return [], best_keys_overall, final_cost_value

    # Checks for termination conditions (time limit or optimal solution found).
    def stop_condition(self, best_cost, metaheuristic_name, tag, pool = None):
        if time.time() - self.start_time > self.max_time:
            if self.print_best and tag != -1:
                print(f"{metaheuristic_name}: FINISHED")
            return True
        
        if pool is not None and self.env.dict_best is not None and pool.best_pair[0] == self.env.dict_best.get(self.env.instance_name):
            print(f"Optimal solution found by another thread: {pool.best_pair[0]}")
            return True

        if self.env.dict_best is not None and best_cost == self.env.dict_best.get(self.env.instance_name):
            if self.print_best and tag != -1:
                print(f"Metaheuristic {metaheuristic_name} found the optimal solution: {best_cost}")
            return True
        
        return False
        
    # Main entry point to run the RKO solver with parallel metaheuristics.
    def solve(self, time_total, brkga=0, ms=0, sa=0, vns=0, ils=0, lns=0, pso=0, ga=0, restart=1, runs=1):
        solutions, times, costs = [], [], []
        for i in range(runs):
            print(f'Instance: {self.env.instance_name}, Run: {i+1}/{runs}')
            limit_time = time_total * restart
            restarts = int(1/restart)
            
            self.max_time = limit_time
            
            manager = Manager()
            shared = manager.Namespace()
            
            shared.best_pair = manager.list([float('inf'), None, None])
            shared.best_pool = manager.list()
            
            shared.pool = SolutionPool(20, shared.best_pool, shared.best_pair, lock=manager.Lock(), print=self.print_best, Best=self.env.dict_best.get(self.env.instance_name), env=self.env)
            for _ in range(20):
                keys = self.random_keys()
                cost = self.env.cost(self.env.decoder(keys))
                shared.pool.insert((cost, list(keys)), 'pool', -1)
            
            lock = manager.Lock()
            
            for k in range(restarts):
                processes = []
                tag = 0
                self.start_time = time.time()
                if self.stop_condition(shared.pool.best_pair[0], 'RKO', tag, pool=shared.pool):
                    break
                
                shared.pool.pool = manager.list()
                
                # Worker definitions
                worker_map = {
                    'brkga': (_brkga_worker, brkga), 'ms': (_MS_worker, ms),
                    'sa': (_SA_worker, sa), 'vns': (_VNS_worker, vns),
                    'ils': (_ILS_worker, ils), 'lns': (_LNS_worker, lns),
                    'pso': (_PSO_worker, pso), 'ga': (_GA_worker, ga)
                }
                
                for name, (worker_func, count) in worker_map.items():
                    for _ in range(count):
                        args = (self.env, shared.pool, tag, self.print_best, self.save_directory)
                        if name in ['vns', 'ils', 'lns']:
                            args = (self.env, self.max_time, *args[1:])
                        
                        p = Process(target=worker_func, args=args)
                        tag += 1
                        processes.append(p)
                        p.start()

                for p in processes:
                    p.join(timeout=self.max_time)
                
                for p in processes:
                    if p.is_alive():
                        print(f"Process {p.pid} timed out and will be terminated.")
                        p.terminate()

            cost, solution, time_elapsed = shared.pool.best_pair
            solutions.append(solution)
            costs.append(round(cost, 2))
            times.append(round(time_elapsed, 2))
            
        if self.save_directory:
            directory = os.path.dirname(self.save_directory)
            if directory:
                os.makedirs(directory, exist_ok=True)
            with open(self.save_directory, 'a', newline='') as f:
                f.write(f'{time_total}, {self.env.instance_name}, {round(sum(costs)/len(costs),2)}, {costs}, {round(sum(times)/len(times),2)}, {times}\n')
        
        return cost, solution, time_elapsed
        
# Worker functions for multiprocessing
def _brkga_worker(env, pool, tag, print_best, save_directory):
    runner = RKO(env, print_best, save_directory)
    runner.BRKGA(tag, pool)
    
def _MS_worker(env, pool, tag, print_best, save_directory):
    runner = RKO(env, print_best, save_directory)
    runner.MultiStart(tag, pool)
    
def _GRASP_worker(env, pool, tag, print_best, save_directory):
    runner = RKO(env, print_best, save_directory)
    runner.MultiStart(tag, pool)
    
def _VNS_worker(env, limit_time, pool, tag, print_best, save_directory):
    runner = RKO(env, print_best, save_directory)
    runner.VNS(limit_time, tag, pool)
    
def _ILS_worker(env, limit_time, pool, tag, print_best, save_directory):
    runner = RKO(env, print_best, save_directory)
    runner.ILS(limit_time, tag, pool)
    
def _SA_worker(env, pool, tag, print_best, save_directory):
    runner = RKO(env, print_best, save_directory)
    runner.SimulatedAnnealing(tag=tag, pool=pool)
    
def _LNS_worker(env, limit_time, pool, tag, print_best, save_directory):
    runner = RKO(env, print_best, save_directory)
    runner.LNS(limit_time=limit_time, tag=tag, pool=pool)
    
def _PSO_worker(env, pool, tag, print_best, save_directory):
    runner = RKO(env, print_best, save_directory)
    runner.PSO(tag=tag, pool=pool)
    
def _GA_worker(env, pool, tag, print_best, save_directory):
    runner = RKO(env, print_best, save_directory)
    runner.GA(tag=tag, pool=pool)

# Manages online parameter control using Q-Learning.
class QLearningManager:
    def __init__(self, parameters_config, max_time, metaheuristic_name, save_report=True, epsilon_min=0.1, df=0.8):
        self.param_keys = list(parameters_config.keys())
        self.max_time = max_time if max_time > 0 else 1.0
        self.epsilon_min = epsilon_min
        self.df = df
        self.metaheuristic_name = metaheuristic_name
        self.save_report = save_report

        self._create_states_and_actions(parameters_config)

        self.current_state_idx = random.randint(0, self.num_states - 1)
        self.last_action_taken = None
        self.epsilon_max = 1.0
        self.restart_epsilon_period = self.max_time * 0.1
        self.next_restart_time = self.restart_epsilon_period

    # Creates all possible parameter combinations (states) and defines possible transitions (actions).
    def _create_states_and_actions(self, parameters_config):
        param_values = list(parameters_config.values())
        all_combinations = list(itertools.product(*param_values))
        self.num_states = len(all_combinations)

        self.states = [{'id': i, 'params': dict(zip(self.param_keys, combo))} for i, combo in enumerate(all_combinations)]

        self.q_table = {}
        for i in range(self.num_states):
            self.q_table[i] = {}
            for j in range(self.num_states):
                hamming_dist = sum(1 for k in self.param_keys if self.states[i]['params'][k] != self.states[j]['params'][k])
                if hamming_dist <= 1:
                    self.q_table[i][j] = random.uniform(0.01, 0.05)
        
        for i in range(self.num_states):
            self._update_max_q(i)

    # Updates the maximum Q-value for a given state.
    def _update_max_q(self, state_idx):
        q_values_for_state = self.q_table.get(state_idx, {})
        self.states[state_idx]['max_q'] = max(q_values_for_state.values()) if q_values_for_state else 0.0

    # Returns the current set of parameters.
    def get_current_parameters(self):
        return self.states[self.current_state_idx]['params']

    # Selects an action (a new set of parameters) using an epsilon-greedy policy.
    def select_action(self, current_time):
        if self.restart_epsilon_period > 0 and current_time > self.next_restart_time:
            self.next_restart_time += self.restart_epsilon_period
            self.epsilon_max = max(self.epsilon_min, self.epsilon_max - 0.1)

        time_in_period = current_time % self.restart_epsilon_period if self.restart_epsilon_period > 0 else 0
        epsilon = self.epsilon_min + 0.5 * (self.epsilon_max - self.epsilon_min) * (1 + math.cos(time_in_period / self.restart_epsilon_period * math.pi))

        possible_actions = list(self.q_table[self.current_state_idx].keys())
        if not possible_actions: return self.get_current_parameters()
        
        action = max(self.q_table[self.current_state_idx], key=self.q_table[self.current_state_idx].get) if random.random() > epsilon else random.choice(possible_actions)

        self.last_action_taken = action
        return self.states[action]['params']

    # Updates the Q-table based on the reward received.
    def update_q_value(self, reward, current_time):
        lf = max(0.1, 1.0 - (0.9 * current_time / self.max_time))
        
        state_idx, action_idx = self.current_state_idx, self.last_action_taken
        if action_idx is None: return

        next_state_idx = action_idx
        
        old_q_value = self.q_table[state_idx][action_idx]
        future_best_q = self.states[next_state_idx]['max_q']
        
        new_q_value = old_q_value + lf * (reward + self.df * future_best_q - old_q_value)
        self.q_table[state_idx][action_idx] = new_q_value
        
        self._update_max_q(state_idx)
        
        self.current_state_idx = next_state_idx
    
    # Saves a final report of the learned Q-table policy.
    def save_final_policy_report(self, instance_name, directory='.'):
        if not self.save_report:
            return

        if not os.path.exists(directory):
            os.makedirs(directory)
            
        filepath = os.path.join(directory, f"policy_report_{self.metaheuristic_name.replace(' ', '_')}_{instance_name}.csv")
        
        report_data = []
        for state_idx, actions in self.q_table.items():
            state_params = self.states[state_idx]['params']
            
            best_action_for_state = max(actions, key=actions.get) if actions else None
            best_params_for_state = self.states[best_action_for_state]['params'] if best_action_for_state is not None else {}

            entry = {f'state_{k}': v for k, v in state_params.items()}
            entry['state_id'] = state_idx
            entry['best_next_state_id'] = best_action_for_state
            entry.update({f'best_param_{k}': v for k, v in best_params_for_state.items()})
            entry.update({f'Q(s,{a})': round(q, 4) for a, q in actions.items()})
            report_data.append(entry)

        try:
            df = pd.DataFrame(report_data).sort_values(by='state_id').set_index('state_id')
            df.to_csv(filepath)
            print(f"Q-Learning convergence report saved to: {filepath}")
        except Exception as e:
            print(f"Error saving convergence report to {filepath}: {e}")
