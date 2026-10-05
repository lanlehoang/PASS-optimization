import numpy as np
import json
import random
import time

from .eoh_interface_ec import InterfaceEC
from .logger import EvolLogger


from src.utils.generators import generate_random, set_global_seed

class EOH:
    def __init__(self, params, problem, **kwargs):
        self.prob = problem

        # Initialize logger
        self.logger = EvolLogger(params.exp_output_path)

        # LLM settings
        self.use_local_llm = params.llm_use_local
        self.llm_local_url = params.llm_local_url
        self.api_endpoint = params.llm_api_endpoint
        self.api_key = params.llm_api_key
        self.llm_model = params.llm_model

        # Experimental settings
        self.pop_size = params.ec_pop_size
        self.n_pop = params.ec_n_pop
        self.operators = params.ec_operators
        self.operator_weights = params.ec_operator_weights

        if params.ec_m > self.pop_size or params.ec_m == 1:
            print("m should not be larger than pop size or smaller than 2, adjust it to m=2")
            params.ec_m = 2
        self.m = params.ec_m

        self.debug_mode = params.exp_debug_mode
        self.output_path = params.exp_output_path
        self.exp_n_proc = params.exp_n_proc
        self.timeout = params.eva_timeout
        self.use_numba = params.eva_numba_decorator

        print("- EoH parameters loaded -")
        set_global_seed(2024)

    def add2pop(self, population, offspring):
        """Add new individuals to population (with duplication check)"""
        for off in offspring:
            # Check for duplicates
            is_duplicate = any(ind["objective"] == off["objective"] for ind in population)
            if is_duplicate and self.debug_mode:
                print("duplicated result, retrying ... ")
            population.append(off)

    def _simple_selection(self, population, size):
        """Simple selection: sort by objective (higher is better) and keep top N"""
        # Filter out None objectives
        valid_pop = [ind for ind in population if ind["objective"] is not None]
        # Sort descending (higher objective = better)
        valid_pop.sort(key=lambda x: x["objective"], reverse=True)
        return valid_pop[:size]

    def run(self):
        print("- Evolution Start -")
        time_start = time.time()

        # Interface for EC operators
        interface_ec = InterfaceEC(
            self.pop_size,
            self.m,
            self.api_endpoint,
            self.api_key,
            self.llm_model,
            self.use_local_llm,
            self.llm_local_url,
            self.debug_mode,
            self.prob,
            n_p=self.exp_n_proc,
            timeout=self.timeout,
            use_numba=self.use_numba,
        )

        # Create initial population
        print("creating initial population:")
        population = interface_ec.population_generation()
        population = self._simple_selection(population, self.pop_size)

        print(f"Pop initial: ")
        for off in population:
            print(" Obj: ", off["objective"], end="|")
        print()
        print("initial population has been created!")

        # Save initial population
        filename = self.output_path + "/pops/population_generation_0.json"
        with open(filename, "w") as f:
            json.dump(population, f, indent=5)

        # Log initial population
        for idx, ind in enumerate(population):
            self.logger.log_individual(0, "i1", idx, ind)
        self.logger.log_best(0, population[0])

        # Main evolution loop
        n_op = len(self.operators)

        for gen in range(self.n_pop):
            for i in range(n_op):
                op = self.operators[i]
                print(f" OP: {op}, [{i + 1} / {n_op}] ", end="|")

                op_w = self.operator_weights[i]
                if generate_random() < op_w:
                    parents, offsprings = interface_ec.get_algorithm(population, op)
                    # Log each offspring
                    for idx, off in enumerate(offsprings):
                        self.logger.log_individual(gen + 1, op, idx, off)
                    self.add2pop(population, offsprings)

                    for off in offsprings:
                        print(" Obj: ", off["objective"], end="|")

                # Selection: keep best individuals
                population = self._simple_selection(population, self.pop_size)

            # Log best individual of this generation
            self.logger.log_best(gen + 1, population[0])

            # Save population
            filename = self.output_path + f"/pops/population_generation_{gen + 1}.json"
            with open(filename, "w") as f:
                json.dump(population, f, indent=5)

            # Save best individual
            filename = self.output_path + f"/pops_best/population_generation_{gen + 1}.json"
            with open(filename, "w") as f:
                json.dump(population[0], f, indent=5)

            # Progress report
            elapsed = (time.time() - time_start) / 60
            print(f"--- {gen + 1} of {self.n_pop} generations finished. Time Cost: {elapsed:.1f} m")
            print("Pop Objs: ", end=" ")
            for ind in population:
                print(str(ind["objective"]) + " ", end="")

        # Save logs to CSV at the end
        self.logger.save_to_csv()
