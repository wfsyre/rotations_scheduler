import random
import datetime

import pandas
from constraint import *
from pandas.tseries.holiday import USFederalHolidayCalendar
import pandas as pd
import pygad
import numpy as np
import math
# This is a sample Python script.

rotations = pd.read_csv("rotations.csv")
solution_index_to_date = {}
solution_variable_to_name = {}
solution_variable_to_name[-2] = "None"


def get_cover_days(year):
    dates = []
    pairs = []
    cal = USFederalHolidayCalendar()
    holidays = cal.holidays(start=str(year)+'-01-01', end=str(year)+'-12-31').to_pydatetime()
    covered = set()
    for holiday in holidays:
        # could pair holidays with the nearest weekend here if you wanted.
        # Right now I only pair if the holiday is a monday or a friday
        if holiday.weekday() == 4:
            sat = holiday + datetime.timedelta(days=1)
            sun = holiday + datetime.timedelta(days=2)
            days = [holiday, sat, sun]
            days = [day for day in days if day.year == year]
            covered.update(days)
            pairs.append(tuple(days))
            dates += days
        elif holiday.weekday() == 0:
            sat = holiday + datetime.timedelta(days=-2)
            sun = holiday + datetime.timedelta(days=-1)
            days = [sat, sun, holiday]
            days = [day for day in days if day.year == year]
            covered.update(days)
            pairs.append(tuple(days))
            dates += days

    d = datetime.datetime(year, 1, 1, 0, 0)  # January 1st
    d += datetime.timedelta(days=6 - d.weekday())  # First Sunday
    while d.year == year:
        if d not in covered:
            yesterday = d + datetime.timedelta(days=-1)
            if yesterday.year == year:
                covered.add(yesterday)
                dates.append(d + datetime.timedelta(days=-1))
                pairs.append(((d + datetime.timedelta(days=-1)), d))
            covered.add(d)
            dates.append(d)
        d += datetime.timedelta(days=7)
    dates = sorted(dates)
    pairs = sorted(pairs)
    for pair in pairs:
        for date in pair:
            if date not in dates:
                print("Date in pairs but not in dates", date)
    return dates, pairs


def get_schedule_csp(year, rotations):
    problem = Problem()
    dates, pairs = get_cover_days(year)
    problem.addVariables(dates, rotations['Name'])
    for pair in pairs:
        if len(pair) == 3:
            problem.addConstraint(lambda name1, name2, name3: name1 != name2
                                                              and name1 != name3
                                                              and name2 != name3, pair)
        elif len(pair) == 2:
            problem.addConstraint(lambda name1, name2: name1 != name2, pair)
    problem.addConstraint(Constraint)
    solution = problem.getSolution()
    return solution


def solution_to_csv(solution, rotations):
    names = rotations["Name"]
    dates = sorted(list(solution.keys()))
    date_strings = [now.strftime("%m/%d") for now in dates]
    df = pd.DataFrame(columns=date_strings + ["Name"])
    df.set_index("Name", drop=True)

    for date in dates:
        assignments = {}
        for name in names:
            assignments[name] = "On Call" if name in solution[date] else ""
        df[date.strftime("%m/%d")] = assignments
    df.to_csv("call_schedule.csv")


def get_viable_names_for_date(date, rotations):
    rotation_assignments = rotations[date.strftime("%B")]
    names = rotations.Name
    values = []
    for i in range(len(rotation_assignments)):
        if rotation_assignments[i] != "NF":
            values.append(i)
    return values


def get_schedule_genetic(year, rotations, mutation_percent_genes=5, num_generations=2000, sol_per_pop=100):
    global solution_variable_to_name
    global solution_index_to_date
    dates, pairs = get_cover_days(year)
    num_parents_mating = 10

    gene_space = []
    count = 0
    random_start = np.zeros((sol_per_pop, 2 * len(dates) + len(pairs)))
    for pair in pairs:
        for date in pair:
            possible_names = get_viable_names_for_date(date, rotations)
            random_start[:, count] = np.random.choice(possible_names, sol_per_pop, replace=True)
            solution_index_to_date[count] = date
            gene_space.append(possible_names)
            count += 1
            random_start[:, count] = -2
            solution_index_to_date[count] = date
            gene_space.append(possible_names + [-2])
            count += 1
        solution_index_to_date[count] = None
        gene_space.append([-1])
        random_start[:, count] = -1
        count += 1
    solution_variable_to_name = {i: rotations["Name"][i] for i in range(len(rotations["Name"]))}

    parent_selection_type = "sss"
    keep_parents = 1

    crossover_type = "single_point"

    mutation_type = "random"

    ga_instance = pygad.GA(num_generations=num_generations,
                           num_parents_mating=num_parents_mating,
                           fitness_func=fitness_function,
                           sol_per_pop=sol_per_pop,
                           num_genes=len(gene_space),
                           parent_selection_type=parent_selection_type,
                           keep_parents=keep_parents,
                           initial_population=random_start,
                           crossover_type=crossover_type,
                           mutation_type=mutation_type,
                           mutation_percent_genes=mutation_percent_genes,
                           parallel_processing=None,
                           gene_space=gene_space)
    ga_instance.run()
    solution, solution_fitness, solution_idx = ga_instance.best_solution()
    # print("Parameters of the best solution : {solution}".format(solution=solution))
    print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))

    decoded_solution = {}
    solution_index = 0
    dates_index = 0
    while solution_index < len(solution):
        if solution[solution_index] == -1:
            solution_index += 1
        else:
            primary = rotations["Name"][solution[solution_index]]
            secondary = rotations["Name"][solution[solution_index + 1]] if solution[solution_index + 1] != -2 else ""
            decoded_solution[dates[dates_index]] = [primary, secondary]
            solution_index += 2
            dates_index += 1
    ga_instance.plot_fitness()
    ga_instance.summary()
    return decoded_solution, solution_fitness


def fitness_function(ga_instance, solution, solution_idx):
    # print(solution, solution_idx)
    i = 0
    fitness = 0.0
    unfilled_backup_spots = 0
    filled_backup_spots = 0
    num_invalid = 0

    # No person will be on call twice or more during the same weekend
    while i < len(solution):
        if solution[i] != -1:
            start_index = i
            end_index = start_index + 1
            while solution[end_index] != -1:
                end_index += 1
            backup = False
            seen_names = set()
            for j in range(start_index, end_index):
                if backup:
                    if solution[j] == -2:
                        unfilled_backup_spots += 1
                    elif solution[j] in seen_names:
                        num_invalid += 1
                    else:
                        filled_backup_spots += 1
                        seen_names.add(solution[j])
                        date = solution_index_to_date[j]
                        name = solution_variable_to_name[solution[j]]
                        rotation = rotations[rotations.Name == name][date.strftime("%B")].values[0]
                        if rotation == "NF":
                            print("Found resident on NF Rotation in the backup spot")
                            num_invalid += 1
                        elif rotation == "ED":
                            fitness += 0.25
                        elif rotation == "CS":
                            fitness += 0.25
                        else:
                            fitness += 1
                else:
                    if solution[j] in seen_names:
                        num_invalid += 1
                    seen_names.add(solution[j])
                    date = solution_index_to_date[j]
                    name = solution_variable_to_name[solution[j]]
                    rotation = rotations[rotations.Name == name][date.strftime("%B")].values[0]
                    if rotation == "NF":
                        print("Found resident on NF Rotation")
                        num_invalid += 1
                    elif rotation == "ED":
                        fitness += 0.5
                    elif rotation == "CS":
                        fitness += 0.5
                    else:
                        fitness += 1
                backup = not backup
            i = end_index + 1
        else:
            i += 1
    val, counts = np.unique(np.array(solution), return_counts=True)
    val = set(val)
    min_value = np.min(counts[1:])
    max_value = np.max(counts[1:])
    for name in range(len(rotations["Name"])):
        if name not in val:
            min_value = 0
    # print(min_value, max_value)
    max_spread = max_value - min_value
    return (fitness / (1.0 + unfilled_backup_spots)) - (max_spread / 2.0) - (1000 * num_invalid)


if __name__ == '__main__':
    # Uncomment below if you want to search the parameter space for a better solution.
    # This will probably have to run overnight depending on how good your computer is
    # best_fitness = -1200000
    # best_solution = None
    # for mutation_percent in np.arange(0, 50, 10):
    #     for sol_per_pop in np.arange(10, 310, 20):
    #         sol, fitness = get_schedule_genetic(2023, rotations, mutation_percent_genes=mutation_percent, sol_per_pop=sol_per_pop)
    #         if best_solution is None:
    #             best_fitness = fitness
    #             best_solution = sol
    # solution_to_csv(best_solution, rotations)
    sol, fitness = get_schedule_genetic(2023, rotations)
    solution_to_csv(sol, rotations)

