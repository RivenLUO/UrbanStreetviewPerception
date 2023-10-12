from itertools import product, permutations


def get_unique_permutations(l, num_to_pick=None):
    """
    Generate all unique permutations of length num_to_pick

    :param l:
    :type l: list
    :param num_to_pick:
    :type num_to_pick: int

    :return:
    :rtype: list
    """

    # Default num_to_pick to length of l
    if num_to_pick is None:
        num_to_pick = len(l)

    # Generate all possible combinations with repetitions
    l_all_combinations = list(product(l, repeat=num_to_pick))

    # Generate permutations of each combination
    permuted_combinations = [list(p) for comb in l_all_combinations
                             for p in permutations(comb)]

    # Remove duplicates by converting to a set and then back to a list
    unique_permuted_combinations = list(set(map(tuple, permuted_combinations)))

    return unique_permuted_combinations
