import itertools
import numpy as np
from collections import defaultdict
#from concurrent.futures import ThreadPoolExecutor

P_max = 3
N_max=6
M_max=20


def a_i(p):
    """
    Generates all combinations of p numbers out of the set {1, 2, 3},
    where each number can appear more than once.
    
    Parameters:
    p (int): The number of elements in each combination.
    
    Returns:
    list of tuples: A list containing all possible combinations.
    """
    elements = [1, 2, 3]
    return list(itertools.product(elements, repeat=p))

def i_i(n, p):
    """
    Generates all unique combinations of p numbers between 0 and (n-1),
    where each combination is sorted in ascending order.
    
    Parameters:
    n (int): The upper limit (exclusive) for the numbers.
    p (int): The number of elements in each combination.
    
    Returns:
    list of tuples: A list containing all unique combinations, sorted in ascending order.
    """
    return list(itertools.combinations(range(n), p))

def pairs_ia(list1, list2):
    """
    Generate pairs of elements from list1 and list2.
    
    Parameters:
    list1 (list): First input list.
    list2 (list): Second input list.
    
    Returns:
    list: List of pairs where each pair is a list containing one element from list1 and one from list2.
    """

    return [[x, y] for x in list1 for y in list2]

def generate_combinations_with_replacement(lists,m):
    """
    Generate all combinations of m elements from the input lists with replacement (elements can be repeated).

    Parameters:
    lists (list of list): Input list of elements to form combinations with
    m (int): Number of elements in each combination.

    Returns:
    list of tuples: List of all possible combinations.
    """
    return list(itertools.product(lists, repeat=m))

def multiply_matrices_in_lists(list_of_lists):
    """
    Multiply all matrices in each sublist and return a list of resulting matrices.
    
    Parameters:
    list_of_lists (list of list of np.ndarray): Input list of lists of matrices.
        Each sublist contains matrices to be multiplied together.
    
    Returns:
    list of np.ndarray: List of resulting matrices, where each resulting matrix is the product
        of the matrices in the corresponding sublist from the input.
    """
    result_list = []
    
    for sublist in list_of_lists:
        if not sublist:
            continue
        # Initialize the result as the identity matrix of appropriate size
        result_matrix = np.eye(sublist[0].shape[0])
        
        for matrix in sublist:
            result_matrix = np.dot(result_matrix, matrix)
        
        result_list.append(result_matrix)
    
    return result_list

def ia_to_Pauli(list1, list2, n):
    """
    Converts two lists to a Pauli operator by first creating pairings using pairs_ia function.
    
    Parameters:
    list1 (list): The first list to create pairings.
    list2 (list): The second list to create pairings.
    n (int): The length of the resulting lists.
    
    Returns:
    list of str: A list of Pauli matrices.
    """
    def pauli_tensorprodukt(indices):
        """
        Converts a list of indices to a Pauli operator by taking the tensor prduct of the matrices with those indices
        
        Parameters:
        indices: A list of indices for the Pauli matrices.
        
        Returns:
        np.ndarray: The resulting Pauli operator.
        """
        # Definiere die Pauli-Matrizen
        pauli_matrices = {
            0: np.array([[1, 0], [0, 1]]),
            1: np.array([[0, 1], [1, 0]]),
            2: np.array([[0, -1j], [1j, 0]]),
            3: np.array([[1, 0], [0, -1]])
        }
        
        # Initialisiere das Zwischenergebnis mit der ersten Matrix
        if not indices:
            return 'Error: List is empty'
        
        if indices[0] not in pauli_matrices:
            return 'Error: Invalid index in list'
        
        zwischenergebnis = pauli_matrices[indices[0]]
        
        # Berechne das Tensorprodukt für die restlichen Matrizen
        for index in indices[1:]:
            if index not in pauli_matrices:
                return 'Error: Invalid index in list'
            matrix = pauli_matrices[index]
            zwischenergebnis = np.kron(zwischenergebnis, matrix)
        
        return zwischenergebnis


    pairings = pairs_ia(list1, list2)
    result = []
    for pauli, indices in pairings:
        temp = [0] * n
        for idx, val in zip(indices, pauli):
            temp[idx] = val
        result.append(pauli_tensorprodukt(temp))
    return result

def Rademacher(list_of_lists):
    """
    Checks if the expectation of the product of random variables distributed according to the Rademacher distribution is zero or one.

    Parameters:
    list_of_lists (list of list): List of pairs i_i (indices in the list) and a_i (pauli indices).

    Returns:
    int: 1 if the expectation is one, 0 otherwise.
    """
    def count_occurrences(lst):
        """Helper function to count occurrences of each list of lists."""
        
        def make_hashable(lst):
            """Helper function to convert nested lists into a hashable type (tuple)."""
            if isinstance(lst, list):
                return tuple(make_hashable(item) for item in lst)
            return lst

        count = defaultdict(int)
        for item in lst:
            hashable_item = make_hashable(item)
            count[hashable_item] += 1
        return count
    
    # Step 1: Count occurrences of each list of lists
    count = count_occurrences(list_of_lists)

    # Step 2: Check if all counts are even
    for c in count.values():
        if c % 2 != 0:
            return 0

    # Step 3: Return 1 if all counts are even
    return 1

def check_commutation(matrices):
    """
    Check if all matrices in the list mutually commute.
    
    Parameters:
    matrices (list of np.ndarray): List of square matrices (NumPy arrays).
    
    Returns:
    0 if all matrices mutually commute, 1 otherwise.
    """
    n = len(matrices)
    for i in range(n):
        for j in range(i + 1, n):
            commutator = np.dot(matrices[i], matrices[j]) - np.dot(matrices[j], matrices[i])
            if not np.allclose(commutator, np.zeros_like(commutator)):
                return 1
    return 0

def sequence_element(m, n, p):
    """
    Compute the sequence element (equation 90) for given values of m, n, and p taking only the anticommuting matrices into account.

    Parameters: 
    m (int): The number of matrices that are multiplied together.
    n (int): The number of qubits in the system. (dimension of the Hilbert space is 2^n)
    p (int): he number of interacting qubits.

    Returns:
    list: A list containing the non-zero results and the corresponding pairs of indices i_i and a_i.
    """
    if m % 2 != 0:
        return 0
    i_i_list = i_i(n, p) # length maximal n over p
    a_i_list = a_i(p) # length 3^p
    pairs_list = pairs_ia(i_i_list, a_i_list) 
    combinations_pairs_list = generate_combinations_with_replacement(pairs_list, m)
    Rademacher_list = [Rademacher(pair) for pair in combinations_pairs_list]
    pauli_list = ia_to_Pauli(a_i_list, i_i_list, n) #this takes up a lot of space
    combinations_list = generate_combinations_with_replacement(pauli_list, m) #big 
    product_list = multiply_matrices_in_lists(combinations_list)
    len_Rademacher = len(Rademacher_list)
    result_list = [
        np.trace(product_list[i]) * Rademacher_list[i] * check_commutation(combinations_list[i])
        for i in range(len_Rademacher)
    ]
    final_list = [
        [result_list[i], combinations_pairs_list[i]]
        for i in range(len_Rademacher) if result_list[i] != 0
    ]
    sum = 0
    for i in range(len(result_list)):
        sum += result_list[i]
    if sum != 0:
        return final_list
    return None

def process_parameters(M, P, N):
    result = sequence_element(M, N, P)
    if result != None:
        print(f"M={M}, N={N}, P={P}: {result}")
        return True

# check if there are non-zero elements for all combinations of M, P, and N
if __name__ == "__main__":
    #for M in range(4,M_max, 2):
    #    for P in range(2, P_max):
    #        for N in range(P, N_max):
    #            print(f'running N={N},M={M}, P={P}')
    process_parameters(4, 2, 2) #M, P, N
