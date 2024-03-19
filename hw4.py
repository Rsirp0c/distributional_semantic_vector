import random
import numpy as np
import scipy.linalg as scipy_linalg
from collections import defaultdict

random.seed(42)

class co_occurrence_matrix():

    def __init__(self,path):
        self.feature_dict = {}
        self.path = path
        self.matrix = None
        self.ppmi_matrix = None
        self.reduced_ppmi_matrix = None
        self.composes_matrix = None
        self.word2vec_matrix = None 

    def load_word_embeddings(self, paths: list):
        """
        Load the word embeddings from the file.
        return: None
        """
        for path in paths:
            with open(path) as f:
                for line in f:
                    words = line.split()
                    word = words[0]
                    if word not in self.feature_dict:
                        self.feature_dict[word] = len(self.feature_dict)

    def create_co_occurrence_matrix(self):
        """
        Creates a co occurrence matrix from a corpus.
        return: None
        """
        unique_words = set()   
        with open(self.path) as f:
            for line in f:
                words = line.split()
                for word in words:
                    if word not in unique_words:
                        self.feature_dict[word] = len(unique_words)
                    unique_words.add(word)
        self.matrix = np.zeros((len(unique_words),len(unique_words)))
        # print(self.feature_dict)

    def fill_co_occurrence_matrix(self):
        """
        fill in the co occurrence matrix from the corpus
        only assume adjacent tokens as co-occurring
        return: Null
        """
        with open(self.path) as f:
            for line in f:
                words = line.split()
                for i, word in enumerate(words):
                    word_index = self.feature_dict[word]
                    prev_word_index = self.feature_dict[words[i-1]] if i > 0 else None
                    next_word_index = self.feature_dict[words[i+1]] if i < len(words)-1 else None

                    if i == 0:
                        self.matrix[word_index][next_word_index] += 1
                    elif i == len(words)-1:
                        self.matrix[word_index][prev_word_index] += 1
                    else:
                        self.matrix[word_index][prev_word_index] += 1
                        self.matrix[word_index][next_word_index] += 1
        # print(self.matrix)

    def smooth_and_create_ppmi_matrix(self):
        """
        smooth the co-occurrence matrix by multiply 10 and adding 1 to all elements
        Create the PPMI matrix from the co-occurrence matrix.
        return: None
        """
        self.matrix = self.matrix*10 + 1

        self.ppmi_matrix = np.zeros((self.matrix.shape[0],self.matrix.shape[1]))
        for i in range(self.matrix.shape[0]):
            for j in range(self.matrix.shape[1]):
                ppmi = np.log((self.matrix[i][j]*self.matrix.sum())/(self.matrix[i].sum()*self.matrix[:,j].sum()))
                self.ppmi_matrix[i][j] = max(ppmi, 0)

        self.ppmi_matrix = np.around(self.ppmi_matrix, 5)

    def compute_distances(self, matrix, pairs):
        """
        Compute the Euclidean distance between the word pairs.
        return: None
        """
        name = 'PPMI' if matrix is self.ppmi_matrix else 'Reduced PPMI'
        print(f'\nEuclidean distance using {name} between word pairs:')
        for pair in pairs:
            word1_index = self.feature_dict[pair[0]]
            word2_index = self.feature_dict[pair[1]]
            word1_vector = matrix[word1_index]
            word2_vector = matrix[word2_index]
           
            dist = scipy_linalg.norm(word1_vector - word2_vector)
            print(f"The Euclidean distance between {pair[0]} and {pair[1]} is {dist:.2f}")

    def perform_svd_and_verify(self):
        """
        Perform SVD on the PPMI matrix and verify the reconstruction.
        return: None
        """
        U, E, Vt = scipy_linalg.svd(self.ppmi_matrix, full_matrices=False)
        E = np.diag(E)  # Convert the singular values into a diagonal matrix
        reconstructed_matrix = U.dot(E).dot(Vt)
        print('\nVerified that you can recover the orginal matrix: ', np.allclose(self.ppmi_matrix, reconstructed_matrix))

    def create_reduced_ppmi_matrix(self):
        """
        Create a reduced PPMI matrix by reducing the dimensions to 3.
        return: None
        """
        U, E, Vt = scipy_linalg.svd(self.ppmi_matrix, full_matrices=False)
        E = np.diag(E)
        V = Vt.T # compute V = conjugate transpose of Vt
        self.reduced_ppmi_matrix = self.ppmi_matrix.dot(V[:, 0:3])


def part1():
    """
    Runs part one of Homework 4.

    Creates the co-occurrence matrix.
    Prints the co-occurrence matrix.

    Creates the ppmi matrix.
    Prints the ppmi matrix.

    Evaluate word similarity with different distance metrics for each word in the word pairs.
    Reduce dimensions with SVD and check the distance metrics on the word pairs again.
    """
    pairs = [
        ("women", "men"),
        ("women", "dogs"),
        ("men", "dogs"),
        ("feed", "like"),
        ("feed", "bite"),
        ("like", "bite"),
    ]
    nlp = co_occurrence_matrix('dataset/dist_sim_data.txt')
    nlp.create_co_occurrence_matrix()
    nlp.fill_co_occurrence_matrix()
    nlp.smooth_and_create_ppmi_matrix()
    print('Raw count matrix (after smoothing): \n', nlp.matrix)
    print('\nPPMI matrix: \n', nlp.ppmi_matrix)
    nlp.compute_distances(nlp.ppmi_matrix, pairs)
    nlp.perform_svd_and_verify()
    nlp.create_reduced_ppmi_matrix()
    print('\nReduced PPMI matrix: \n', nlp.reduced_ppmi_matrix)
    nlp.compute_distances(nlp.reduced_ppmi_matrix, pairs)
    
class compute_distributional_vector():

    def __init__(self) -> None:
        self.feature_dict = {}
        self.composes_matrix = None
        self.word2vec_matrix = None 
    
    def load_word_embeddings(self, paths: list):
        """
        Load the word embeddings from the file.
        return: None
        """
        for path in paths:
            with open(path) as f:
                for line in f:
                    words = line.split()
                    word = words[0]
                    if word not in self.feature_dict:
                        self.feature_dict[word] = len(self.feature_dict)
            
        self.composes_matrix = np.zeros((len(self.feature_dict),500))
        self.word2vec_matrix = np.zeros((len(self.feature_dict),300))

        for path in paths:
            with open(path) as f:
                for line in f:
                    words = line.split()
                    word = words[0]
                
                    word_index = self.feature_dict[word]
                    if path == 'dataset/EN-wform.w.2.ppmi.svd.500-filtered.txt':
                        self.composes_matrix[word_index] = words[1:]
                    else:
                        self.word2vec_matrix[word_index] = words[1:]
        
    def make_synonym_test(self):
        """
        Create a synonym test dataset.
        return: None
        """
        file = open('dataset/synonyms_test.txt', 'a')

        word_set = set()
        hash_map = defaultdict(set)  
        with open('dataset/EN_syn_verb.txt') as f:  
            for line in f:
                if line[0:2] != 'to':
                    continue
                words = line.split()
                word = words[0].split('_')[1]
                synonym = words[1].split('_',1)[1] if len(words[1].split('_')) > 1 else words[1]
                hash_map[word].add(synonym)
                word_set.add(word)
                word_set.add(synonym)

        word_set = list(word_set)
        for word in hash_map:
            for synonym in hash_map[word]:
                for _ in range(2):
                    word_list = [word, synonym]
                    while len(word_list) < 6:
                        rand_word = random.choice(word_set)
                        if rand_word not in word_list:
                            word_list.append(rand_word)
                        random.shuffle(word_set)
                    file.write(' '.join(word_list) + '\n')


    def compute_distances(self, file):
        """
        Compute the Euclidean distance and cosine similarity between the word pairs.
        return: None
        """
        euc_composes_accuracy, euc_word2vec_accuracy = [0,0], [0,0]
        cos_conposes_accuracy, cos_word2vec_accuracy = [0,0], [0,0]

        for line in file:
            words = line.split()
            pairs =[(words[0], words[1]),(words[0], words[2]),(words[0], words[3]),(words[0], words[4]),(words[0], words[5])]
            
            euc_comp_list, euc_w2v_list, cos_comp_list, cos_w2v_list = [], [], [], []

            for i, pair in enumerate(pairs):
                word1_index = self.feature_dict[pair[0]] if pair[0] in self.feature_dict else None
                if word1_index:
                    comp_word1_vector = self.composes_matrix[word1_index] 
                    w2v_word1_vector = self.word2vec_matrix[word1_index] 
                else:
                    comp_word1_vector = np.zeros(500)
                    w2v_word1_vector = np.zeros(300)

                # check if the second word is a phrase
                if "_" in pair[1]:
                    phrase_words = pair[1].split("_")
                    # Only include words that are in the feature_dict
                    phrase_word_indexes = [self.feature_dict[word] for word in phrase_words if word in self.feature_dict]
                    if phrase_word_indexes:
                        # avg approach
                        # comp_word2_vector = np.sum([nlp.composes_matrix[index] for index in phrase_word_indexes], axis=0)/len(phrase_word_indexes) 
                        # w2v_word2_vector = np.sum([nlp.word2vec_matrix[index] for index in phrase_word_indexes], axis=0)/len(phrase_word_indexes) 
                        # sum approach
                        comp_word2_vector = np.sum([self.composes_matrix[index] for index in phrase_word_indexes], axis=0)
                        w2v_word2_vector = np.sum([self.word2vec_matrix[index] for index in phrase_word_indexes], axis=0) 
                    else:
                        comp_word2_vector = np.zeros(500)
                        w2v_word2_vector = np.zeros(300)
                else:
                    word2_index = self.feature_dict[pair[1]] if pair[1] in self.feature_dict else None
                    if word2_index:
                        comp_word2_vector = self.composes_matrix[word2_index] 
                        w2v_word2_vector = self.word2vec_matrix[word2_index] 
                    else:
                        comp_word2_vector = np.zeros(500)
                        w2v_word2_vector = np.zeros(300)
                
                # Compute the euclidean distance between the two words
                comp_dist = scipy_linalg.norm(comp_word1_vector - comp_word2_vector)
                w2v_dist = scipy_linalg.norm(w2v_word1_vector - w2v_word2_vector)
                
                # Compute the cosine similarity between the two words
                if np.any(comp_word1_vector) and np.any(comp_word2_vector):
                    comp_similarity = np.dot(comp_word1_vector, comp_word2_vector) / (scipy_linalg.norm(comp_word1_vector) * scipy_linalg.norm(comp_word2_vector))
                else:
                    comp_similarity = 0

                if np.any(w2v_word1_vector) and np.any(w2v_word2_vector):
                    w2v_similarity = np.dot(w2v_word1_vector, w2v_word2_vector) / (scipy_linalg.norm(w2v_word1_vector) * scipy_linalg.norm(w2v_word2_vector))
                else:
                    w2v_similarity = 0
                    
                euc_comp_list.append((i, comp_dist))
                euc_w2v_list.append((i, w2v_dist))
                cos_comp_list.append((i, comp_similarity))
                cos_w2v_list.append((i, w2v_similarity))

            min_index_comp, _ = min(euc_comp_list, key=lambda x: x[1]) if euc_comp_list else (-1, _)
            min_index_w2v, _ = min(euc_w2v_list, key=lambda x: x[1]) if euc_w2v_list else (-1, _)
            max_index_comp, _ = max(cos_comp_list, key=lambda x: x[1]) if cos_comp_list else (-1, _)
            max_index_w2v, _ = max(cos_w2v_list, key=lambda x: x[1]) if cos_w2v_list else (-1, _)  

            if min_index_comp == 0:
                euc_composes_accuracy[0] += 1
            if min_index_w2v == 0:
                euc_word2vec_accuracy[0] += 1

            if max_index_comp == 0:
                cos_conposes_accuracy[0] += 1
            if max_index_w2v == 0:
                cos_word2vec_accuracy[0] += 1

            euc_composes_accuracy[1] += 1
            euc_word2vec_accuracy[1] += 1
            cos_conposes_accuracy[1] += 1
            cos_word2vec_accuracy[1] += 1

        print('\nEuclidean distance:')
        print(f'Composes accuracy: {euc_composes_accuracy[0]}/{euc_composes_accuracy[1]}')
        print(f'Word2Vec accuracy: {euc_word2vec_accuracy[0]}/{euc_word2vec_accuracy[1]}')
        print('Cosine similarity:')
        print(f'Composes accuracy: {cos_conposes_accuracy[0]}/{cos_conposes_accuracy[1]}')
        print(f'Word2Vec accuracy: {cos_word2vec_accuracy[0]}/{cos_word2vec_accuracy[1]}')


    def run_synonym_test(self):
        """
        Sets up the synonym test, loads the word embeddings and runs the evaluation.
        Prints the overall accuracy of the synonym task.
        """
        try:
            file = open('dataset/synonyms_test.txt', 'r')
            
        except FileNotFoundError:
            self.make_synonym_test()
            file = open('dataset/synonyms_test.txt', 'r')
            
        self.compute_distances(file)
        
        file.close()
    
    def solve_analogy(self, file):
        """
        Solve the analogy task.
        return: None
        """
        dig2char = {1: 'a', 2: 'b', 3: 'c', 4: 'd', 5: 'e'}
        question = None
        choices = []
        answer = None
        for i, line in enumerate(file):
            if i == 0:
                question = line.split()[0:2]
            elif i == 6:
                answer = line.split()[0]
            else:
                ans = line.split()[0:2]
                choices.append([dig2char[i]]+ans)

        answer_index1 = self.feature_dict[question[0]] if question[0] in self.feature_dict else None
        answer_index2 = self.feature_dict[question[1]] if question[1] in self.feature_dict else None
        answer_vector1 = self.word2vec_matrix[answer_index1] if answer_index1 else np.zeros(300)
        answer_vector2 = self.word2vec_matrix[answer_index2] if answer_index2 else np.zeros(300)
            
        for choice in choices:
            choice_index1 = self.feature_dict[choice[1]] if choice[1] in self.feature_dict else None
            choice_index2 = self.feature_dict[choice[2]] if choice[2] in self.feature_dict else None
            choice_vector1 = self.word2vec_matrix[choice_index1] if choice_index1 else np.zeros(300)
            choice_vector2 = self.word2vec_matrix[choice_index2] if choice_index2 else np.zeros(300)
            
            exp = answer_vector2 - answer_vector1 + choice_vector1
            if np.any(exp) and np.any(choice_vector2):
                similarity = np.dot(exp, choice_vector2) / (scipy_linalg.norm(exp) * scipy_linalg.norm(choice_vector2))
            else:
                similarity = 0
            choice.append(similarity)

        my_choice, _, _, _ = max(choices, key=lambda x: x[3]) 

        if my_choice == answer:
            return True
        
    def run_sat_test(self):
        """
        Sets up the SAT test, loads the word embeddings and runs the evaluation.
        Prints the overall accuracy of the SAT task.
        """
        file = open('dataset/SAT-package-V3.txt', 'r')

        accuracy = [0,0]
        question_cell = []
        # line_count = 0
        for line in file:
            # if line_count > 500:
            #     break
            if line[0:2] == '19' or line[0:2] == 'ML' or line[0:2] == 'KS' or line[0] == '#' or line[0] == '\n':
                continue
            if len(question_cell) < 7:
                question_cell.append(line)
            else:
                # process this question
                if self.solve_analogy(question_cell):
                    accuracy[0] += 1
                accuracy[1] += 1
                # append the first line of the next question to the question cell
                question_cell = [line]
            # line_count += 1
        
        print('\nSAT accuracy:', accuracy[0], '/', accuracy[1])
        

def part2():
    """
    Runs the two tasks for part two of Homework 4.
    """
    
    nlp = compute_distributional_vector()
    nlp.load_word_embeddings(['dataset/EN-wform.w.2.ppmi.svd.500-filtered.txt', 'dataset/GoogleNews-vectors-negative300-filtered.txt'])
    # nlp.run_synonym_test()
    nlp.run_sat_test()



if __name__ == "__main__":
    # DO NOT MODIFY HERE

    # part1()
    part2()