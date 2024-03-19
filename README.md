# CS115 HW4 write up

# Distributional Semantics Takes the SAT

### 1. Create distributional semantic word vectors

1. **Comparison of the word vector for “dogs” before and after PPMI:**
    
    ```python
    # dog vector in count matrix
    ['the': 91.0, 'men': 1.0, 'feed': 1.0, 'dogs': 1.0, 'women': 1.0, 'bite': 31.0, 'like': 11.0]
    
    # dog vector in PPMI matrix
    ['the': 0.605, 'men': 0.0, 'feed': 0.0, 'dogs': 0.0, 'women': 0.0, 'bite': 1.143, 'like': 0.0]
    ```
    
    **Does PPMI do the right thing to the count matrix?**
    
    I think it does. PPMI aims to reduces the bias of generally common words(”the” is this case), and focus on the significance of the co-occurrence. By utilizing PPMI, the ratio between **the/bite** drop from arround **3:1** to **0.6:1.1**. This definitely emphasize the co-occurence relation between **dogs and bite**, which is intuitively more useful!
    
2. **Word Pairs Calculations**
    
    
    | Word Pairs | Euclidean Distance (PPMI) | Euclidean Distance (Reduced PPMI) |
    | --- | --- | --- |
    | Women and Men (human noun vs. human noun) | 0.47 | 0.13 |
    | Women and Dogs (human noun vs. animal noun) | 1.52 | 1.07 |
    | Men and Dogs (human noun vs. animal noun) | 1.40 | 0.97 |
    | Feed and Like (human verb vs. human verb) | 0.65 | 0.44 |
    | Feed and Bite (human verb vs. animal verb) | 1.59 | 1.13 |
    | Like and Bite (human verb vs. animal verb) | 1.31 | 0.91 |
    
    **Do the distances you compute above confirm our intuition from distributional semantics?**
    
    They do, distance between the same category(human n vs. human n) is **significantly closer** than distance between different categories(human n vs animal n). Similarly, distances of the same comparison between category run into similar ranges; the distance between **women and dogs** is similar to the distance between **men and dogs**, which all calculate over **human noun vs. animal noun**. 
    
    **Does the compact/reduced matrix still keep the information we need for each word vector?**
    
    Yes, it does. The observations I made on distance, in last question, still holds in the column of **reduced PPMI.**
    

### 2. Computing with distributional semantic word vectors

1. **Synonym test results**  (Euclidean distance vs. cosine similarity & COMPOSES vs. word2vec)
    
    I also came across different method to calculate the **vector of a phrase/sentence** and post it on the table.
    
    | Method | Metrics | Composes Accuracy | Word2Vec Accuracy |
    | --- | --- | --- | --- |
    | Sum Approach | Euclidean Distance | 50.03% (603/1198) | 56.51% (677/1198) |
    | Sum Approach | Cosine Similarity | 54.92% (658/1198) | 67.86% (813/1198) |
    | Avg. Approach | Euclidean Distance | 50.17% (601/1198) | 53.01% (635/1198) |
    | Avg. Approach | Cosine Similarity | 54.92% (658/1198) | 67.86% (813/1198) |
2. **Analogy task discussion and results**
    
    
    I use the google pre-train word2vec model to vectorize the analogy task. 
    
    I felt the SAT analogy task here is similar to asking    "a is to b as c is to what???" Therefore I come up with a function `expectation = b - a + c` and calculate the cosine similarity between `expectation` and `d`. 
    
    Saving all the cosine similarities between different pairs of words, the output will be the one having **highest cosine similarity**. 
    
    **My accuracy is 31.37% (117 / 373)**
    
    ```python
    target = [a,b]
    pairs = [[c,d],[e,f] ... ]
    
    for pair in pairs:
    		exp = b - a + pair[0]
    		cos_similarity = func(exp,pair[1])
    		pair.append(cos_similarity)
    		
    AI_choice = max(pairs, key=lambda x: x[2])
    ```