## Data Preparation
```*.ratings``` file: rating file with format ```(uid, iid, rating)```

```*.uid``` file: json-like file containing dictionary mapping original user ID to integer index

```*.iid``` file: json-like file containing dictionary mapping original item ID to integer index

```*.vocab``` file: json-like file containing dictionary mapping a word in vocabulary to integer index

```*.user.words.tfidf.npz``` file: tf-idf matrix of user's associated words with size ```(n_user, n_words)```. This file can be obtained by computing directly from user's texts or from user's interacted items' texts, e.g., sum/average.