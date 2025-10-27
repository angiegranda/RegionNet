import numpy as np
import random
from itertools import tee
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from typing import Optional, Iterable
import gc

def similarity_to_rating(similarity: float, min_rating: float, max_rating: float) -> float:
    normalized = (similarity + 1) / 2
    return normalized * (max_rating - min_rating) + min_rating


def bounded_gauss_noise(mu, sigma, low, high):
    while True:
        x = random.gauss(mu, sigma)
        if low <= x <= high:
            return x

"""
Hybrid Recommender Class:

- Uses content based recommender to prevent the so called cold-start user problem 
- Funk MF for discovering hidden latten features. It is slightly modified by adding predicted ratings using a 
user-based NN model and considering each possible new rating only on 30% of cases.
- In order to optimize the most the computations of the 
content-based model the Author class contains a accumulated feature vector and ratings that allows to calculate the author 
profile in O(1) and an iterarator of authors 
"""

class HybridRecommender:

    def __init__(self, similarity_function: bool, alpha=0.5, min_rating=1.0, max_rating=5.0, num_factors=15, lr=0.01, reg=0.01, epochs=15):
        """ Hyperparameters obtained from Program->Application
            alpha represents the factor of the content based result in the final prediction, 1-alpha represents the factor of Funk MF
            similarity_function if True then Pearson Correlation is taken, else Cosine Similarity
            num_factors is the hidden latent factors chosen for Funk MF
            Matrices A, H arrays ba, bh, and the global mean are necessary for Funk MF computations
        """
        self.alpha = alpha
        self.min_rating = min_rating
        self.max_rating = max_rating

        self.similarity_function = similarity_function # If False then cosine similarity, else pearson correlation

        self.authors_id_to_index = {}
        self.hotel_id_to_index = {}
        self.hotel_index_to_id = {}
        self.all_authors_id = set()
        self.all_hotels_id = set()

        # Funk MF
        self.num_factors = num_factors
        self.lr = lr # learning rate 
        self.reg = reg # regularization parameter
        self.epochs = epochs
        
        self.A = None # latent features for authors 
        self.H = None # latent features for hotels 
        self.ba = None # bias for authors 
        self.bh = None # bias for hotels 
        self.global_mean = 0.0

        # Content Based
        self.similarity_matrix = None




    def _build_author_profile(self, author, hotel_vectors, low_noise=0, high_noise=0.05, mu=0.1, sigma=0.03):
        """
        The author profile is computed as a weighted sum of hotel vectors, where the weights
        are the mean-centered ratings with added small Gaussian noise. Adding noise helps
        avoid situations where an author who gave identical ratings to all hotels would 
        result in a zero vector after mean-centering. 
        """
        profile_len = hotel_vectors.shape[1]
        author_profile = np.zeros(profile_len, dtype=np.float32)
        noisy_ratings = []
        hotel_indices = []

        for hotel_id, rating in author.training_ratings:
            noisy_rating = rating + bounded_gauss_noise(mu=mu, sigma=sigma, low=low_noise, high=high_noise)
            noisy_ratings.append(noisy_rating)
            hotel_idx = self.hotel_id_to_index[hotel_id]
            hotel_indices.append(hotel_idx)

        if noisy_ratings:
            mean_rating = np.mean(noisy_ratings)
            mean_centered = np.array(noisy_ratings) - mean_rating
            for idx, h_id in enumerate(hotel_indices):
                author_profile += hotel_vectors[h_id] * mean_centered[idx]
            norm = np.linalg.norm(author_profile)
            if norm > 0:
                author_profile /= norm #unit vector 

        return author_profile


    def __calculate_similarity_matrix(self, authors_iter: Iterable, hotels: list, n_authors: int, n_hotels: int):
        """
        If self.similarity_function is True when we use Pearson Correlation as the similarity metric. Iterate over 
        all authors and use the function _build_author_profile to create the author profile, computer the similarity matric
        and fill the similarity_matrix or also called utility matrix
        """
        hotel_vectors = np.array([hotel.feature_vector for hotel in hotels], dtype=np.float32)
        similarity_matrix = np.empty((n_authors, n_hotels), dtype=np.float32)

        if self.similarity_function: 
            hotel_means = hotel_vectors.mean(axis=1, keepdims=True)
            hotel_centered = hotel_vectors - hotel_means

        for author in authors_iter:
            author_index = self.authors_id_to_index[author.author_id]
            author_profile = self._build_author_profile(author, hotel_vectors)

            if self.similarity_function:
                author_centered = author_profile - author_profile.mean()
                # pearson correlation: (centered author ⋅ centered hotel) / (||centered author|| * || centered hotel||)
                numerator = hotel_centered @ author_centered
                denominator = np.linalg.norm(hotel_centered, axis=1) * np.linalg.norm(author_centered)
                results = numerator / (denominator + 1e-8)  # avoid division by zero
            else:
                # cosine similarity
                results = cosine_similarity(author_profile.reshape(1, -1), hotel_vectors).flatten()

            similarity_matrix[author_index, :] = results

        return similarity_matrix


    def _generate_neighbor_pseudo_ratings(self, authors_iter: Iterable, hotels: list, k_neighbors=5, prob_acceptance=0.3, similarity_boundary=0.7):
        """
        Hotel feature vector are created with PCA and by lowering the dimention, both the hotel features and authors profiles contain hidden 
        factors. User-Based NN is implemented by adding new rating based on a 70% of similarity of neighbors with the current author. Users close
        in distance might not necessarily have the same preferences hence this brings diversity. With 30% of probability this new ratings is considered.
        """
        hotel_vectors = np.array([hotel.feature_vector for hotel in hotels], dtype=np.float32)
        authors_dict = {author.author_id: author for author in authors_iter}
        rating_dicts = {author.author_id: dict(author.training_ratings) for author in authors_dict.values()} # author_id : {hotel_id: rating, ...}

        user_profiles = np.zeros((self.n_authors, hotel_vectors.shape[1]), dtype=np.float32)
        for author_id, author in authors_dict.items():
            idx = self.authors_id_to_index[author_id]
            user_profiles[idx] = self._build_author_profile(author, hotel_vectors)

        nn = NearestNeighbors(n_neighbors=k_neighbors, metric='cosine')
        nn.fit(user_profiles)
        distances, indices = nn.kneighbors(user_profiles) 
        # distances = cosine distances to neighbors for each author and indices = indices of nearest neighbors for each author

        pseudo_ratings = []

        for author_index in range(self.n_authors):
            neighbor_indices = indices[author_index]
            neighbor_similarities = 1 - distances[author_index] # cosine distance 1 means perpendicular, oposite
            author_id = self.author_index_to_id[author_index]
            rated_hotels = set(rating_dicts[author_id].keys())

            for hotel_index in range(self.n_hotels):
                hotel_id = self.hotel_index_to_id[hotel_index]
                if hotel_id in rated_hotels:
                    continue
                if np.random.rand() >= prob_acceptance:
                    continue

                neighbor_ratings = []
                weights = []
                for neighbor_idx, similarity in zip(neighbor_indices, neighbor_similarities):
                    if similarity > similarity_boundary:
                        neighbor_author_id = self.author_index_to_id[neighbor_idx]
                        rating = rating_dicts[neighbor_author_id].get(hotel_id)
                        if rating is not None:
                            neighbor_ratings.append(rating)
                            weights.append(similarity)
                if neighbor_ratings:
                    pseudo = np.average(neighbor_ratings, weights=weights)
                    pseudo_ratings.append((author_id, hotel_id, pseudo))

        return pseudo_ratings

    def train(self,
              authors_iter: Iterable,
              authors_id_to_index: dict,
              hotels: list,
              hotel_id_to_index: dict,
              ratings_info: list[tuple], 
              global_mean: float, 
              ):
        """
        This method initializes author and hotel embeddings, computes the content-based similarity 
        matrix, and performs stochastic gradient descent to learn latent factors and biases 
        from the provided ratings. It also augments training data with pseudo-ratings from neighboring 
        items to improve generalization.
        """
        self.authors_id_to_index = authors_id_to_index
        self.hotel_id_to_index = hotel_id_to_index
        self.hotel_index_to_id = {idx: hid for hid, idx in self.hotel_id_to_index.items()}
        self.author_index_to_id = {idx: aid for aid, idx in self.authors_id_to_index.items()}
        self.all_authors_id = set(authors_id_to_index.keys())
        self.all_hotels_id = set(hotel_id_to_index.keys())
        self.global_mean = global_mean

        self.n_authors = len(authors_id_to_index)
        self.n_hotels = len(hotel_id_to_index)

        author_iterator_similarity_matrix, author_iterator_item_NN_data_augmentation = tee(authors_iter, 2)

        # Content Based Recommender calculation
        self.similarity_matrix = self.__calculate_similarity_matrix(author_iterator_similarity_matrix, hotels, self.n_authors, self.n_hotels)
        gc.collect() 

        # Funk MF Recommender calculation 
        self.A = np.random.normal(scale=0.1, size=(self.n_authors, self.num_factors)).astype(np.float32)
        self.H = np.random.normal(scale=0.1, size=(self.n_hotels, self.num_factors)).astype(np.float32)
        self.ba = np.zeros(self.n_authors, dtype=np.float32)
        self.bh = np.zeros(self.n_hotels, dtype=np.float32)
        
        ratings_info_augmented = ratings_info + self._generate_neighbor_pseudo_ratings(author_iterator_item_NN_data_augmentation, hotels)
        # random order
        for _ in range(self.epochs): # Train with SGD 
            for author_id, hotel_id, rating in ratings_info_augmented:
                author_index = self.authors_id_to_index[author_id]
                hotel_index = self.hotel_id_to_index[hotel_id]
                prediction = self.global_mean + self.ba[author_index] + self.bh[hotel_index] + np.dot(self.A[author_index], self.H[hotel_index])
                err = rating - prediction
                self.ba[author_index] += self.lr * (err - self.reg * self.ba[author_index])
                self.bh[hotel_index] += self.lr * (err - self.reg * self.bh[hotel_index])
                self.A[author_index] += self.lr * (err * self.H[hotel_index] - self.reg * self.A[author_index])
                self.H[hotel_index] += self.lr * (err * self.A[author_index] - self.reg * self.H[hotel_index])

    def predict(self, author_id: str, hotel_id: int) -> Optional[float]:
        """If author is not found, it return None. If found, combine the content based model with 
        Funk MF and return the prediction"""
        if author_id not in self.all_authors_id or hotel_id not in self.all_hotels_id:
            return None
        
        author_index = self.authors_id_to_index[author_id]
        hotel_index = self.hotel_id_to_index[hotel_id]

        results = self.similarity_matrix[author_index, hotel_index]
        cb_prediction = similarity_to_rating(results, self.min_rating, self.max_rating)

        mf_prediction = (
                  self.global_mean 
                   + self.ba[author_index] 
                   + self.bh[hotel_index] 
                   + np.dot(self.A[author_index], self.H[hotel_index])
                )

        final_prediction = self.alpha * mf_prediction + (1 - self.alpha) * cb_prediction
        return np.clip(final_prediction, self.min_rating, self.max_rating)



    def predict_selected_hotels_for_author(self, author_id: str, hotel_ids: list[str]):
        """Function needed for stats.py. For a specific user and a list of hotels, it returns a list of the 
        predictions for the. Uses nice slides of numpy"""
        if author_id not in self.authors_id_to_index:
            return []

        author_index = self.authors_id_to_index[author_id]
        hotel_indices = [self.hotel_id_to_index[hotel_id] for hotel_id in hotel_ids if hotel_id in self.all_hotels_id]
        if not hotel_indices:
            return []
        
        cb_predictions = similarity_to_rating(
                                self.similarity_matrix[author_index, hotel_indices], 
                                self.min_rating, 
                                self.max_rating
                                )

        author_latent = self.A[author_index]                  # shape: (latent_dim, 1)
        hotel_latents = self.H[hotel_indices]                 # shape: (k hotel indices, latent_dim)
        dot_products = np.dot(hotel_latents, author_latent)   # shape: (k,1)

        mf_predictions = (
            self.global_mean
            + self.ba[author_index]                           # scalar
            + self.bh[hotel_indices]                          # shape: (k,)
            + dot_products                                     # shape: (k,)
        )

        final_predictions = self.alpha * mf_predictions + (1 - self.alpha) * cb_predictions
        final_predictions = np.clip(final_predictions, self.min_rating, self.max_rating)

        return list(zip(hotel_ids, final_predictions.tolist())) # output = [(hotel_id, prediction), ...]
