import os
from json import loads
import random
import numpy as np
from itertools import tee
from recommender import HybridRecommender
from Databases.authorsdb import AuthorsDatabase, Author, create_authors_database_dir
from Databases.hotelsdb import HotelsDatabase, Hotel, create_hotels_database_dir
from Databases.countries_classification import CountriesClassifications, create_classifications_dir
from copy import deepcopy
from stats import Recommender_Statistics, create_output_dir_stats
import gc
import heapq


""" 
Application class:
- Manage insertion, updating, and deletion of authors, hotels, and reviews.
- Precompute PCA-based hotel features for content-based recommendations.
- Maintain precomputed author feature profiles for efficient recommendations.
- Train and use a HybridRecommender to suggest hotels to authors.
- Provide statistics on recommender performance if requested.
"""

class Application:

    def __init__(self):
        create_classifications_dir() 
        create_output_dir_stats()
        create_authors_database_dir()
        create_hotels_database_dir()
        
        self.__authors_database = AuthorsDatabase()
        self.__hotels_database = HotelsDatabase()
        self.is_initialized =  self.__authors_database.contains_rows and self.__hotels_database.contains_rows
    


    ###### --------------------- Private functions ---------------------- #######



    def __process_current_author(self, author : Author,
                            reviews_info: dict, 
                            ratings: list, 
                            rated_feature_matrix: list[np.ndarray], 
                            length_feature_vec: int,
                            hotels_info: list,
                            hotels_id_to_index: dict):
        """
        Update an author's training, test ratings, accumulated feature vector and accumulated_ratings.
        Keep the last review as a test rating if there is at least one previous review.
        Training ratings accumulate feature vectors weighted by ratings.
        This allows the content-based component to quickly access author profiles without recomputation.
        """
        if len(reviews_info) == 0:
            return
        if len(reviews_info) == 1:
            if not author.training_ratings:
                author.training_ratings = deepcopy(reviews_info)
                author.accumulated_feature_vector = (np.array(rated_feature_matrix[0]) * ratings[0]).tolist()
                author.accumulated_ratings = ratings[0]
            elif not author.test_rating:
                author.test_rating = deepcopy(reviews_info)
            else:
                (hotel_id2, rating2) = author.test_rating[0]
                author.test_rating = deepcopy(reviews_info)
                author.training_ratings.append((hotel_id2, rating2))
                author.accumulated_feature_vector = (
                    np.array(author.accumulated_feature_vector or np.zeros(length_feature_vec)) +
                    np.array(hotels_info[hotels_id_to_index[hotel_id2]].feature_vector) * rating2
                ).tolist() 
                author.accumulated_ratings += rating2
        else:
            (hotel_id_test, rating_test) = reviews_info[-1]
            training = reviews_info[:-1]
            features = rated_feature_matrix[:-1]
            train_ratings = ratings[:-1]

            if author.test_rating:
                (old_test_hotel, old_test_rating) = author.test_rating[0]
                training.append((old_test_hotel, old_test_rating))
                features.append(hotels_info[hotels_id_to_index[old_test_hotel]].feature_vector)
                train_ratings.append(old_test_rating)

            author.test_rating = [(hotel_id_test, rating_test)]
            author.training_ratings.extend(training)

            accum = np.sum([np.array(vec) * rat for vec, rat in zip(features, train_ratings)], axis=0)
            author.accumulated_feature_vector = (
                np.array(author.accumulated_feature_vector or np.zeros(length_feature_vec)) + accum
            ).tolist() 
            author.accumulated_ratings += sum(train_ratings)

    def __authors_generator(self, author_ids: list[str]):
        """Yield authors one by one to avoid loading millions of authors into memory at once."""
        for author_id in author_ids:
            author = self.__authors_database.get_author_by_id(author_id)
            yield author

        # Recommender helper functions, mostly to be able to delete quickly the local variables 
    
    def __rating_tuples_generator(self, authors_iter):
        """
        Aggregate all training ratings from authors.
        Returns a list of tuples (author_id, hotel_id, rating) and 
        a global_mean across all authors, later used for Funk MF
        """
        ratings_info = []
        ratings = []
        for author in authors_iter:
            for (hotel_id, rating) in author.training_ratings:
                ratings_info.append((author.author_id, hotel_id, rating))
                ratings.append(rating)

        return ratings_info, np.mean(np.array(ratings)) if ratings else 0.01


    def __prepare_recommender(self, similarity_function: bool, alpha: float, latent_factors: int, lr: float, reg: float, epochs: int): 
        """
        Initialize and train the hybrid recommender.
        1. Load hotel feature vectors.
        2. Generate author iterators.
        3. Compute global mean rating.
        4. Train the HybridRecommender with both content-based and collaborative filtering components.
        """
        hotels_profiles = self.__hotels_database.get_hotels_feature_vectors()# hotels with hotel_id and feature_vector 

        authors_id_to_index = {author_id: index for index, author_id in enumerate(self.author_ids)}
        hotel_id_to_index = {hotel.hotel_id: index for index, hotel in enumerate(hotels_profiles)}

        authors_iter = self.__authors_generator(self.author_ids)

        # Generators are exhausted after a full iteration hence two iterators needed, for content based recommender and for reviews info
        authors_for_ratings, authors_for_content = tee(authors_iter, 2)
        ratings_info, global_mean = self.__rating_tuples_generator(authors_for_ratings)

        # Initialize and train the combined recommenders
        recommender = HybridRecommender(similarity_function=similarity_function, alpha=alpha, num_factors=latent_factors, lr=lr, reg=reg, epochs=epochs)
        recommender.train(authors_for_content, authors_id_to_index, hotels_profiles, hotel_id_to_index, ratings_info, float(global_mean))
        self.__recommender = recommender
    

    def __get_previous_reviews_author(self, author):
        """Return up to 3 of the author's highest-rated previous reviews from the training data for comparison with recommendations."""
        previous_top_ratings = sorted(author.training_ratings, key=lambda x: x[1], reverse=True) 
        previous_hotel_reviews = []
        for _, (hotel_id, rating) in enumerate(previous_top_ratings[:3]):
            hotel = self.__hotels_database.session.get(Hotel, hotel_id)
            previous_hotel_reviews.append((hotel.state, hotel.web_url, rating, hotel.price))

        return previous_hotel_reviews

    def __get_top_hotels_for_author(self, author: Author, hotel_states: list[str], num_recommendations: int, avoidKnownHotels: bool = True):
        """
        Recommend top hotels for an author. If avoidKnownHotels is True when only unseen hotels are recommended, if avoidKnownHotels is false
        then it can be observed how often the known hotels are recommended on the top num_recommendations.
        """
        avoid_hotels_ids = set([hotel_id for hotel_id, _ in author.training_ratings])
        hotels_ids = self.__hotels_database.get_hotels_ids_from_regions(hotel_states)
        self.hotels_ids =  [] 

        for hotel_id in hotels_ids:
            if avoidKnownHotels:
                if hotel_id not in avoid_hotels_ids:
                    self.hotels_ids.append(hotel_id)
            else:
                self.hotels_ids.append(hotel_id)

        predictions = self.__recommender.predict_selected_hotels_for_author(author.author_id, self.hotels_ids)
        top_n = heapq.nlargest(num_recommendations, predictions, key=lambda x: x[1])
        results = []
        for hotel_id, _ in top_n:
            hotel = self.__hotels_database.session.get(Hotel, hotel_id)
            results.append((hotel.state, hotel.web_url, hotel.price))
        return results 



    ###### --------------------- Public functions ------------------------ ######



    def initialize_app(self, hotels_json_file: str, authors_json_file: str, reviews_json_files: str,
                        amenities_top_components: int = 20, styles_top_components: int = 5):
        """Initialize the application: load classifications, insert data, and compute hotel PCA features."""
        CountriesClassifications.create_classification()
        self.insert_hotels(hotels_json_file)
        self.__hotels_database.calculate_pca_components(amenities_top_components, styles_top_components)
        self.insert_authors(authors_json_file)
        self.insert_reviews(reviews_json_files)
    
    def clean_database(self):
        """Delete all authors and hotels from the database."""
        self.__hotels_database.clean_database()
        self.__authors_database.clean_database()

    # Functions to insert data

    def insert_hotels(self, hotels_json_file: str):
        """Load hotels from a JSON file into the database."""
        self.__hotels_database.load_hotels_from_json(hotels_json_file)

    def insert_authors(self, authors_json_file: str):
        """Load authors from a JSON file into the database, mapping them to classified regions."""
        regions = CountriesClassifications.load_classifications()
        self.__authors_database.load_authors_from_json(regions, authors_json_file)

    def insert_reviews(self, reviews_file: str):
        """
        Load reviews from a JSON file and update author feature profiles. The file should have been previously sorted by the authors 
        id, it is crucial for this function to be optimal. This operation can take up to 40 minutes.
        - Keeps the last review as a test rating.
        - Accumulates feature vectors for content-based recommendations.
        """
        hotels_info = self.__hotels_database.info_for_reviews() #contains hotel_id, feature_vectors and rated_authors

        all_authors_ids = set(self.__authors_database.get_all_authors_ids())
        all_hotels_ids = set([h.hotel_id for h in hotels_info])

        if not all_authors_ids or not all_hotels_ids:
            return #no hotels or authors in the database
        
        hotels_id_to_index = {h.hotel_id: index for index, h in enumerate(hotels_info)}
        length_feature_vec = len(hotels_info[0].feature_vector)
        rated_feature_matrix, ratings, reviews_info = [], [], []
        gc_garbage_track = 0
        with open(reviews_file, 'r', encoding='utf-8') as f:
            current_author = None
            for line in f:
                line = line.strip()
                if not line: # ignore empty lines 
                    continue
                
                review = loads(line)

                author_id = str(review['author_id'])
                hotel_id = int(review['hotel_id'])
                rating = float(review['rating'])

                if current_author is None: 
                    current_author = self.__authors_database.get_author_by_id(author_id)

                if author_id not in all_authors_ids or hotel_id not in all_hotels_ids:
                    continue

                if author_id != current_author.author_id:
                    self.__process_current_author(current_author,
                                                reviews_info,
                                                ratings, 
                                                rated_feature_matrix, 
                                                length_feature_vec,
                                                hotels_info,
                                                hotels_id_to_index)
                    rated_feature_matrix = []
                    ratings = []
                    reviews_info = []
                    current_author = self.__authors_database.get_author_by_id(author_id)
                gc_garbage_track += 1
                hotel_obj = hotels_info[hotels_id_to_index[hotel_id]]

                rated_feature_matrix.append(hotel_obj.feature_vector)
                ratings.append(rating)
                reviews_info.append((hotel_id, rating))
                if author_id not in hotel_obj.rated_by_authors_ids:
                    hotel_obj.rated_by_authors_ids.append(author_id)

                if gc_garbage_track % 500_000 == 0:
                    gc.collect()
                    
            self.__process_current_author(current_author,
                                        reviews_info,
                                        ratings, 
                                        rated_feature_matrix, 
                                        length_feature_vec, 
                                        hotels_info,
                                        hotels_id_to_index)
            
            self.__authors_database.session.commit()
            self.__hotels_database.session.commit()
            self.__authors_database.session.expunge_all() 
            gc.collect() 

    # Functions to delete data - tested in test.py 

    def delete_hotel(self, hotel_id: int):
        """Delete a hotel and remove references to it from authors' reviews."""
        authors_rated = self.__hotels_database.authors_rated_to_hotel(hotel_id)
        self.__hotels_database.delete_hotel(hotel_id)
        self.__authors_database.delete_hotel_from_ratings(authors_rated, hotel_id)

    def delete_author(self, author_id: str):
        """Delete an author and remove their ratings from affected hotels."""
        hotels_affected = self.__authors_database.delete_author(author_id)
        if hotels_affected: #we need to delete the authors from the HotelRatingsIndex
            self.__hotels_database.delete_author_from_hotels(author_id, hotels_affected)

    def update_components(self, amenities_top_components: int, styles_top_components: int):
        """
        Recompute PCA components for hotels and update authors' accumulated feature vectors.
        Ensures that author profiles remain consistent after feature updates.
        """
        self.__hotels_database.calculate_pca_components(amenities_top_components, styles_top_components) 

        all_hotels_profiles = self.__hotels_database.get_hotels_feature_vectors() 
        if not all_hotels_profiles:
            print("No hotels found!")
            return # No hotels in the database 

        hotels_feature_vectors = {
            hotel_profile.hotel_id: np.array(hotel_profile.feature_vector)
            for hotel_profile in all_hotels_profiles
        }

        profile_length = len(all_hotels_profiles[0].feature_vector)

        for author in self.__authors_database.stream_authors(batch_size=1000):
            accumulated_feature_vector = np.zeros(profile_length)
            accumulated_rating = 0
            for (hotel_id, rating) in author.training_ratings:
                accumulated_feature_vector += hotels_feature_vectors[hotel_id] * rating
                accumulated_rating += rating
            author.accumulated_feature_vector = accumulated_feature_vector.tolist()
            author.accumulated_ratings = accumulated_rating

        self.__authors_database.session.commit()
        self.__authors_database.session.expunge_all() 

    def update_means_stds(self):
        """Update hotels' mean and standard deviation to keep recommender calculations consistent."""
        self.__hotels_database.update_means_and_stds(update_feature_vectors=True)

    # Functions that manage the recommendation and statistics generation

    def recommend(self, author_id: str, hotel_states: list[str], num_recommendations: int, similarity_function: bool, 
                  alpha : float, latent_factors: int, lr : float, reg: float, epochs: int):
        self.__prepare_recommender(similarity_function, alpha, latent_factors, lr, reg, epochs)
        """Generate hotel recommendations for a given author."""
        # Get previous ratings history of user and 
        author = self.__authors_database.get_author_by_id(author_id)
        previous_reviews_info = self.__get_previous_reviews_author(author)
        top_recommendations_to_author = self.__get_top_hotels_for_author(author, hotel_states, num_recommendations)
        return previous_reviews_info, top_recommendations_to_author
    
    def generate_statistics(self, regions, hotel_states, output_filename):
        """Generate recommender statistics for selected regions and hotels given the recommender model."""
        authors_iter = self.__authors_generator(self.author_ids)  # generator of authors
        stats = Recommender_Statistics(regions, hotel_states, self.__recommender, self.hotels_ids, authors_iter, output_filename)
        stats.get_stats()

    def get_author_ids_sampled(self, regions: list[str], num_authors_display: int, min_num_author_training_reviews: int):
        """Return a sampled list of author ids based on region and minimum reviews."""
        self.author_ids = self.__authors_database.get_all_authors_ids_by_region_and_min_training(regions, min_num_author_training_reviews)
        if not self.author_ids:
            return []
        
        if len(self.author_ids) <= num_authors_display:
            return self.author_ids
        
        sampled = random.sample(self.author_ids, num_authors_display)
        return sampled
