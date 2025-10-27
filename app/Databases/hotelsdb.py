import os
from json import loads
from joblib import load, dump
import numpy as np
from sqlalchemy import create_engine, Column, Integer, String, Float, JSON, Boolean
from sqlalchemy.orm import declarative_base, sessionmaker
from sklearn.decomposition import PCA
from sklearn.preprocessing import MultiLabelBinarizer
from sqlalchemy.ext.mutable import MutableList
from sqlalchemy.orm import load_only

""" Most of the properties of Hotel class are directly retrieval from tripadvisor API, others such as *_trip_rat category 
    is computed by me as a result of total_cat1_trips/total_trips when retriving the data since it was quite messy combining 
    the data from the github repository mentioned and aditional one from the API + manual filling, I didnt include the retrieval 
    of the data, so we assume that the files should have some specific format. 
    
    The total amenities features of the hotels are around 250 and the total styles about 50. PCA helps to reduce the dimentions
    which is needed for working with the recommenders.

    Main functionalities of HotelsDatabase is to insert hotels, calculate the feature vector of the hotels and update it 
    when the principal components constraint changes. Keep the database clean and provide queries for application.py
"""

Base = declarative_base()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ABS_PATH_JOBLIB_FOLDER = os.path.join(BASE_DIR, 'joblib_dumps')
ABS_PATH_DB_FOLDER = os.path.join(BASE_DIR, 'databases')


def create_hotels_database_dir():
    os.makedirs(ABS_PATH_DB_FOLDER, exist_ok=True)

class Hotel(Base):
    __tablename__ = 'hotels'

    hotel_id = Column(Integer, primary_key=True)
    city = Column(String)
    state = Column(String)
    web_url = Column(String)
    num_rooms = Column(Integer)
    styles = Column(JSON)
    name = Column(String)
    avg_rating = Column(Float) 
    location_rat = Column(Float)
    sleep_rat = Column(Float)
    room_rat = Column(Float)
    service_rat = Column(Float)
    price_rat = Column(Float)
    clean_rat = Column(Float)
    price = Column(Float)
    business_trip_rat = Column(Float) # bussiness_trips_registered / total_trips
    couple_trip_rat = Column(Float) # couple_trips_registered / total_trips 
    solo_trip_rat = Column(Float) # same logic, information calculated from tripadvisor
    family_trip_rat = Column(Float) 
    friends_trip_rat = Column(Float) 
    amenities = Column(JSON)
    stars = Column(Float) 
    feature_vector = Column(JSON)
    rated_by_authors_ids = Column(MutableList.as_mutable(JSON))  

class HotelsDatabase:

    def __init__(self,  hotelsdb='hotels.db', 
                        amenities_pca_joblib_file = 'amenities_pca.joblib',
                        styles_pca_joblib_file = 'styles_pca.joblib', 
                        amenities_binarizer_joblib_file = 'amenities_binarizer.joblib', 
                        styles_binarizer_joblib_file = 'styles_binarizer.joblib', 
                        standarization_info_joblib = 'standarization_info.joblib' ):
        
        db_url = f'sqlite:///{os.path.join(ABS_PATH_DB_FOLDER, hotelsdb)}'
        self.engine = create_engine(db_url, echo=False)
        Base.metadata.create_all(self.engine)
        Session = sessionmaker(bind=self.engine)
        self.session = Session()
        self.contains_rows = self.session.query(Hotel).first() is not None
        self.amenities_pca_joblib_file = os.path.join(ABS_PATH_JOBLIB_FOLDER, amenities_pca_joblib_file)
        self.styles_pca_joblib_file = os.path.join(ABS_PATH_JOBLIB_FOLDER, styles_pca_joblib_file)
        self.amenities_binarizer_joblib_file = os.path.join(ABS_PATH_JOBLIB_FOLDER, amenities_binarizer_joblib_file)
        self.styles_binarizer_joblib_file = os.path.join(ABS_PATH_JOBLIB_FOLDER, styles_binarizer_joblib_file)
        self.standarization_info_joblib = os.path.join(ABS_PATH_JOBLIB_FOLDER, standarization_info_joblib)
        self.session.expunge_all() 
        


    ###### --------------------- Private functions ---------------------- #######



    def __create_feature_vector(self, hotel: Hotel, amenities_pca: np.ndarray, styles_pca: np.ndarray):
        """
        Create the feature vector for a single hotel by combining z-standaratization for rooms, prices, stars, 
        averages ratings, PCA components for both amenities and styles and types of trips.
        """
        info = load(self.standarization_info_joblib)
        flat_amenities = amenities_pca[0].tolist()   # amenities_pca and styles_pca has shape (1, D) 
        flat_styles = styles_pca[0].tolist()

        features = [
                (hotel.num_rooms - info['room_mean']) / info['room_std'], 
                (hotel.price - info['price_mean']) / info['price_std'], 
                (hotel.stars - info['stars_mean']) / info['stars_std'],
                (hotel.avg_rating - info['average_rating_mean']) / info['average_rating_std'],
                (hotel.sleep_rat - info['sleep_rating_mean']) / info['sleep_rating_std'], 
                (hotel.service_rat - info['service_rating_mean']) / info['service_rating_std'],
                (hotel.price_rat - info['price_rating_mean']) / info['price_rating_std'],
                (hotel.clean_rat - info['clean_rating_mean']) / info['clean_rating_std']
            ] + [
                hotel.couple_trip_rat, hotel.solo_trip_rat, hotel.family_trip_rat, 
                hotel.business_trip_rat, hotel.friends_trip_rat
            ] + flat_amenities + flat_styles
        
        floats_feature = [float(num) for num in features] #done because some numbers from features are np.float
        hotel.feature_vector = floats_feature


    def __set_feature_vector(self, hotel: Hotel, update: bool = False):
        """
        Compute or update a hotel's feature vector, recalculating PCA if new amenities or styles are found.  
        This ensures that new categories in hotel data are always incorporated into the model, however, 
        since this call is expensive, the update will be done using a boolean.
        """
        pca_amenities, amenities_binarizer, pca_styles, styles_binarizer = self.__load_pca_data()
        if (update):
            unknown_styles = set(hotel.styles) - set(styles_binarizer.classes_)
            unknown_amenities = set(hotel.amenities) - set(amenities_binarizer.classes_)
            # if new amenities or styles categories found then compute PCA
            if unknown_styles or unknown_amenities:
                self.__run_pca(pca_amenities.n_components_, pca_styles.n_components_)
                pca_amenities, amenities_binarizer, pca_styles, styles_binarizer = self.__load_pca_data()

        amenities_binarized = amenities_binarizer.transform([hotel.amenities])
        styles_binarized = styles_binarizer.transform([hotel.styles])
        amenities_pca = pca_amenities.transform(amenities_binarized)
        styles_pca = pca_styles.transform(styles_binarized)

        self.__create_feature_vector(hotel, amenities_pca, styles_pca)

    def __run_pca(self, top_k_amenities: int, top_k_styles: int): # we want to save those changes in Hotels Management 
        """
        Train and fit PCA for amenities and styles across all hotels and store the PCA models.  
        This reduces high-dimensional features 250 for amenities and 50 for styles to lower 
        dimentions making easier the computations and also capturing hidden features.
        """
        all_amenities = []
        all_styles = []

        for hotel in self.__stream_hotels():
            all_amenities.append(hotel.amenities)
            all_styles.append(hotel.styles)

        amenities_binarizer = MultiLabelBinarizer()
        styles_binarizer = MultiLabelBinarizer()
        amenities_binarized = amenities_binarizer.fit_transform(all_amenities)
        styles_binarized = styles_binarizer.fit_transform(all_styles)
        pca_amenities = PCA(n_components=top_k_amenities)
        pca_styles = PCA(n_components=top_k_styles)
        pca_amenities.fit(amenities_binarized)
        pca_styles.fit(styles_binarized)
        self.__save_pca_data(pca_amenities, amenities_binarizer, pca_styles, styles_binarizer)


    def __save_pca_data(self, pca_amenities, amenities_binarizer, pca_styles, styles_binarizer):
        """
        Save PCA and binarizer objects to disk using joblib for later loading.
        """
        dump(pca_amenities, self.amenities_pca_joblib_file)
        dump(amenities_binarizer, self.amenities_binarizer_joblib_file)
        dump(pca_styles, self.styles_pca_joblib_file)
        dump(styles_binarizer, self.styles_binarizer_joblib_file)

    def __load_pca_data(self):
        """
        Load PCA and binarizer objects from disk.
        """
        pca_amenities = load(self.amenities_pca_joblib_file)
        amenities_binarizer = load(self.amenities_binarizer_joblib_file)
        pca_styles = load(self.styles_pca_joblib_file)
        styles_binarizer = load(self.styles_binarizer_joblib_file)
        return pca_amenities, amenities_binarizer, pca_styles, styles_binarizer
    
    def __create_hotel_entry(self, hotel_id: int, hotel_data: JSON, amenities: list, styles: list):
        """
        Create a new Hotel object from JSON data, filling missing fields with defaults.  
        """
        return Hotel ( hotel_id=hotel_id,
                city=hotel_data['city'],
                state=hotel_data['state'],
                web_url=f"https://www.tripadvisor.com/{hotel_data['web_url']}",
                num_rooms=int(hotel_data.get('num_rooms') or 10),
                styles=styles,
                name=hotel_data['name'],
                avg_rating=float(hotel_data.get('avg_rating') or 2.5),
                location_rat=float(hotel_data.get('location_rat') or 2.5),
                sleep_rat=float(hotel_data.get('sleep_rat') or 2.5),
                room_rat=float(hotel_data.get('room_rat') or 2.5),
                service_rat=float(hotel_data.get('service_rat') or 2.5),
                price_rat=float(hotel_data.get('price_rat') or 2.5),
                clean_rat=float(hotel_data.get('clean_rat') or 2.5),
                price=float(hotel_data.get('price') or 50.0),
                business_trip_rat=float(hotel_data.get('business_trip_rat') or 0.2),
                couple_trip_rat=float(hotel_data.get('couple_trip_rat') or 0.2),
                solo_trip_rat=float(hotel_data.get('solo_trip_rat') or 0.2),
                family_trip_rat=float(hotel_data.get('family_trip_rat') or 0.2),
                friends_trip_rat=float(hotel_data.get('friends_trip_rat') or 0.2),
                amenities=amenities,
                stars=float(hotel_data.get('stars') or 2.5), 
                feature_vector=[], 
                rated_by_authors_ids=[])

    def __stream_hotels(self, batch_size=100):
        """
        Yield hotels from the database in a memory efficient way using batches.  
        Useful for processing all hotels without loading the entire database into memory.
        """
        query = self.session.query(Hotel).yield_per(batch_size)
        for hotel in query:
            yield hotel



    ###### --------------------- Public functions ---------------------- #######



    def update_means_and_stds(self, update_feature_vectors : bool = False): 
        """
        Calculate means and standard deviations for numeric hotel fields and store them using joblib. 
        If update_feature_vectors is true then update all hotel feature vectors based on these new statistics.  
        """
        hotels = self.session.query(Hotel).yield_per(1000) 
        rooms, prices, stars = [], [], []
        average_ratings, sleep_ratings, service_ratings = [], [], []
        price_ratings, clean_ratings, location_ratings= [], [], []

        for hotel in hotels:
            rooms.append(hotel.num_rooms)
            prices.append(hotel.price)
            stars.append(hotel.stars)
            average_ratings.append(hotel.avg_rating)
            sleep_ratings.append(hotel.sleep_rat)
            service_ratings.append(hotel.service_rat)
            price_ratings.append(hotel.price_rat)
            clean_ratings.append(hotel.clean_rat)
            location_ratings.append(hotel.location_rat)

        standarization_info = { 'room_mean': np.mean(rooms), 
                                'room_std': np.std(rooms), 
                                'price_mean': np.mean(prices), 
                                'price_std': np.std(prices),
                                'stars_mean': np.mean(stars), 
                                'stars_std': np.std(stars),
                                'average_rating_mean': np.mean(average_ratings), 
                                'average_rating_std': np.std(average_ratings),
                                'sleep_rating_mean': np.mean(sleep_ratings), 
                                'sleep_rating_std': np.std(sleep_ratings),
                                'service_rating_mean': np.mean(service_ratings), 
                                'service_rating_std': np.std(service_ratings),
                                'price_rating_mean': np.mean(price_ratings), 
                                'price_rating_std': np.std(price_ratings),
                                'clean_rating_mean': np.mean(clean_ratings), 
                                'clean_rating_std': np.std(clean_ratings),
                                'location_rating_mean': np.mean(location_ratings),
                                'location_rating_std': np.std(location_ratings) } 
        
        dump(standarization_info, self.standarization_info_joblib)

        if (update_feature_vectors): 
            for hotel in hotels:
                self.__set_feature_vector(hotel)
            self.session.commit()
    
    def clean_database(self): 
        """
        Reset the hotel database completely by dropping and recreating all tables.
        """
        Base.metadata.drop_all(self.engine)
        Base.metadata.create_all(self.engine)

    def load_hotels_from_json(self, json_path: str):
        """
        Load hotels from a JSON file into the database.  
        Skips existing hotels, fills missing values with defaults, and calculates feature vectors if needed.
        """
        existing_ids = {hotel_id for (hotel_id,) in self.session.query(Hotel.hotel_id).all()} 
        new_hotels = []
        with open(json_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line: # ignore empty lines 
                    continue
                hotel_data = loads(line)
                hotel_id = int(hotel_data['hotel_id'])
                if hotel_id in existing_ids:
                    continue
                styles = hotel_data.get('styles') or []
                amenities = hotel_data.get('amenities') or []
                hotel = self.__create_hotel_entry(hotel_id, hotel_data, amenities, styles)
                if existing_ids:
                    self.__set_feature_vector(hotel)
                new_hotels.append(hotel)

        if new_hotels:
            self.session.bulk_save_objects(new_hotels)
        self.session.commit()

        if not existing_ids:
            self.update_means_and_stds()

    def calculate_pca_components(self, top_k_amenities: int, top_k_styles: int):
        """
        Recompute PCA for amenities and styles with given number of components, usually
        when the dimensionality of features needs adjustment or new categories are added.
        """
        self.__run_pca(top_k_amenities, top_k_styles) 
        for hotel in self.__stream_hotels():
            self.__set_feature_vector(hotel)
        self.session.commit()

    # deleting functions 

    def delete_author_from_hotels(self, author_id: str, hotels_affected: list[int]):
        """
        Remove a specific author's id from the rated_by_authors_ids of affected hotels.  
        Keeps the hotel data consistent after an author is deleted.
        """
        for hotel_id in hotels_affected:
            hotel = self.session.get(Hotel, hotel_id)
            if hotel: 
                if author_id in hotel.rated_by_authors_ids:
                    hotel.rated_by_authors_ids.remove(author_id)
        self.session.commit()

    def delete_hotel(self, hotel_id: int):
        """
        Delete a hotel from the database completely.
        """
        hotel = self.session.get(Hotel, hotel_id)
        if hotel:
            self.session.delete(hotel)
            self.session.commit()



    ###### --------------------- Queries functions ---------------------- #######



    def info_for_reviews(self):
        """
        Return minimal hotel info needed for recommendations: hotel_id, feature_vector, and authors who rated it.
        """
        return self.session.query(Hotel).options(
            load_only(Hotel.hotel_id, Hotel.feature_vector, Hotel.rated_by_authors_ids)
        ).all()

    def authors_rated_to_hotel(self, hotel_id: int):
        """
        Return a list of author ids who rated a given hotel.
        """
        hotel = self.session.get(Hotel, hotel_id)
        if not hotel:
            return []
        return hotel.rated_by_authors_ids
    
    def get_hotels_ids_from_regions(self, states: list[str]):
        """
        Return hotel ids for all hotels located in specific states.
        """
        return [hotel_id for (hotel_id,) in self.session.query(Hotel.hotel_id).filter(Hotel.state.in_(states)).all()]
    
    def get_hotels_feature_vectors(self):
        """
        Return hotels with only their hotel_id and feature_vector.  
        """
        query = (
        self.session.query(Hotel)
        .options(load_only(Hotel.hotel_id, Hotel.feature_vector)))
        hotels = query.all()
        return hotels

