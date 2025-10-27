import os 
from json import loads
from sqlalchemy import create_engine, Column, String, Integer
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.ext.mutable import MutableList
from sqlalchemy import PickleType
from typing import Optional

""" AuthorsDatabase takes care of adding new authors and keep the database clean. Its public functions are tested in test.py or 
indirectly in application.py by functions used in test.py. Since this database is thought to have 2 million 
of users, we use yield as much as we can to avoid saturating the memory with all authors retrivied at once. 
"""

Base = declarative_base()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ABS_PATH_DB_FOLDER = os.path.join(BASE_DIR, 'databases')

def create_authors_database_dir():
    os.makedirs(ABS_PATH_DB_FOLDER, exist_ok=True)

class Author(Base):
    """
    Author_id is the same id as in TripAdvisor. State will be initialized only for population from USA. 
    Countries that are not found using the utitlities (dictionaries) will be clasified as unknown. 
    Training ratings will be used for the model, there should be at least one training pair, and after that 
    test_rating will keep the last rating found for testing. Accumulated_feature_vector is the weighted 
    (by the ratings) sum of hotel feature vectors and accumulated_ratings is the list of the ratings. 
    This was supposed to accelerate the computations of author profile but so far it is not used due that 
    currently for each author we are subtracting the average ranting to delete differences between too strict 
    users and too kind, so we can evaluate how similar they are. 
    """
    __tablename__ = 'authors'

    author_id = Column(String, primary_key=True)
    state = Column(String)
    country = Column(String)
    region = Column(String)
    training_ratings = Column(MutableList.as_mutable(PickleType), default=list)  # list of [hotel_id, rating]
    test_rating = Column(MutableList.as_mutable(PickleType), default=list)       # [hotel_id, rating]
    accumulated_feature_vector = Column(MutableList.as_mutable(PickleType), default=list)
    accumulated_ratings = Column(Integer, default=0)


class AuthorsDatabase:

    def __init__(self, authorsdb: str = 'authors.db'):
        db_path = os.path.join(ABS_PATH_DB_FOLDER, authorsdb)
        db_url = f'sqlite:///{db_path}'
        self.engine = create_engine(db_url, echo=False)
        Base.metadata.create_all(self.engine)
        Session = sessionmaker(bind=self.engine)
        self.session = Session()
        self.contains_rows = self.session.query(Author).first() is not None

    

    ###### --------------------- Public functions ------------------------ ######



    def load_authors_from_json(self, region_classification_dict: dict, json_path: str):
        """
        Load authors from a JSON file and add them to the database if they don't exist yet.
        Then assigns each author to a region based on their country.
        """
        rows = []
        existing_author_ids = set(self.get_all_authors_ids()) 
        with open(json_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line: # ignore empty lines 
                    continue
                author_json = loads(line)
                author_id = str(author_json['author_id'])
                if author_id in existing_author_ids:
                    continue 
                rows.append(Author(author_id=author_id,
                            state=author_json['state'],
                            country=author_json['country'],
                            region=region_classification_dict.get(author_json['country'], "unknown"),
                            training_ratings=[],
                            test_rating=[],
                            accumulated_feature_vector=[],
                            accumulated_ratings=0))
        if rows:
            self.session.bulk_save_objects(rows)
            self.session.commit()
            
    def delete_author(self, author_id: str) -> list:
        """
        Delete a specific author from the database.
        Returns a list of hotel ids that were affected by removing this author (from training or test ratings).
        Those hotels contain a list of the users that they were rated, the delete author will be removed from those lists.
        """
        author = self.session.get(Author, author_id)
        if not author:
            return []
        hotels_affected = author.training_ratings.copy()
        if author.test_rating:
            hotels_affected.extend(author.test_rating)
        self.session.delete(author)
        self.session.commit()
        author = self.session.get(Author, author_id)
        
        return [] if not hotels_affected else [hotel_id for (hotel_id, _) in  hotels_affected]
    
    def clean_database(self): 
        """
        Reset the database completely by dropping all tables and recreates them from scratch.
        """
        Base.metadata.drop_all(self.engine)
        Base.metadata.create_all(self.engine)

    def delete_hotel_from_ratings(self, rated_authors: list[str], hotel_id: int):
        """
        Remove a specific hotel from all authors' training and test ratings.
        """
        for author_id in rated_authors:
            author = self.session.get(Author, author_id)
            if not author:
                continue
            author.training_ratings = [
                (id, rating) for (id, rating) in author.training_ratings if id != hotel_id
            ]
            if author.test_rating and author.test_rating[0][0] == hotel_id:
                author.test_rating = []
        self.session.commit()
   


    ###### --------------------- Public Queries ------------------------ ######



    def get_all_authors_ids(self) -> list[str]:
        """
        Return a list of all authors id's 
        """
        return [author_id for (author_id,) in self.session.query(Author.author_id).all()]
    
    def get_author_by_id(self, author_id: str) -> Optional[Author]:
        """
        Returns a single author by their id, if not found then return None
        """
        return self.session.get(Author, author_id)
    
    def get_authors_by_regions(self, regions: list, batch_size: int = 1000) -> list[Author]:
        """
        Get all authors belonging to certain regions, loading in batches to avoid memory issues.
        last_id helps to keep track on the position where the batch stopped. 
        """
        last_id = 0
        all_authors = []
        while True:
            batch = (self.session.query(Author).filter(Author.region.in_(regions), Author.author_id > last_id)
                    .order_by(Author.author_id).limit(batch_size).all())
            if not batch:
                break
            all_authors.extend(batch)
            last_id = batch[-1].author_id  # last cursor 
        return all_authors
    
    def get_rated_hotels_from_author(self, author_id) -> list[str]:
        """
        Return a list of hotel ids that this author has rated chekcing both training and test ratings.
        """
        author = self.session.get(Author, author_id)
        if not author:
            return []
        hotels_rated = [hotel_id for hotel_id, _ in author.training_ratings]
        if author.test_rating:
            (hotel_id, _) = author.test_rating[0]
            hotels_rated.append(hotel_id)
        return hotels_rated


    def authors_id_training_rev_by_regions(self, regions: list):
        """
        Fetch author ids along with their training ratings, filtered by regions.
        """
        return self.session.query(Author.author_id, Author.training_ratings).filter(Author.region.in_(regions)).all()
    

    def get_all_authors_ids_by_region_and_min_training(self, regions, min_num_author_training_reviews) -> list[str]:
        """
        Return ids of authors in certain regions who have at least a minimum number of training ratings.
        """
        authors = (
            self.session.query(Author.author_id, Author.training_ratings)
            .filter(Author.region.in_(regions))
            .all()
        )
        if not authors:
            return []
        return [
            author_id
            for author_id, training_ratings in authors
            if training_ratings and len(training_ratings) >= min_num_author_training_reviews
        ]

    def stream_authors(self, batch_size=1000):
        """
        Generator that yields authors in batches, useful for iterating through a large database without loading everything at once in memory.
        """
        last_id = None
        while True:
            q = (
                self.session
                    .query(Author)
                    .order_by(Author.author_id)
            )
            if last_id is not None:
                q = q.filter(Author.author_id > last_id)

            batch = q.limit(batch_size).all()
            if not batch:
                break

            for author in batch:
                yield author

            last_id = batch[-1].author_id