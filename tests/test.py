import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../app')))

from application import Application
from Databases.authorsdb import AuthorsDatabase
from Databases.hotelsdb import HotelsDatabase, Hotel

app = Application()
authors_database = AuthorsDatabase()
hotels_database = HotelsDatabase()


"""Tested functions:
    From AuthorsDatabase: get_author_by_id
    From HotelsDatabase: get_hotels_rating_history 
"""
def test_hotel_and_author_profiles(author_id: str = "046995795A7F4147090BE5E7D357F18A", 
                                   hotel_id: int = 122332):
    
    hotel = hotels_database.session.get(Hotel, hotel_id)
    assert hotel is not None
    author = authors_database.get_author_by_id(author_id)
    assert len(author.accumulated_feature_vector) == len(hotel.feature_vector) #same dimentions 

    hotels_rated_by_author = authors_database.get_rated_hotels_from_author(author_id)
    assert len(hotels_rated_by_author) > 0
    assert hotel_id in hotels_rated_by_author

    rated_authors = hotels_database.authors_rated_to_hotel(hotel_id)
    assert len(rated_authors) > 0
    assert author_id in rated_authors


""" Tested functions:
    From Application: update_means_stds
    Shouldnt change nothing yet since there were no modifications in the dataset 
    hotel_id can be changed for any other from input_data
"""
def test_update_means_and_stds(hotel_id: int = 122332):
    hotel = hotels_database.session.get(Hotel, hotel_id)
    assert hotel is not None
    old_hotel_profile = hotel.feature_vector

    app.update_means_stds()
    authors_database.session.expunge_all()

    hotel = hotels_database.session.get(Hotel, hotel_id)
    assert old_hotel_profile == hotel.feature_vector


"""Tested functions: This function takes about 10 minutes
    From Application: update_components 
    Queries to check if the hotel feature vector and user profile (weigthed feature vector) was correctly uploaded
    author_id and hotel_id can be changed to any id from input_data used to initialized 
"""

# def test_pca_updates_and_dimentions_of_features(author_id: str = "A86B734C11BBAC987C1E6211A85C06F9",
#                                                 hotel_id: int = 113317, 
#                                                 k_amenities: int = 22, 
#                                                 l_styles = 11):
#     hotel = hotels_database.session.get(Hotel, hotel_id)
#     assert hotel is not None
#     hotel_profile = hotel.feature_vector

#     #There are 13 features (ratings if the hotels at different aspects) + amenities and styles which are computed with PCA
#     current_amenities_plus_styles = len(hotel_profile) - 13 
#     correct_diff = current_amenities_plus_styles - (k_amenities + l_styles)

#     app.update_components(k_amenities, l_styles)
#     hotels_database.session.expire_all()
#     authors_database.session.expire_all()

#     author = authors_database.get_author_by_id(author_id)
#     hotel = hotels_database.session.get(Hotel, hotel_id)
#     hotel_profile = hotel.feature_vector

#     assert len(author.accumulated_feature_vector) == len(hotel_profile) #same dimentions 
#     assert correct_diff == (current_amenities_plus_styles - (len(hotel_profile) - 13))


"""  Tested functions: 
    From Application: insert_authors, insert_reviews, delete_author
    From AuthorsDatabase: get_author_by_id, get_rated_hotels_from_author
        within app.delete_author: delete_author 
    From HotelsDatabase: authors_rated_to_hotel
        within app.delete_author: delete_author_from_hotels 
"""
def test_inserting_deleting_author(author_filename: str = "file1.json", 
                                    review_filename: str = "reviews1.json",
                                    author_id: str = "F82AE1AEB896EB0611776F315AFAA877"):
    # inserting fake author and reviews with existing hotels
    app.insert_authors(author_filename)
    app.insert_reviews(review_filename)

    #Test authors database that the author and reviews were well processed 
    author = authors_database.get_author_by_id(author_id)
    assert author is not None

    hotels_id = authors_database.get_rated_hotels_from_author(author_id)
    for hotel_id in hotels_id:
        history = hotels_database.authors_rated_to_hotel(hotel_id)
        assert author_id in history

    # Deleting fake author
    app.delete_author(author_id)
    authors_database.session.expire_all() #cleaning the cache, necessary to update the changes in the database 

    #Check the author was erased well from hotels 
    author = authors_database.get_author_by_id(author_id)
    assert author is None
    for hotel_id in hotels_id:
        history = hotels_database.authors_rated_to_hotel(hotel_id)
        assert author_id not in history


""" Tested functions: 
    From Application: insert_hotels, insert_reviews, delete_hotel
    From AuthorsDatabase: get_author_by_id, get_rated_hotels_from_author
        within app.delete_hotel: delete_hotel_from_ratings 
    From HotelsDatabase: authors_rated_to_hotel
        within app.delete_hotel: delete_hotel 
    existing_author is an author_id from input_data which is used in file1.json, if it is changed for another author then make sure it is also set in file1.json
    existing_hotel_id can be changed for any in input_data
"""
def test_insert_and_delete_hotel(hotel_filename: str = "file2.json", 
                                reviews_filename: str = "reviews2.json",
                                existing_author: str = "A86C61DEC98B1624B9EE5D960182BD92", #author assigned to the fake reviews
                                new_hotel_id: int = 93523456789,   
                                existing_hotel_id: int = 217616):
    app.insert_hotels(hotel_filename) 
    app.insert_reviews(reviews_filename) 

    new_hotel = hotels_database.session.get(Hotel, new_hotel_id)
    assert new_hotel is not None
    new_hotel_profile = new_hotel.feature_vector

    existing_hotel = hotels_database.session.get(Hotel, existing_hotel_id)
    assert existing_hotel is not None
    existing_hotel_profile = existing_hotel.feature_vector

    assert len(new_hotel_profile) == len(existing_hotel_profile)

    # Test reviews information correctly introduced in the hotels and authors databases
    authors_rated = hotels_database.authors_rated_to_hotel(new_hotel_id)
    assert authors_rated is not None
    assert existing_author in authors_rated
    author = authors_database.get_author_by_id(existing_author)
    hotels_rated = authors_database.get_rated_hotels_from_author(existing_author)
    assert new_hotel_id in hotels_rated

    # Deleting the fake hotel, then the information of the reviews also should be updated in databases
    app.delete_hotel(new_hotel_id) # this shouldn raise error if not found 
    hotels_database.session.expire_all()
    authors_database.session.expire_all()

    # Test databases that dont contain information about this hotels or the reviews associated with him
    hotel = hotels_database.session.get(Hotel, new_hotel_id)
    assert hotel is None
    hotels_rated = authors_database.get_rated_hotels_from_author(existing_author)
    assert new_hotel_id not in hotels_rated
    