import os
import argparse
import random
from application import Application

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ABS_PATH_DB_FOLDER = os.path.join(BASE_DIR, 'Databases', 'input_data')

HOTEL_DATA_JSON_ABSOLUTE_PATH = os.path.join(ABS_PATH_DB_FOLDER, 'hotels_data.json')
AUTHORS_DATA_JSON_ABSOLUTE_PATH = os.path.join(ABS_PATH_DB_FOLDER, 'authors_data.json')
REVIEWS_DATA_JSON_ABSOLUTE_PATH = os.path.join(ABS_PATH_DB_FOLDER, 'sorted_reviews.json')

MIN_AMENITIES_PC = 10
MAX_AMENITIES_PC = 100
MIN_STYLES_PC = 5
MAX_STYLES_PC = 40
MIN_NUM_AUTHORS_DISPLAY = 3
MAX_NUM_AUTHORS_DISPLAY = 10
MIN_NUM_RECOMMENDATIONS = 1
MAX_NUM_RECOMMENDATIONS = 10
MIN_NUM_TRAINING_REVIEWS = 1
MAX_NUM_TRAINING_REVIEWS = 5

parser = argparse.ArgumentParser(description="Hotel Recommendation System")
parser.add_argument("--hotels_json_file", type=str, default=HOTEL_DATA_JSON_ABSOLUTE_PATH, help="Introduce a correctly formatted hotel object, readable by line. Works only for initializing the database.")
parser.add_argument("--authors_json_file", type=str, default=AUTHORS_DATA_JSON_ABSOLUTE_PATH, help="Introduce a correctly formatted author object, readable by line. Works only for initializing the database.")
parser.add_argument("--reviews_json_files", type=str, default=REVIEWS_DATA_JSON_ABSOLUTE_PATH, help="Introduce reviews as json file readable by lines. It is expected that the reviews should be from existing authors and hotels, else are ignored.")
### below those two work only for initialization 
parser.add_argument("--amenities_pc", type=int, default=20, help=f"Principal components for amenities. Works only for initializing the database. Choose between [{MIN_AMENITIES_PC}, {MAX_AMENITIES_PC}]")
parser.add_argument("--styles_pc", type=int, default=10, help=f"Principal components for styles. Works only for initializing the database. Choose between [{MIN_STYLES_PC}, {MAX_STYLES_PC}]")
####
parser.add_argument("--min_num_author_training_reviews", type=int, default=1, help=f"Minimum number of training reviews authors from the selection list shoudl have. Choose between [{MIN_NUM_TRAINING_REVIEWS}, {MAX_NUM_TRAINING_REVIEWS}]")
parser.add_argument("--num_authors_display", type=int, default=5, help=f"Maximum number of authors displayed to choose from. Choose between [{MIN_NUM_AUTHORS_DISPLAY}, {MAX_NUM_AUTHORS_DISPLAY}]")
parser.add_argument("--num_recommendations", type=int, default=5, help=f"Maximum number of recommendations. Choose between [{MIN_NUM_RECOMMENDATIONS}, {MAX_NUM_RECOMMENDATIONS}]")
parser.add_argument("--regions", nargs="+", type=int, default=[], help=f"Choose a list of regions:\n1: europe\n2: africa\n3: asia\n4: oceania\n5: latin america\n6: north america\n")
parser.add_argument("--states", nargs="+", type=int, default=[], help=f"Choose the states to visit:\n1: Pennsylvania\n2: Indiana\n3: California\n4: Maryland\n5: North Carolina\n6: Arizona\n7: Washington\n8: Colorado\n9: Ohio\n10: Illinois\n11: Florida\n12: New York\n13: Michigan\n14: Tennessee\n15: District of Columbia\n16: Massachusetts\n17: Texas")
parser.add_argument("--similarity_function", type=bool, default=False, help="Choose content based similarity function, False is the default cosine similarity and True is Pearson Correlation")
parser.add_argument("--random_author", type=bool, default=False, help="Enable/Disable choosing randombly an author")
parser.add_argument("--stats_filename", type=str, default="", help="Introduce filename for the statistic analysis of the recommender, it will be saved at pdfs directory")
parser.add_argument("--alpha", type=float, default=0.5, help="Choose between (0,1) the weight for the content based recommender")
parser.add_argument("--latent_factors", type=int, default=15, help="Latent factors for SVD model")
parser.add_argument("--lr", type=float, default=0.01, help="Learning rate controls how much the model weights are updated during training")
parser.add_argument("--reg", type=float, default=0.01, help="Regularization factor, controls how much we penalize large weights to prevent overfitting")
parser.add_argument("--epochs", type=int, default=15, help="Number of epochs for SGD used in SVD (Funk MF)")



REGIONS_CLASSIFICATION = ['europe', 'africa', 'asia', 'oceania', 'latin america', 'north america', 'unknown', 'Exit']
STATES_OF_HOTELS = ['Pennsylvania', 'Indiana', 'California', 'Maryland', 'North Carolina',
                    'Arizona', 'Washington', 'Colorado', 'Ohio', 'Illinois', 'Florida', 'New York',
                    'Michigan', 'Tennessee', 'District of Columbia', 'Massachusetts', 'Texas', 'Exit']


"""Database: if it is initialized then the program starts right away, if not then it will take around 30 mins
The program displays a list of regions where the authors are groupped, then a list of possible states 
to visit (where we have hotels data) and finally choses some author from some random option.
Depending of the region there are from 5_000 to almost 1_000_000 authors, so the time to wait for the model 
to be generated depends on the amount of authors. 
If chosen statistics a file will be created with them. Check stats.py file for more information about them."""



####### ------------------------- Helper Functions -------------------------------- ######



# Introduce option to stop the program when getting the first options and also for authors introduce introduce yourself an author ID 

def print_options(title: str, options: list):
    print(f"\n{title}")
    for index, option in enumerate(options):
        print(f"{index+1}: {option}")

def get_valid_input(prompt: str, min_val:int, max_val: int) -> int:
    while True:
        try:
            regions_indices = input(prompt)
            indices = [int(num) for num in regions_indices.split(" ")]
            if all(min_val <= num <= max_val for num in indices):
                return indices
            else:
                print(f"Select numbers within the correct range")
        except ValueError:
            print(f"Enter valid numbers of the chosen region")


####### ----------------------------- Program  ----------------------------------- ######


class Program:

    def __init__(self, args):
        self.args = args
        self.app = Application()

    def check_initialization(self):
        """self.app.is_initialized returns True is the databases .db exists, if not it expects the paths of the JSON data files 
        to be imported. This process takes about 30-40 minutes so it is adviced to download the databases and locate them inside 
        Databases/databases, link of the databases: https://drive.google.com/drive/folders/1MUxCoy1YGs44kwCZiNEedRG_Hy40WxFT?usp=sharing"""

        if not self.app.is_initialized:
            print("Initializing app. Can take up to 30 minutes.")
            self.app.initialize_app(
                self.args.hotels_json_file,
                self.args.authors_json_file,
                self.args.reviews_json_files,
                self.args.amenities_pc,
                self.args.styles_pc
            )
        else:
            if self.args.amenities_pc != 20 or self.args.styles_pc != 10:
                self.app.update_components(self.args.amenities_pc, self.args.styles_pc)

    def passed_conditions(self):
        """Check the constraint of those variables are met, the boundaries are shown in --help"""
        return (
            MIN_NUM_AUTHORS_DISPLAY <= args.num_authors_display <= MAX_NUM_AUTHORS_DISPLAY and
            MIN_AMENITIES_PC <= args.amenities_pc <= MAX_AMENITIES_PC and
            MIN_NUM_RECOMMENDATIONS <= args.num_recommendations <= MAX_NUM_RECOMMENDATIONS and
            MIN_NUM_TRAINING_REVIEWS <= args.min_num_author_training_reviews <= MAX_NUM_TRAINING_REVIEWS and 
            MIN_STYLES_PC <= args.styles_pc <= MAX_STYLES_PC
        )
    
    def run(self):

        if not self.passed_conditions():
            print("Invalid parameters")
            return 
        
        self.check_initialization()

        """Select regions where the users belong (aka continents), only 5 are displayed"""

        regions = []
        if (self.args.regions == []):
            print_options("Select Regions", REGIONS_CLASSIFICATION)
            self.args.regions = get_valid_input("Selected region numbers, space separated single string: ", 1, len(REGIONS_CLASSIFICATION))

        if any(val == len(REGIONS_CLASSIFICATION) for val in self.args.regions):
            return; # Exit option

        for region_val in self.args.regions:
                regions.append(REGIONS_CLASSIFICATION[region_val-1])
        print()


        """Select states of where the hotels are located, there are 17 states displayed"""
        
        states = []
        if (self.args.states == []):
            print_options("Select Hotel States", STATES_OF_HOTELS)
            self.args.states = get_valid_input("Selected state numbers, space separated single string: ", 1, len(STATES_OF_HOTELS))
        
        if any(val == len(STATES_OF_HOTELS) for val in self.args.states):
            return; # Exit option
    
        for state_val in self.args.states:
            states.append(STATES_OF_HOTELS[state_val-1])
        print()


        """Select author, the recommendations will be tailored to him/her"""

        # Get author 
        print("Preparing selection of authors...")
        authors = self.app.get_author_ids_sampled(regions, self.args.num_authors_display, self.args.min_num_author_training_reviews)
        if not authors:
            print("No authors found for that selection.")
            return
        
        author_id = 0
        if (self.args.random_author == False):
            print("\nChoose an author from the following list (generated randomly from all possible authors):")
            for i, author in enumerate(authors, 1):
                print(f"{i}: {author}")
            print(f"{len(authors) + 1}: Exit")
            while True:
                try:
                    selection = int(input("Select author id:"))
                    if (1 <= selection <= (len(authors)+1)):
                        break
                except ValueError:
                    continue
            if (selection == len(authors)+1):
                return
            author_id = authors[selection-1]
        else:
            author_id = random.choice(authors)



        """If filename provided then the statistics from stats.py will be calculated with the recommender model, 
        this can make the program quite slow but the results are interesting, pdfs folder contain the file"""
    


        filename = ""
        are_stats_requested = (self.args.stats_filename != "")
        if not are_stats_requested:
            filename = input("\nStatistics take a long time to compute.\nIf you want to save basic statistics about the model please enter output file name without .pdf else just leave it empty: ").strip()
            are_stats_requested = True if filename else False
        else:
            filename = self.args.stats_filename


        """Preparing recommendations and statistics (if any)"""

                
        print("\nPreparing recommendations...\n")
        prev_reviews, recommendations = self.app.recommend(author_id, states, self.args.num_recommendations, self.args.similarity_function, 
                                                            self.args.alpha, self.args.latent_factors, self.args.lr, self.args.reg, self.args.epochs)

        if are_stats_requested:
            print("Preparing statistics about the model...\n")
            self.app.generate_statistics(regions, states, filename)

        self.display_results(prev_reviews, recommendations)


    def display_results(self, prev_reviews, recommendations):
        print("\n     Previous Hotel Reviews      \n\n")
        for state, url, rating, price in prev_reviews:
            print(f"State: {state} - URL: {url} - Rating: {rating} - Price: ${price}")
            print()
        print("\n     Top Recommendations     \n\n")
        for state, url, price in recommendations:
            print(f"State: {state} - URL: {url} - Price: ${price}")
            print()

if __name__ == "__main__":
    args = parser.parse_args()
    Program(args).run()