from joblib import load, dump
import os

AFRICA_COUNTRIES = [
    'Algeria', 'Angola', 'Benin', 'Botswana', 'Burkina Faso', 'Burundi', 'Cabo Verde', 'Togo',
    'Cameroon', 'Central African Republic', 'Chad', 'Comoros', 'Democratic Republic of the Congo',
    'Republic of the Congo', 'Côte d\'Ivoire', 'Djibouti', 'Egypt', 'Equatorial Guinea', 'Tunisia',
    'Eritrea', 'Eswatini', 'Ethiopia', 'Gabon', 'Gambia', 'Ghana', 'Guinea', 'Guinea-Bissau', 'Kenya',
    'Lesotho', 'Liberia', 'Libya', 'Madagascar', 'Malawi', 'Mali', 'Mauritania', 'Mauritius', 'Morocco',
    'Mozambique', 'Namibia', 'Niger', 'Nigeria', 'Rwanda', 'São Tomé and Príncipe', 'Senegal', 'Seychelles',
    'Sierra Leone', 'Somalia', 'South Africa', 'South Sudan', 'Sudan', 'Tanzania', 'Uganda', 'Zambia', 'Zimbabwe'
] 

NORTH_AMERICA_COUNTRIES = ['Canada', 'USA']

LATIN_AMERICA_COUNTRIES = [
    'Mexico', 'Belize', 'Costa Rica', 'El Salvador', 'Guatemala', 'Honduras', 'Nicaragua', 'Panama',
    'Cuba', 'Dominican Republic', 'Haiti', 'Jamaica', 'Trinidad and Tobago',
    'Bahamas', 'Barbados', 'Grenada', 'Antigua and Barbuda', 
    'Saint Kitts and Nevis', 'Saint Lucia', 'Saint Vincent and the Grenadines', 'Dominica',
    'Argentina', 'Bolivia', 'Brazil', 'Chile', 'Colombia', 'Ecuador',
    'Guyana', 'Paraguay', 'Peru', 'Suriname', 'Uruguay', 'Venezuela'
]

EUROPE_COUNTRIES = [
    'Albania', 'Andorra', 'Armenia', 'Austria', 'Belarus', 'Belgium', 'Bosnia and Herzegovina',
    'Bulgaria', 'Croatia', 'Cyprus', 'Czech Republic', 'Denmark', 'Estonia', 'Finland', 'France', 'Georgia',
    'Germany', 'Greece', 'Hungary', 'Iceland', 'Ireland', 'Italy', 'Kazakhstan', 'Kosovo', 'Latvia', 'Liechtenstein',
    'Lithuania', 'Luxembourg', 'Malta', 'Moldova', 'Monaco', 'Montenegro', 'Netherlands', 'North Macedonia',
    'Norway', 'Poland', 'Portugal', 'Romania', 'Russia', 'San Marino', 'Serbia', 'Slovakia', 'Slovenia',
    'Spain', 'Sweden', 'Switzerland', 'Ukraine', 'United Kingdom'
]

ASIA_COUNTRIES = [
    'Afghanistan', 'Armenia', 'Azerbaijan', 'Bahrain', 'Bangladesh', 'Bhutan', 'Brunei', 'Cambodia',
    'China', 'Cyprus', 'East Timor', 'Georgia', 'India', 'Indonesia', 'Iran', 'Iraq', 'Israel', 'Japan',
    'Jordan', 'Kazakhstan', 'Kuwait', 'Kyrgyzstan', 'Laos', 'Lebanon', 'Malaysia', 'Maldives', 'Mongolia',
    'Myanmar', 'Nepal', 'North Korea', 'Oman', 'Pakistan', 'Palestine', 'Philippines', 'Qatar', 'Saudi Arabia',
    'Singapore', 'South Korea', 'Sri Lanka', 'Syria', 'Tajikistan', 'Thailand', 'Turkey', 'Turkmenistan',
    'United Arab Emirates', 'Uzbekistan', 'Vietnam', 'Yemen'
]

OCEANIA_COUNTRIES = [
    'Australia', 'Fiji', 'Kiribati', 'Marshall Islands', 'Micronesia', 'Nauru', 'New Zealand',
    'Palau', 'Papua New Guinea', 'Samoa', 'Solomon Islands', 'Tonga', 'Tuvalu', 'Vanuatu'
]

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ABS_PATH_JOBLIB_FOLDER = os.path.join(BASE_DIR, 'joblib_dumps')

def create_classifications_dir():
    os.makedirs(ABS_PATH_JOBLIB_FOLDER, exist_ok=True)

class CountriesClassifications:

    @staticmethod
    def create_classification(region_classification_file: str = 'regions_classification.joblib'):
        """ 
        Create Dictionaries so it would be straightforward and fast to map the countries of the authors 
        given the JSON file to the specific continent (and hence make queries faster). This function is 
        needed when importing authors. 
        """
        file_path = os.path.join(ABS_PATH_JOBLIB_FOLDER, region_classification_file)
        africa_dict = {country: 'africa' for country in AFRICA_COUNTRIES}
        europe_dict= {country: 'europe' for country in EUROPE_COUNTRIES}
        asia_dict = {country: 'asia' for country in ASIA_COUNTRIES}
        oceania_dic  = {country: 'oceania' for country in OCEANIA_COUNTRIES}
        latin_america_dict = {country: 'latin america' for country in LATIN_AMERICA_COUNTRIES}
        north_america_dict = {country: 'north america' for country in NORTH_AMERICA_COUNTRIES}
        region_classification_dict = {**africa_dict, **europe_dict, **asia_dict, **oceania_dic, 
                                     **latin_america_dict, **north_america_dict}
        dump(region_classification_dict, file_path)

    @staticmethod
    def load_classifications(region_classification_file: str = 'regions_classification.joblib'):
        """
        Retrieved the dictionaried to authordb.py module
        """
        file_path = os.path.join(ABS_PATH_JOBLIB_FOLDER, region_classification_file)
        try:
            return load(file_path)
        except FileNotFoundError:
            return [] # Means create_classification() was not runned first 