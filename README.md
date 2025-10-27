## RegionNet 
### Region-Based Recommender System

- **Student: Angie Aslhy Granda Nunez**  
- **Supervisor: doc. RNDr. Iveta Mrázová, CSc.**  
- **Individual Software Project NPRG045 – Summer 2025**  

## Project Specifications  

- **Preferred language:** Python  
- **Software tools for implementation:** Surprise, LightFM, Implicit, PyTorch, TensorFlow, Scikit-Learn, Selenium
- **Main research question:** Does grouping users by region of origin enhance the performance of recommender systems?  

## Introduction

What preferences do people from the same region have when choosing a hotel? A hybrid hotel recommender system can help us analyze how guest preferences vary based on nationality. This project aims to develop a user-friendly recommendation system that suggests hotels to users in cities from the dataset, based on their region of origin and the reviews of people who have previously visited those cities.

## Goals  

- Provide the **top 5 hotel recommendations** for a specific user given hyperparameters such as PCA components for the feature vector, latent factors for SVD model, similarity metric chosen from either cosine similarity or pearson correlation. And many more, see --help. 
- Analyze how the **information on reviewers' nationality** improves the results of the recommendation system.  
- Enable hotel owners to assess their hotel ratings based on guest nationalities, which could help attract specific demographics.  

## Main Steps  

The objective is to build a **hotel recommendation system** that tailors suggestions to users by considering their preferences, which are inferred from hotel characteristics and the ratings users have given to hotels. The system will rely on three key data types: **user data, hotel features, and user reviews with ratings**.  

The **hybrid recommendation system** will combine **Content-Based Filtering**, which utilizes hotel features for predictions, and **Collaborative Filtering** through **Matrix Factorization (SVD)**.  

## Data Organization  

- Users will be grouped by **IDs and countries**.  
- Hotel features (e.g., basic information, amenities) will be preprocessed using **feature engineering techniques**.  
- Ratings will be used to build a **user-item interaction matrix**.  

## Content-Based Filtering  

A reliable approach when the feature vectors representing items are large. By leveraging **vector representations of hotel features**, we will evaluate different similarity functions for optimal prediction performance. The options include:  

- **Cosine Similarity**  
- **Pearson Correlation**  

## Collaborative Filtering using SVD (Singular Value Decomposition - Matrix Factorization)  

Applying **SVD** to the user-item interaction matrix allows us to decompose it into **lower-dimensional matrices**, helping to identify **latent factors** that capture hidden relationships between users and hotels.  

## Hybrid Approach  

This method merges predictions from **SVD and Content-Based Filtering** using a **weighted average**, mitigating the **cold start problem** that arises when collaborative filtering is used alone. The user-item interactions will be analyzed to uncover hidden patterns, making the model effective for **sparse data**.  

## Evaluation Metrics  

- **RMSE (Root Mean Squared Error)**  
- **MAE (Mean Absolute Error)**  

## Decomposition 

Modules:
- Inside app/Databases/ there are authorsdb.py and hotelsdb.py handle the data of authors, hotels and ratings.
- app/ contains:
    - application.py - Manages the databases, the recommender and the statistic class. Interacts with Program.
    - program.py:  Takes input and redirects it to Application Class. Obtains output and renders it in the console. 
    - recommender.py: Hybrid recommender system, controlled by Application Class.
    - stats.py: Reports statistics of the recommender model. Application Class owns it and hence passes the instance of the Hybrid Recommender.
For more information see Report.pdf.

## Bibliography  

1. Aggarwal, C. C. (2016). Recommender systems: The textbook. Springer. Available from Springer at [link](https://link.springer.com/book/10.1007/978-3-319-29659-3)
2. Behun, M. (2022). TripAdvisor data-scraper. GitHub. [https://github.com/](https://github.com/elkablo/gnn-social-tripadvisor?tab=readme-ov-file)
3. Kulkarni, A., & Shivananda, A. (2023). Applied recommender systems with Python: Build recommender systems with deep learning, NLP, and graph-based techniques. Apress. [https://github.com/](https://github.com/elkablo/gnn-social-tripadvisor?tab=readme-ov-file)
4. Parashar, N. (2023). Memory-based vs. model-based collaborative filtering techniques. Retrieved from [link](https://medium.com/@niitwork0921/memory-based-vs-model-based-collaborative-filtering-techniques-c0a7f6ec4f5f)
5. Leskovec, J. (2025). CS246: Mining massive datasets, Chapter 7: Recommender systems. Stanford University. Available at [Lecture slides](https://web.stanford.edu/class/cs246/slides/07-recsys1.pdf)
6. Ullman, J. D. (n.d.). Recommender Systems. Stanford University. Available at [lecture notes](http://infolab.stanford.edu/~ullman/mmds/ch9.pdf)
