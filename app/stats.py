import os
import heapq
import random
import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
from typing import Tuple, Iterable
from collections import defaultdict
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.metrics import mean_absolute_error, mean_squared_error
from recommender import HybridRecommender

"""Statistics about the recommender model. 
    - __get_author_training_size_distribution() displays the distribution of authors based on the training reviews 
    - __get_prediction_top_hotels_distribution() pie chart showing how ofter some spefici hotel appearns in the top 5 recommendations
    Others category is formed for all hotels with less than 2% chosen time/ total times
    - __get_accuracy_per_authors_cluster_by_training_size() wnats to show how prediction error changes based on the number 
    of training samples per author. 
    - __evaluate() get the MAE and RMSE 
"""

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, 'pdfs')

def create_output_dir_stats():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

class Recommender_Statistics:

    def __init__(self, 
                 regions:list[str], 
                 states: list[str], 
                 recommender: HybridRecommender,
                 hotels_ids: list, 
                 authors_iter: Iterable, 
                 filename: str = 'output'):
        self.recommender = recommender
        self.regions = regions
        self.states = states
        self.pdf = PdfPages(os.path.join(OUTPUT_DIR, f"{filename}.pdf"))
        self.hotels_ids = hotels_ids
        self.authors_iter = authors_iter  # iterable or generator of authors

    def get_stats(self):
        """
        This functions gathers the data necessary for posterior tests while sampling around 70% of the total authors.
        training_sizes -> list of the size of the autjor's training data 
        hybrid_truths -> contains the true ratings given by the author and hybrid_preds is the prediction of the hybrid recommender
        top_5_hotels_scores -> records a dictionary where the key is the hotel.id and the value is the amount of times that hotel
        appeared on the top 5 hotels recommended for all authors sampled.
        """
        training_sizes = []
        hybrid_truths, hybrid_preds = [], []
        top_5_hotels_scores = defaultdict(int)
        cluster_errors = defaultdict(list)

        factor = 0.3
        
        for author in self.authors_iter:
            if random.random() < factor:
                author_id = author.author_id

                training_size = len(author.training_ratings)
                training_sizes.append(training_size)


                if author.test_rating:
                    hotel_id, true_rating = author.test_rating[0]
                    hybrid_pred = self.recommender.predict(author_id,hotel_id)
                    hybrid_truths.append(true_rating)
                    hybrid_preds.append(hybrid_pred)
                    cluster_errors[training_size].append(abs(true_rating - hybrid_pred))

                # predictions of the authors within the selected state, to check how bias is the recommender to hotels 
                predictions = self.recommender.predict_selected_hotels_for_author(author_id, self.hotels_ids) #[(hotel_id, pred) ...]
                top_5 = heapq.nlargest(5, predictions, key=lambda x: x[1])
                for hotel_id, _ in top_5:
                    top_5_hotels_scores[hotel_id] += 1
        
        self.__print_basic_info()
        self.__get_author_training_size_distribution(training_sizes)
        self.__get_prediction_top_hotels_distribution(top_5_hotels_scores)
        self.__get_accuracy_per_authors_cluster_by_training_size(cluster_errors)
        self.__evaluate(hybrid_truths, hybrid_preds)
        self.pdf.close()

    def __print_basic_info(self):
        """
        First page of the pdf generated.
        Regions' information and selected states
        """
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.axis('off')
        
        regions_str = ", ".join(self.regions)
        states_str = ", ".join(self.states)
        
        text = "\n".join([
            "Statistics results of Hybrid Recommender performance",
            f"Authors from region(s): {regions_str}",
            f"Hotels from USA - state(s): {states_str}"
        ])
        
        ax.text(0.01, 0.99, text, va='top', fontsize=10)
        self.pdf.savefig(fig)
        plt.close()

    def __get_author_training_size_distribution(self, training_sizes: list[int]):
        """
        Distribution of the authors by their training data size. 
        """
        plt.hist(training_sizes, bins=10, color='purple', edgecolor='black')
        plt.title("Distribution of authors by training size")
        plt.xlabel("Size of training list")
        plt.ylabel("Number of authors")
        self.pdf.savefig()
        plt.close()

    def __get_prediction_top_hotels_distribution(self, score: dict[int, int]):
        """
        1. Sort the hotels (score dictionary key) by the amount of time it appeared in top 5 (values)
        2. Ignore hotels with less than 2%, those will be summed to small_total and grouped in Other.
        3. Create a pie chart with the hotels that appeared in more than 2% of the authors.
        """
        total = sum(score.values())
        if total == 0:
            return  
        
        items = sorted(score.items(), key=lambda x: x[1], reverse=True) #sorted by count 
        large = [(hotel_id, count) for hotel_id, count in items if count / total >= 0.02]
        small_total = sum(count for _, count in items if count / total < 0.02)
        
        labels = [f"Hotel {hotel_id}" for hotel_id, _ in large]
        sizes  = [count for _, count in large]

        if small_total > 0:
            labels.append("Others")
            sizes.append(small_total)

        fig, ax = plt.subplots(figsize=(10, 6))
        wedges, texts, autotexts = ax.pie(
            sizes,
            labels=None,
            autopct="%1.1f%%",
            startangle=90,
            colors=plt.cm.Set3.colors[: len(sizes)]
        )
        ax.axis("equal")  # keep it a circle

        ax.legend(
            wedges,
            labels,
            title="Hotel ids",
            loc="center left",
            bbox_to_anchor=(0.85, 0.5),
            fontsize="small"
        )

        self.pdf.savefig(fig)
        plt.close(fig)


    def __get_accuracy_per_authors_cluster_by_training_size(self, cluster_errors: dict[int, list[float]]):
        """
        Classify the errors of (true rating - prediction)  by training data size so we can observe how important 
        was the amount of training data to lower the error -> better generalization. In canse of the contrary, 
        it would be a case of overfitting.
        """
        sizes_sorted = sorted(cluster_errors.keys())
        data_sorted = [cluster_errors[size] for size in sizes_sorted]
        plt.boxplot(data_sorted, labels=sizes_sorted)
        plt.grid(True)
        plt.title("Prediction error vs. author training size")
        plt.xlabel("Training size")
        plt.ylabel("Absolute prediction error")
        self.pdf.savefig()
        plt.close()

    def __evaluate(self, truths: list[float],
                 hybrid_preds: list[float]):
        """
        Calculate Error Metrics MAE (Mean Absolute Error) and RMSE (Root Mean Squared Error)
        MAE: Computes the average absolute difference between predicted and true ratings. It is 
        less sensitive to outliers, so it gives a general sense of prediction accuracy.
        RMSE: Takes the square root of the average squared differences. Penalizes larger errors 
        more heavily than MAE, making it sensitive to outliers or large mistakes. 
        """
        hybrid_mae = mean_absolute_error(np.array(truths), np.array(hybrid_preds))
        hybrid_rmse = sqrt(mean_squared_error(np.array(truths), np.array(hybrid_preds)))
        
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.axis('off')
        text = "\n".join([
            "Evaluation Results:",
            "\n",
            "Hybrid:",
            f"MAE = {hybrid_mae:.3f}",
            f"RMSE = {hybrid_rmse:.3f}"
        ])
        ax.text(0.01, 0.99, text, va='top', fontsize=10)
        self.pdf.savefig(fig)
        plt.close()
