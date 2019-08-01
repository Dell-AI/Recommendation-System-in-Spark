# Recommendation System in Spark

These notebooks walk through how to use Machine Learning (ML) and Deep Learning (DL) models to create a recommendation system for users and items in Apache Spark. The MovieLens 1M and 20M reviews datasets (https://grouplens.org/datasets/movielens/) are used for explicit user-item reviews, and movie metadata is collected from The Open Movie Database (http://www.omdbapi.com/). The notebooks show how to collect and process the movie metdata and user-item reviews, train Collaborative Filtering (CF) and Content-Based Filtering (CBF) models, and make recommendations for new and existing users. 

Notebooks ending in '20m' are using the 20M ratings dataset and all other notebooks use the 1M ratings dataset. The 1M reviews dataset contains 1 million movie ratings made by 4,000 users on 6,000 movies. The 20M ratings dataset contains 20 million reviews made by 138,000 users on 27,000 movies.


## Data Preparation

The Movie_Metadata_Collection notebook collects the movie metadata from The Open Movie Database, including IMDb rating, number of IMDb votes, Metacritic score, runtime, MPAA rating, genres, actors, and directors. The fields containing character values or lists of character values are binarized and this data is saved for later use.

The Movie_Metadata_Preparation_and_User_Feature_Creation notebook combines the user-item ratings and the binarized movie metadata to create a dataset containing user profiles. The user-item ratings are spread to create a sparse user-item matrix with values being the explicity rating, or zero if no rating. This sparse matrix is then combined with the binarized movie metadata to create a dataset containing the users' average ratings for each binarized column. The binarized actors and directors columns are reduced to the top 250 and 50 respectively, based on the total count of user ratings for each column.

The movie metadata dataset also has its binarized actors and directors columns reduced to match the user average ratings dataset so that the datasets can be merged and used for content based filtering.

For the movies in the 20M ratings set, the binarization of character based fields happens in the Movie_Metadata_Preparation_and_User_Feature_Creation_20m notebook because it is unnecessary and unreasonable to save this data to a file.


## Model Building

#### Collaborative Filtering Models
The model training and testing for the CF models is completed in the Collaborative_Filtering_ML_ALS_vs_BigDL_NCF notebook. The Alternating Lease Squares (ALS) model is part of the pyspark.ml library and performs an iterative matrix factorization to find a best set of latent factors for users and for items. Matrix factorization has been a popular method for recommendation systems since the Netflix Prize competition, and ALS has been a popular model for this. The Neural Collaborative Filtering (NCF) model is part of Intel's Analytics Zoo library, using the BigDL optimizer for its backend, and this uses both matrix factorization and a neural network to create recommendations. NCF is more recent and its goal was to create a better recommendation model by augmenting matrix factorization with current techniques. 

The data necessary for these models is just the user-item ratings dataset, and because formatting for use in the models is simple, both models can be done in the same notebook. The ALS model takes input from a Spark dataframe with three columns - userId, itemId, label (in this case rating). The NCF model takes the same three inputs, but requires them to in the RDD Sample format. RDD Samples are a BigDL type that are used by the BigDL optimizer to imrpove parallel performance.

When the two models are trained and tested, ALS trains and predicts significantly faster than NCF (which is expected since NCF is neural networks and not just matrix factorization), but when comparing the mean absolute error (MAE) and root mean square error (RMSE) score there is minimal difference between the two models. However, neither model's hyperparameters were tuned so this may just be a result of the default models. One interesting thing to note is that in the 20M dataset, the NCF results appear to give better recommendations, even though both models have very similar test metrics scores.

#### Content Based Filtering Models
The model training and testing for the CBF models are in the Content_Based_Filtering_GBT_and_Random_Forest and Content_Based_Filtering_wide_and_deep_neural_network notebooks. Both notebooks use the user profile data and movie profile data created in the Movie_Metadata_Preparation_and_User_Feature_Creation notebook. The Gradient Boosted Trees (GBT) and Random Forest (RF) models are part of the pyspark.ml library and the Wide and Deep model is part of the Analytics Zoo library. These CBF models are meant to work with their respective ML and DL collaborative filtering models to improve recommendations.

For the CBF models the data used is a combination of ratings, user profiles, and item profiles. The ratings are what they models attempt to predict, and the user and item profiles are the predictors. To create this dataset, the user and item profiles are joined onto the ratings dataset, so each rated movie has the user profile for the user who rated the movie and the item profile for the movie that was rated.

The GBT and RF models take data input from a Spark dataframe with two columns, one with the label to be predicted and one with the vectorized form of the features. Once the data was formatted GBTs and RF models were trained and tested using a few different numbers of trees and tree depths, and a final model was selected to be used. The models were trained for both classification and regression, and in the end a GBT regression model was chosen. However, the hyparamters were not well tuned and the MAE score on test data was poor. This model is saved out for later use.

The Wide and Deep model takes input from a RDD of Sample. The RDD of sample contains rows with all feature and label columns, but with additional information about which columns are for which part of the model (the column_info). This is where feature engineering could be important, but for the sake of this example it was kept simple and columns were mostly grouped by type. The model was trained as a classification model, but had minimal hyperparameter tuning. The model accuracy score was similar to the classifical tree-based models. This model is saved out for later use.


## Recommendation Comparison

The Mini_Movie_Recommender_ALS_with_Trees and Mini_Movie_Recommender_NCF_with_WnD notebooks go through the process of creating a brand new user and making recommendations for that user. They both go through the process of adding the user's favorite movies to the full ratings dataset, all with 5 star ratings, and creating the set of unrated movies for that user. Then the new full ratings dataset is used to retrain the ALS and NCF models (which must be done since a new user was added).

The user's unrated movies set go through the data preparation process performed in the Movie_Metadata_Preparation_and_User_Feature_Creation notebook to get the user's profile.

The user's unrated movies are then formatted for both the ALS and NCF models, and recommendations are made for the user. Next, a larger set of recommendations from the CF models is made and these recommended items are combined with the user's profile and item profiles to create the dataset necessary for the CBF models. This dataset is then formatted appropriately for both the GBT and Wide and Deep models, and both the models predict ratings. The items are sorted by the predicted rating and the top set are selected as recommendations.

In the notebooks using the 1M ratings data, the recommendations are sporadic and tend to pick the same set of movies for most users. In the notebooks using the 20M ratings data, the recommendations are much better and seem reasonable. However, the NCF with Wide and Deep recommendations appear to give more user specific and better ratings.


# Running the notebooks

The following parameters are tested to work:

| Parameter | Value for 1 M | Value for 20 M |
|---|---|---|
| Num executors | 16 | 16 |
| Num cores | 16 | 16 |
| Executor memory | 128gb | 256gb |
| Driver memory | 256gb | 512gb |
