import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import svds
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from keras.models import Model
from keras.layers import Input, Dense, Concatenate, Embedding, Flatten, Dropout, Lambda
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.metrics import MeanAbsoluteError, RootMeanSquaredError
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error


# Load data
ratings = pd.read_csv('filtered_ratings.csv')
movies = pd.read_csv('filtered_movies.csv')

# Merge datasets
merged_df = pd.merge(ratings, movies, on='movieId', how='left')

# Creating a sparse matrix
user_ids = merged_df['userId'].values - 1
movie_ids = merged_df['movieId'].values - 1
ratings = merged_df['rating'].values

# Create the sparse matrix
interaction_matrix = coo_matrix((ratings, (user_ids, movie_ids)), shape=(user_ids.max() + 1, movie_ids.max() + 1))


# Perform SVD
k = 100  # Number of latent factors
U, sigma, Vt = svds(interaction_matrix, k=k)

# Encode categorical data
genre_encoder = LabelEncoder()
director_encoder = LabelEncoder()
merged_df['Genre_encoded'] = genre_encoder.fit_transform(merged_df['genres'])
merged_df['Director_encoded'] = director_encoder.fit_transform(merged_df['Director'])

# Normalaize imdbVotes
merged_df['imdbVotes'] = merged_df['imdbVotes'] / merged_df['imdbVotes'].max()

assert merged_df[['Rotten Tomatoes', 'imdbRatings', 'imdbVotes']].notna().all().all(), "There are missing values in the data"

rotten_tomatoes_input = Input(shape=(1,), name='rotten_tomatoes')
imdb_rating_input = Input(shape=(1,), name='imdb_rating')
imdb_voting_input = Input(shape=(1,), name='imdb_voting')


# Filter merged_df to ensure it only includes the user_ids and movie_ids that were in the sparse matrix
valid_user_ids = np.unique(user_ids)  # This extracts the unique user_ids that have ratings
merged_df = merged_df[merged_df['userId'].isin(valid_user_ids + 1)]  # +1 to match the original IDs

# Prepare inputs for training, ensure to extract data only for valid users
genre_data = merged_df['Genre_encoded'].values.reshape(-1, 1)
director_data = merged_df['Director_encoded'].values.reshape(-1, 1)
ratings = merged_df['rating'].values


# Align user_features using an index that matches filtered merged_df
user_features = U[merged_df['userId'].values - 1, :]
# movie_features = Vt[merged_df['movieId_mapped'].values, :]

rotten_tomatoes_data = merged_df['Rotten Tomatoes'].values.reshape(-1, 1)
imdb_rating_data = merged_df['imdbRatings'].values.reshape(-1, 1)
imdb_voting_data = merged_df['imdbVotes'].values.reshape(-1, 1)


X_train, X_test, genre_train, genre_test, director_train, director_test, rotten_tomatoes_train, rotten_tomatoes_test, imdb_rating_train, imdb_rating_test, imdb_voting_train, imdb_voting_test, y_train, y_test = train_test_split(
    user_features, genre_data, director_data, rotten_tomatoes_data, imdb_rating_data, imdb_voting_data, ratings, test_size=0.2, random_state=42
)


# Model Inputs
user_features_input = Input(shape=(k,), name='user_features')
# movie_features_input = Input(shape=(k,), name='movie_features')
genre_input = Input(shape=(1,), name='genre_input')
director_input = Input(shape=(1,), name='director_input')
genre_embedding = Embedding(input_dim=merged_df['Genre_encoded'].nunique(), output_dim=15)(genre_input)
director_embedding = Embedding(input_dim=merged_df['Director_encoded'].nunique(), output_dim=15)(director_input)
genre_vector = Flatten()(genre_embedding)
director_vector = Flatten()(director_embedding)
concatenated_features = Concatenate()([user_features_input, genre_vector, director_vector,
                                       rotten_tomatoes_input, imdb_rating_input, imdb_voting_input
                                       ])

assert len(user_features) == len(genre_data) == len(director_data) == len(ratings), "Data arrays must all be the same length"

print("\n\nStarting model training...")

print("--------------------------------------------------")


# Model architecture
x = Dense(256, activation='relu')(concatenated_features)
x = Dropout(0.2)(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.2)(x)
x = Dense(64, activation='relu')(x)
output = Dense(1, activation='linear')(x)

model = Model(inputs=[user_features_input, genre_input, director_input, rotten_tomatoes_input, imdb_rating_input, imdb_voting_input], outputs=output)
model.compile(optimizer=Adam(learning_rate=0.0003), loss='mean_squared_error', metrics=['MAE', RootMeanSquaredError(name='RMSE')])

# Setup checkpoints and early stopping
checkpoint = ModelCheckpoint('Results/Models/Hybrid_Model_02_ML25M_Run01.keras', monitor='val_loss', save_best_only=True, verbose=1)
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)

# Train the model
history = model.fit(
    [X_train, genre_train, director_train, rotten_tomatoes_train, imdb_rating_train, imdb_voting_train], y_train,
    validation_data=([X_test, genre_test, director_test, rotten_tomatoes_test, imdb_rating_test, imdb_voting_test], y_test),
    epochs=100,
    batch_size=64,
    callbacks=[checkpoint, early_stopping]
)


# Evaluate the model
test_loss = model.evaluate([X_test, genre_test, director_test, rotten_tomatoes_test, imdb_rating_test, imdb_voting_test], y_test)
predicted_ratings = model.predict([X_test, genre_test, director_test, rotten_tomatoes_test, imdb_rating_test, imdb_voting_test]).flatten()

 # Calculate MAE and RMSE
mae = mean_absolute_error(y_test, predicted_ratings)
rmse = np.sqrt(mean_squared_error(y_test, predicted_ratings))
print("------------------------------------------------------------------------------------")
print("Final results:")
print(f'\nTest Loss: {test_loss}\n')
print(f'Mean Absolute Error (MAE): {mae:.4f}\n')
print(f'Root Mean Square Error (RMSE): {rmse:.4f}')
print("------------------------------------------------------------------------------------")


# Plotting training and validation loss
plt.figure(figsize=(7, 5))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss Over Epochs')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.show()

# Plotting training and validation MAE
plt.figure(figsize=(7, 5))
plt.plot(history.history['MAE'], label='Train MAE')
plt.plot(history.history['val_MAE'], label='Validation MAE')
plt.title('Mean Absolute Error (MAE) Over Epochs')
plt.ylabel('Mean Absolute Error')
plt.xlabel('Epoch')
plt.legend()
plt.show()

# Plotting training and validation RMSE
plt.figure(figsize=(7, 5))
plt.plot(history.history['RMSE'], label='Train RMSE')
plt.plot(history.history['val_RMSE'], label='Validation RMSE')
plt.title('RMSE Over Epochs')
plt.ylabel('Root Mean Square Error')
plt.xlabel('Epoch')
plt.legend()
plt.show()


# Create a mapping from userId to a sequential index starting at 0
user_id_mapping = {uid: idx for idx, uid in enumerate(sorted(merged_df['userId'].unique()))}


# Before starting the loop, ensure num_movies is defined as the total unique movie counts.
num_movies = merged_df['movieId'].nunique()


# Create an empty list to hold top 10 recommendations for all users
top_10_recommendations = []

# Get unique user IDs
unique_users = merged_df['userId'].unique()

print("Top 10 Recommendations for first 10 users:")

sampled_user_ids = unique_users[:10]


for user_id in sampled_user_ids:

    print(f"\nProcessing recommendations for User ID: {user_id}\n")
    rated_movies = merged_df[merged_df['userId'] == user_id]['movieId'].unique()

    # Get IDs of movies not rated by the user
    unrated_movie_ids = movies[~movies['movieId'].isin(rated_movies)]['movieId']


    # Fetch details of unrated movies from merged_df
    unrated_movies = merged_df[merged_df['movieId'].isin(unrated_movie_ids)]

    print(f"\nUnrated movies for user {user_id}:\n {unrated_movies.head()}\n")

    if unrated_movies.empty:
        print("No unrated movies available for recommendations.")
        top_10_recommendations.append((user_id, []))
        continue


    user_features = U[user_id - 1].reshape(1, -1)  # User features for the model


    # Collect all features needed for the unrated movies
    genre_data = unrated_movies['Genre_encoded'].values.reshape(-1, 1)
    director_data = unrated_movies['Director_encoded'].values.reshape(-1, 1)
    rotten_tomatoes_data = unrated_movies['Rotten Tomatoes'].values.reshape(-1, 1)
    imdb_rating_data = unrated_movies['imdbRatings'].values.reshape(-1, 1)
    imdb_voting_data = unrated_movies['imdbVotes'].values.reshape(-1, 1)

    # Make predictions for the unrated movies
    predictions = model.predict([
            np.repeat(user_features, len(unrated_movies), axis=0),
            genre_data,
            director_data,
            rotten_tomatoes_data,
            imdb_rating_data,
            imdb_voting_data
        ]).flatten()

    print(f'\n Prediction shape: {predictions.shape}')

    print(f"Predictions for User {user_id}:\n {predictions}")

    plt.figure(figsize=(10, 5))
    plt.hist(predictions, bins=30, alpha=0.7)
    plt.title(f'Distribution of Predictions for User {user_id}')
    plt.xlabel('Predicted Ratings')
    plt.ylabel('Frequency')
    plt.show()


    unrated_movies['Predicted_Rating'] = predictions

    unrated_movies_unique = unrated_movies.drop_duplicates(subset='movieId')

    unrated_movies_sorted = unrated_movies_unique.sort_values(by='Predicted_Rating', ascending=False)


    top_rated_movies = unrated_movies_sorted.head(10)


    top_movies_ratings = [(row['movieId'], row['title'], row['Predicted_Rating']) for _, row in top_rated_movies.iterrows()]

    top_10_recommendations.append((user_id, top_movies_ratings))



for user_id, recommendations in top_10_recommendations:
    print(f"\nTop 10 Recommendations for User {user_id}:\n")
    for movie_id, title, rating in recommendations:
        print(f"{movie_id:6d} {title:50} {rating:.1f}")







