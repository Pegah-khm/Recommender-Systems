import pandas as pd

# Loading datasets
ratings = pd.read_csv("Original Data/ratings.csv")
movies = pd.read_csv("Original Data/featured-movies.csv")

print(f"\nRatings Shape: {ratings.shape}")
print(f"\nMovies Shape: {movies.shape}")

print(movies.isnull().sum())

for column in movies.columns:
    unknown_counts = (movies[column] == 'Unknown').sum()

print(f'\nUnknown counts: {unknown_counts}')

updated_movies = movies[movies['Director'] != 'unknown']

print(f'\nShape of updated_movies: {updated_movies.shape}')

updated_movies = updated_movies[~updated_movies['Director'].str.lower().str.strip().eq('unknown')]

unknown_counts = {col: (updated_movies[col].str.lower() == 'unknown').sum() for col in updated_movies.columns if updated_movies[col].dtype == 'object'}

print("\nNumber of 'unknown' entries per column:")
for column, count in unknown_counts.items():
    print(f"{column}: {count}")

updated_movies = updated_movies.drop('Country', axis=1)
updated_movies = updated_movies.drop('Actors', axis=1)

unknown_imdbRatings = updated_movies[updated_movies['imdbRatings'].str.lower() == 'unknown']
unknown_imdbVotes = updated_movies[updated_movies['imdbVotes'].str.lower() == 'unknown']

updated_movies = updated_movies[updated_movies['imdbRatings'].str.lower() != 'unknown']

updated_movies['Rotten Tomatoes'] = movies['Rotten Tomatoes'].replace('Unknown', None)
condition = updated_movies['Rotten Tomatoes'].isnull() & updated_movies['imdbRatings'].notnull()
filled_count = condition.sum()
print(f'\nFilled Count: {filled_count}')

# Convert and fill 'Rotten Tomatoes' to same scale as IMDb Ratings
updated_movies.loc[condition, 'Rotten Tomatoes'] = updated_movies.loc[condition, 'imdbRatings'].apply(lambda x: str(float(x) * 10) + '%')

new_unknown_counts = {col: (updated_movies[col].str.lower() == 'unknown').sum() for col in updated_movies.columns if updated_movies[col].dtype == 'object'}
print("\nNumber of 'unknown' entries per column:")
for column, count in new_unknown_counts.items():
    print(f"{column}: {count}")

# Convert movieId in updated_movies to a set for faster lookup
valid_movie_ids = set(updated_movies['movieId'])

# Filter ratings to only include valid movieIds
cleaned_ratings = ratings[ratings['movieId'].isin(valid_movie_ids)]

print("Original ratings count:", len(ratings))
print("Cleaned ratings count:", len(cleaned_ratings))

cleaned_ratings.to_csv("Cleaned Data/cleaned_ratings.csv", index=False)
updated_movies.to_csv("Cleaned Data/cleaned_movies.csv", index=False)
