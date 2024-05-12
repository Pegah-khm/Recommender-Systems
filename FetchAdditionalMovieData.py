import pandas as pd
import requests

# Defining a function to fetch movie details
def get_movie_details(title, api_key='64766f6e'):
    url = f"http://www.omdbapi.com/?t={title}&apikey={api_key}"
    print(url)
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
    }
    try:
        response = requests.get(url, headers=headers)

        if response.status_code == 200:
            movie_data = response.json()
            # Extracting Rotten Tomatoes rating from the 'Ratings' array
            rotten_tomatoes_rating = 'Unknown'  # Default if not found
            for rating in movie_data.get('Ratings', []):
                if rating['Source'] == 'Rotten Tomatoes':
                    rotten_tomatoes_rating = rating['Value']
            return {
                'Director': movie_data.get('Director', 'Unknown'),
                'Actors': movie_data.get('Actors', 'Unknown'),
                'Rotten Tomatoes': rotten_tomatoes_rating,
                'imdbRating': movie_data.get('imdbRating', 'Unknown'),
                'imdbVotes': movie_data.get('imdbVotes', 'Unknown').replace(',', ''),
                'Country': movie_data.get('Country', 'Unknown')
            }
        else:
            return {'Director': 'Unknown', 'Actors': 'Unknown', 'Rotten Tomatoes': 'Unknown', 'imdbRating': 'Unknown',
                    'imdbVotes': 'Unknown', 'Country': 'Unknown'}
    except requests.RequestException:
        return {'Director': 'Unknown', 'Actors': 'Unknown', 'Rotten Tomatoes': 'Unknown', 'imdbRating': 'Unknown',
                'imdbVotes': 'Unknown', 'Country': 'Unknown'}


# Loading movies.csv into a DataFrame
movies_df = pd.read_csv('ml-25m/movies_with_suffix.csv')

# Initialising lists to store the new column data
directors = []
actors = []
rottenTomatoes = []
imdbRating = []
imdbVotes = []
countries = []

# Iterating over the movie titles and fetch their details
for title in movies_df['title']:
    title_adjusted = title.split(' (')[0]  # Assuming title format "Movie Title (Year)"
    details = get_movie_details(title_adjusted)
    directors.append(details['Director'])
    actors.append(details['Actors'])
    rottenTomatoes.append(details['Rotten Tomatoes'])
    imdbRating.append(details['imdbRating'])
    imdbVotes.append(details['imdbVotes'])
    countries.append(details['Country'])

# Adding the new data as columns to the DataFrame
movies_df['Director'] = directors
movies_df['Actors'] = actors
movies_df['Rotten Tomatoes'] = rottenTomatoes
movies_df['imdbRatings'] = imdbRating
movies_df['imdbVotes'] = imdbVotes
movies_df['Country'] = countries

# Saving the updated DataFrame to a new CSV file
movies_df.to_csv('ml-25m/refined_title_movies.csv', index=False)
print(movies_df)
