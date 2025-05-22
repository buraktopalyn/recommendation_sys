# Şarkı Öneri Sistemi (Song Recommendation System)

This project is a music recommendation system built with Streamlit that suggests songs based on user preferences. The system uses machine learning techniques to analyze song features and find similar tracks to those selected by the user.

## Features

- Interactive web interface built with Streamlit
- Multi-select functionality to choose favorite songs
- Recommendation engine based on cosine similarity
- Song feature analysis (danceability, energy, tempo, etc.)
- Detailed information about recommendations

## Dataset

The system uses a dataset (`dataset.csv`) containing songs with various audio features extracted from Spotify, including:

- popularity
- danceability
- energy
- key
- loudness
- mode
- speechiness
- acousticness
- instrumentalness
- liveness
- valence
- tempo

## Installation

1. Clone this repository
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Web Interface (Streamlit App)

Run the Streamlit application:

```bash
streamlit run streamlit_app.py
```

Then:
1. Select songs you like from the dropdown menu
2. Click the "Şarkı Öner" (Recommend Songs) button
3. View your personalized song recommendations

### Command Line Script

Alternatively, you can use the command line script for quick testing:

```bash
python recommend_song.py
```

This will randomly select songs and generate recommendations based on them.

## How It Works

The recommendation system works by:

1. Standardizing numerical features of songs
2. Creating a user profile based on selected songs
3. Calculating cosine similarity between the user profile and all songs
4. Filtering out already selected songs
5. Returning the most similar songs as recommendations

## Requirements

See `requirements.txt` for a list of dependencies.

## License

This project is open source and available for educational and personal use.