import streamlit as st
from Recommender import get_recommendations_movie, get_recommendations_show

st.title('Movie Recommender')

# Input for movie title
movie_title = st.text_input('Enter a movie title:')

# Manage button state with session_state
if 'button_clicked' not in st.session_state:
    st.session_state.button_clicked = False

# Button click
if st.button('Get Movie Recommendations', key="movie_recommendation_button"):
    st.session_state.button_clicked = True

# Run recommendation logic when the button is clicked
if st.session_state.button_clicked and movie_title:
    get_recommendations_movie(movie_title)
else:
    st.write('Please enter a movie title.')



# Define input for TV show name
st.title('TV Show Recommender')

# Input for TV show title
show_title = st.text_input('Enter a TV show title:')

# Manage button state with session_state
if 'show_button_clicked' not in st.session_state:
    st.session_state.button_clicked = False

# Button click
if st.button('Get Show Recommendations', key="show_recommendation_button"):
    st.session_state.button_clicked = True

# Run recommendation logic when the button is clicked
if st.session_state.button_clicked and show_title:
    get_recommendations_show(show_title)
else:
    st.write('Please enter a TV show title.')


