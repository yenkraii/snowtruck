import streamlit as st
import pandas as pd
import numpy as np
import PIL.Image
import snowflake.connector
import pydeck as pdk
import joblib
from joblib import load
import pickle
from sklearn import preprocessing

st.set_page_config(page_title="SnowTruck", page_icon=":truck:")

#st.image(snowtruck_logo)
fp = open("images/Snowtruck_ProductBox.jpeg","rb")
snowtruck_logo = PIL.Image.open(fp)

st.image(snowtruck_logo,width=64)
st.title("SnowTruck:minibus:")

# connects to the snowflake account 
# if need to use the data there
conn = snowflake.connector.connect(**st.secrets["snowflake"])

# tabs
tab1,tab2,tab3,tab4,tab5 = st.tabs(["tab1","tab2","tab3","tab4","tab5"])

with tab1:
  st.write()

with tab2:

  # Define the app title and favicon
    st.title('What are the top-selling food items by menu?') 
    st.markdown("This tab allows you to make predictions on the price of a listing based on the neighbourhood and room type. The model used is a Random Forest Regressor trained on the Airbnb Singapore listings dataset.")
    st.write('Choose a neighborhood group, neighborhood, and room type to get the predicted average price.')
    
    with open('xgb_xinle.pkl', 'rb') as file:
        xgb = pickle.load(file)

    # Load the cleaned and transformed dataset
    df = pd.read_csv('df_xinle.csv')
    price = df[['TOTAL_QUANTITY']] # extract price column from listings_new2.csv

    dowmappings={'Monday':0,'Tuesday':1,'Wednesday':2,'Thursday':3,'Friday':4,'Saturday':5,'Sunday':6}
    dow_reverse_mapping = {v: k for k, v in dowmappings.items()}
    dow_labels = list(dowmappings.keys())
    
    # # Define the user input functions
    # ng_mapping = {'Central Region': 0, 'East Region': 1, 'North Region': 2, 'North-East Region': 3, 'West Region': 4}
    # ng_reverse_mapping = {v: k for k, v in ng_mapping.items()}
    # ng_labels = list(ng_mapping.keys())
    
    min_mapping = {'Ice Tea': 0,
     'Fish Burrito': 1,
     'Lean Beef Tibs': 2,
     'Bottled Soda': 3,
     'Bottled Water': 4,
     'Mango Sticky Rice': 5,
     'Rack of Pork Ribs': 6,
     'Buffalo Mac & Cheese': 7,
     'Tonkotsu Ramen': 8,
     'Two Scoop Bowl': 9,
     'Waffle Cone': 10,
     'Mothers Favorite': 11,
     'Lean Chicken Tikka Masala': 12,
     'Seitan Buffalo Wings': 13,
     'The Salad of All Salads': 14,
     'Lean Chicken Tibs': 15,
     'The King Combo': 16,
     'Coney Dog': 17,
     'Two Taco Combo Plate': 18,
     'Wonton Soup': 19,
     'Spicy Miso Vegetable Ramen': 20,
     'Sugar Cone': 21,
     'Pulled Pork Sandwich': 22,
     'Standard Mac & Cheese': 23,
     'Veggie Combo': 24,
     'Tandoori Mixed Grill': 25,
     'Italian': 26,
     'Crepe Suzette': 27,
     'Combo Fried Rice': 28,
     'Lean Burrito Bowl': 29,
     'Greek Salad': 30,
     'Two Meat Plate': 31,
     'The Classic': 32,
     'Spring Mix Salad': 33,
     'Lobster Mac & Cheese': 34,
     'The Ranch': 35,
     'Miss Piggie': 36,
     'Ice Cream Sandwich': 37,
     'Three Meat Plate': 38,
     'Three Taco Combo Plate': 39,
     'Fried Pickles': 40,
     'Hot Ham & Cheese': 41,
     'Veggie Burger': 42,
     'Combo Lo Mein': 43,
     'The Original': 44,
     'Creamy Chicken Ramen': 45,
     'Lemonade': 46,
     'Popsicle': 47,
     'Veggie Taco Bowl': 48,
     'Pastrami': 49,
     'Chicago Dog': 50,
     'The Kitchen Sink': 51,
     'Gyro Plate': 52,
     'Chicken Burrito': 53,
     'New York Dog': 54,
     'Chicken Pot Pie Crepe': 55,
     'Combination Curry': 56,
     'Breakfast Crepe': 57}

    mt_mapping = {'BBQ': 0,
     'Tacos': 1,
     'Ethiopian': 2,
     'Poutine': 3,
     'Gyros': 4,
     'Chinese': 5,
     'Ice Cream': 6,
     'Grilled Cheese': 7,
     'Mac & Cheese': 8,
     'Ramen': 9,
     'Indian': 10,
     'Vegetarian': 11,
     'Hot Dogs': 12,
     'Crepes': 13,
     'Sandwiches': 14}

    ic_mapping = {'Beverage': 0, 'Dessert': 1, 'Main': 2, 'Snack': 3}

    isc_mapping= {'Cold Option': 0, 'Warm Option': 1, 'Hot Option': 2}
    
    # room_type_mapping = {'Private room': 0, 'Entire home/apt': 1}
    # room_type_reverse_mapping = {}
    # for k, v in room_type_mapping.items():
    #     room_type_reverse_mapping[v] = k
    #     #print("Added room type %s with key %s" % (v, k))
    
    # ng_labels = [ng_reverse_mapping[i] for i in sorted(ng_reverse_mapping.keys())]
    # #n_labels = [n_reverse_mapping[i] for i in sorted(n_reverse_mapping.keys())]

    # rt_labels = [room_type_reverse_mapping[i] for i in sorted(room_type_reverse_mapping.keys())]

    
    def get_dayOfWeek():
        dayOfWeek = st.selectbox('Select a day of week', dow_labels)
        return dayOfWeek


    # def get_neighbourhood(neighbourhood_group):
    #     # show only the neighbourhoods in the selected neighbourhood group
    #     neighbourhoods = df[df['neighbourhood_group'] == ng_mapping[neighbourhood_group]]['neighbourhood'].unique()
    #     # map n_mapping to a categorical value based off of the selected neighbourhood group using the n_mapping dictionary key value pairs
    #     #n_mapping = {i: neighbourhood for i, neighbourhood in enumerate(neighbourhoods)}
    #     #n_mapping = {neighbourhood: i for i, neighbourhood in enumerate(neighbourhoods)}
    #     # show the neighbourhoods in categorical order based off of the selected neighbourhood group

    #     neighbourhood = st.selectbox('Select a neighbourhood for the neighbourhood group', n_mapping)
    #     #neighbourhood = st.selectbox('Select a neighbourhood for the neighbourhood group', list(n_mapping.keys()), format_func=lambda x: n_mapping[x])
    #     #return n_mapping[neighbourhood]
    #     return neighbourhood
    
    # def get_room_type():
    #     #room_type = st.selectbox('Select a room type', df['room_type'].unique())
    #     room_type = st.selectbox('Select a room type', rt_labels)
    #     return room_type


    # # Define the user input fields
    # ng_input = get_neighbourhood_group()
    # n_input = get_neighbourhood(ng_input)
    # room_type_input = get_room_type()

    # # Map user inputs to integer encoding
    # ng_int = ng_mapping[ng_input]
    # n_int = n_mapping[n_input]
    # rt_int = room_type_mapping[room_type_input]
    
    # # Display the prediction
    # if st.button('Predict Price'):
        
    #     # Make the prediction   
    #     input_data = [[minimum_nights, number_of_reviews, calculated_host_listings_count, availability_365, ng_int, n_int, rt_int, reviews_per_month, is_zero]]
    #     input_df = pd.DataFrame(input_data, columns=['minimum_nights', 'number_of_reviews', 'calculated_host_listings_count','availability_365','neighbourhood_group', 'neighbourhood', 'room_type', 'reviews_per_month', 'is_zero'])
    #     prediction = rf.predict(input_df)   
    #     # convert output data and columns, including price, to a dataframe avoiding TypeError: type numpy.ndarray doesn't define __round__ method
    #     output_data = [minimum_nights, number_of_reviews, calculated_host_listings_count, availability_365, ng_input, n_input, rt_int, reviews_per_month, is_zero, prediction[0]]

    
    #     output_df = pd.DataFrame([output_data], columns=['minimum_nights', 'number_of_reviews', 'calculated_host_listings_count','availability_365','neighbourhood_group', 'neighbourhood', 'room_type', 'reviews_per_month', 'is_zero', 'predicted_price'])

    #     # Make the prediction   
    #     # show prediction on price in dollars and cents using the price column 
    #     input_data = [[minimum_nights, number_of_reviews, calculated_host_listings_count, availability_365, ng_int, n_int, rt_int, reviews_per_month, is_zero]]

    #     predicted_price = rf.predict(input_df)[0]
    #     st.write('The predicted average price is ${:.2f}.'.format(predicted_price))
    #     st.dataframe(output_df)


with tab3:
  st.write()

with tab4:
  st.write()

with tab5:
  st.write()
