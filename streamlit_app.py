import streamlit as st
import pandas as pd
import numpy as np
import PIL.Image
import snowflake.connector
import pydeck as pdk
import pickle

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

    dow_mapping={'Monday':0,'Tuesday':1,'Wednesday':2,'Thursday':3,'Friday':4,'Saturday':5,'Sunday':6}
    dow_reverse_mapping = {v: k for k, v in dow_mapping.items()}
    #dow_labels = list(dow_mapping.keys())
    dow_labels = [dow_reverse_mapping[i] for i in sorted(dow_reverse_mapping.keys())]

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

    def get_dayOfWeek():
      dayOfWeek = st.selectbox('Select a day of week', dow_labels)
      return dayOfWeek

    def get_menuItemName(menuItemName):
      # show only the neighbourhoods in the selected neighbourhood group
      neighbourhoods = df[df['MENU_ITEM_NAME'] == min_mapping[DAY_OF_WEEK]]['MENU_ITEM_NAME'].unique()

      menuItemName = st.selectbox('Select a neighbourhood for the neighbourhood group', n_mapping)
      return menuItemName

    # Define the user input fields
    dow_input = get_dayOfWeek()
    min_input = get_menuItemName(dow_input)
  

    # Map user inputs to integer encoding
    dow_int = dow_mapping[dow_input]
    min_int = min_mapping[min_input]


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




    

with tab3:
  st.write()

with tab4:
  st.write()

with tab5:
  st.write()
