import streamlit as st
import pandas as pd
import numpy as np
import PIL.Image
import snowflake.connector
import pydeck as pdk
import pickle
from PIL import Image
from sklearn import preprocessing


st.set_page_config(page_title="SnowTruck", page_icon=":truck:")

#st.image(snowtruck_logo)

snowtruck_logo = Image.open("images/Snowtruck_ProductBox.jpeg")

st.image(snowtruck_logo, width=700)
st.title("SnowTruck:minibus:")

# connects to the snowflake account 
# if need to use the data there
conn = snowflake.connector.connect(**st.secrets["snowflake"])

# tabs
tab1,tab2,tab3,tab4,tab5 = st.tabs(["tab1","tab2","tab3","tab4","tab5"])

with tab1:
  # Define the app title and favicon
  st.title('How much can you make from the TastyBytes locations?')
  st.markdown("This tab allows you to make predictions on the price of a listing based on the neighbourhood and room type. The model used is a Random Forest Regressor trained on the Airbnb Singapore listings dataset.")
  st.write('Choose a Truck Brand Name, City, Truck Location and Time Frame to get the predicted sales.')

  with open('xgb_alethea.pkl', 'rb') as file:
    xgb_gs = pickle.load(file)
    
    # Load the cleaned and transformed dataset
    df = pd.read_csv('df_alethea.csv')
    quantity = df[['TOTAL_QUANTITY']]
  
    dow_mapping={'Monday':0,'Tuesday':1,'Wednesday':2,'Thursday':3,'Friday':4,'Saturday':5,'Sunday':6}
    dow_reverse_mapping = {v: k for k, v in dow_mapping.items()}
    dow_labels = [dow_reverse_mapping[i] for i in sorted(dow_reverse_mapping.keys())]

    bn_mapping = { "Cheeky Greek": 0,
                  "Guac n' Roll": 1,
                  "Smoky BBQ": 2,
                  "Peking Truck": 3,
                  "Tasty Tibs": 4,
                  "Better Off Bread": 5,
                  "The Mega Melt": 6,
                  "Le Coin des Crêpes": 7,
                  "The Mac Shack": 8,
                  "Nani's Kitchen": 9,
                  "Plant Palace": 10,
                  "Kitakata Ramen Bar": 11,
                  "Amped Up Franks": 12,
                  "Freezing Point": 13,
                  "Revenge of the Curds": 14 }

    ct_mapping = {'San Mateo': 0, 'Seattle': 1, 'New York City': 2, 'Boston': 3, 'Denver':4}

    def get_dayOfWeek():
      DayOfWeek = st.selectbox('Select a day of week', dow_labels,key='tab1_dayOfWeekSelect')
      return DayOfWeek

    # Define the user input fields
    dowInput = get_dayOfWeek()

    # Map user inputs to integer encoding
    dowInt = dow_mapping[dowInput]



with tab2:

  # Define the app title and favicon
    st.title('What are the top-selling food items by menu?') 
    st.markdown("This tab allows you to make predictions on the price of a listing based on the neighbourhood and room type. The model used is a Random Forest Regressor trained on the Airbnb Singapore listings dataset.")
    st.write('Choose a neighborhood group, neighborhood, and room type to get the predicted average price.')
    
    with open('xgb_xinle.pkl', 'rb') as file:
        xgb_xinle = pickle.load(file)

    # Load the cleaned and transformed dataset
    df = pd.read_csv('df_xinle.csv')
    profit = df[['PROFIT']] # extract price column from df_xinle

    dowMapping={'Monday':0,'Tuesday':1,'Wednesday':2,'Thursday':3,'Friday':4,'Saturday':5,'Sunday':6}
    dowReverseMapping = {v: k for k, v in dowMapping.items()}
    dowLabels = [dowReverseMapping[i] for i in sorted(dowReverseMapping.keys())]

    mt_mapping = {'Vegetarian': 0,'Crepes': 1,'Ethiopian': 2,'Hot Dogs': 3,'Poutine': 4,
                  'Gyros': 5,'Chinese': 6,'Indian': 7,'Sandwiches': 8,'Ice Cream': 9,
                  'BBQ': 10,'Tacos': 11,'Mac & Cheese': 12,'Ramen': 13,'Grilled Cheese': 14}
  
    min_mapping = {'Veggie Burger': 0,'Seitan Buffalo Wings': 1,'Bottled Soda': 2,'Bottled Water': 3,
                   'The Salad of All Salads': 4,'Ice Tea': 5,'Chicken Pot Pie Crepe': 6,'Breakfast Crepe': 7,
                   'Crepe Suzette': 8,'Lean Chicken Tibs': 9,'Lean Beef Tibs': 10,'Veggie Combo': 11,
                   'New York Dog': 12,'Chicago Dog': 13,'Coney Dog': 14,'The Classic': 15,'The Kitchen Sink': 16,
                   'Mothers Favorite': 17,'Gyro Plate': 18,'The King Combo': 19,'Greek Salad': 20,'Combo Lo Mein': 21,
                   'Combo Fried Rice': 22,'Wonton Soup': 23,'Lean Chicken Tikka Masala': 24,
                   'Tandoori Mixed Grill': 25,'Combination Curry': 26,'Italian': 27,
                   'Pastrami': 28,'Hot Ham & Cheese': 29,'Mango Sticky Rice': 30,'Sugar Cone': 31,
                   'Waffle Cone': 32,'Popsicle': 33, 'Two Scoop Bowl': 34,'Ice Cream Sandwich': 35,
                   'Lemonade': 36,'Fried Pickles': 37,'Pulled Pork Sandwich': 38,'Three Meat Plate': 39,
                   'Spring Mix Salad': 40,'Rack of Pork Ribs': 41,'Two Meat Plate': 42,'Two Taco Combo Plate': 43,
                   'Chicken Burrito': 44,'Three Taco Combo Plate': 45,'Lean Burrito Bowl': 46,'Veggie Taco Bowl': 47,
                   'Fish Burrito': 48,'Standard Mac & Cheese': 49,'Buffalo Mac & Cheese': 50,'Lobster Mac & Cheese': 51,
                   'Spicy Miso Vegetable Ramen': 52,'Tonkotsu Ramen': 53,'Creamy Chicken Ramen': 54,
                   'The Ranch': 55,'Miss Piggie': 56,'The Original': 57}
  
    ic_mapping = {'Main': 0, 'Snack': 1, 'Beverage': 2, 'Dessert': 3}

    isc_mapping= {'Hot Option': 0, 'Cold Option': 1, 'Warm Option': 2}

    tbn_mapping = {'Plant Palace': 0,'Le Coin des Crêpes': 1,'Tasty Tibs': 2,'Amped Up Franks': 3,
                   'Revenge of the Curds': 4,'Cheeky Greek': 5,'Peking Truck': 6,"Nani's Kitchen": 7,
                   'Better Off Bread': 8,'Freezing Point': 9,'Smoky BBQ': 10,"Guac n' Roll": 11,
                   'The Mac Shack': 12,'Kitakata Ramen Bar': 13,'The Mega Melt': 14}
  
    c_mapping = {'New York City': 0, 'Seattle': 1, 'Denver': 2, 'Boston': 3, 'San Mateo': 4}
   

    def get_dayOfWeek2():
      dayOfWeek = st.selectbox('Select a day of week', dowLabels,key='tab2_dayOfWeekSelect')
      return dayOfWeek

    def get_menuType():
      #MENU_TYPES = df[df['DAY_OF_WEEK'] == dowMapping[DAY_OF_WEEK]]['MENU_TYPE'].unique()
      MENU_TYPE = st.selectbox('Select a menu type', mt_mapping)
      return MENU_TYPE
      
    def get_MenuItemName():
      MENU_ITEM_NAME = st.selectbox('Select a item name', min_mapping)
      return MENU_ITEM_NAME  

    def get_itemCat():
      ITEM_CATEGORY = st.selectbox('Select a item category', ic_mapping)
      return ITEM_CATEGORY  

    def get_itemSubCat(ITEM_CATEGORY):
      #ITEM_SUBCATEGORYs = df[df['ITEM_CATEGORY'] == dow_mapping[ITEM_CATEGORY]]['ITEM_SUBCATEGORY'].unique()
      ITEM_SUBCATEGORY = st.selectbox('Select a item sub-category', isc_mapping)
      return ITEM_SUBCATEGORY  

    def get_TruckBrandName():
      TRUCK_BRAND_NAME = st.selectbox('Select a truck brand name', tbn_mapping)
      return TRUCK_BRAND_NAME  

    def get_City():
      CITY = st.selectbox('Select a city', c_mapping)
      return CITY  

    # Define the user input fields
    dow_input = get_dayOfWeek2()
    mt_input = get_menuType()    
    min_input = get_MenuItemName()
    ic_input = get_itemCat()
    isc_input = get_itemSubCat(ic_input)  
    tbn_input = get_TruckBrandName()  
    c_input = get_City()    

    # Map user inputs to integer encoding
    dow_int = dowMapping[dow_input]
    mt_int = mt_mapping[mt_input]
    min_int = min_mapping[min_input]
    ic_int = ic_mapping[ic_input]
    isc_int = isc_mapping[isc_input]  
    tbn_int = tbn_mapping[tbn_input]
    c_int = c_mapping[c_input]  
  
    # Display the prediction
    if st.button('Predict Price'):
        
    # Make the prediction   
    input_data = [[dow_int, mt_int, min_int, ic_int, isc_int, tbn_int, c_int]]
    input_df = pd.DataFrame(input_data, columns=['DAY_OF_WEEK', 'MENU_TYPE', 'MENU_ITEM_NAME', 
                                                 'ITEM_CATEGORY', 'ITEM_SUBCATEGORY', 'TRUCK_BRAND_NAME', 'CITY'])
    prediction = xgb_xinle.predict(input_df)   

    # Convert output data and columns, including profit, to a dataframe
    output_data = [DAY_OF_WEEK, MENU_ITEM_NAME, MENU_TYPE, ITEM_CATEGORY, ITEM_SUBCATEGORY, TRUCK_BRAND_NAME, CITY, prediction[0]]
    output_df = pd.DataFrame([output_data], columns=['DAY_OF_WEEK', 'MENU_ITEM_NAME', 'MENU_TYPE', 'ITEM_CATEGORY', 
                                                     'ITEM_SUBCATEGORY', 'TRUCK_BRAND_NAME', 'CITY', 'PREDICTED_PROFIT'])

    # Show prediction on profit
    predicted_profit = output_df['PREDICTED_PROFIT'].iloc[0]
    st.write('The predicted profit is {:.2f}.'.format(predicted_profit))
    st.dataframe(output_df)





    

with tab3:
  st.write()

with tab4:
  st.write()

with tab5:
  st.write()
