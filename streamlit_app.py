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
  # Define the app title and favicon
    st.write('How much can you make from the TastyBytes locations?')
    st.markdown("This tab allows you to make predictions on the price of a listing based on the neighbourhood and room type. The model used is a Random Forest Regressor trained on the Airbnb Singapore listings dataset.")
    st.write('Choose a Truck Brand Name, City, Truck Location and Time Frame to get the predicted monetary sales.')


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
      dayOfWeek = st.selectbox('Select a day of week', dow_labels)
      return dayOfWeek
        
    def get_truckBrandName(TRUCK_BRAND_NAME):
      TRUCK_BRAND_NAME = df[df['DAY_OF_WEEK'] == dow_mapping[DAY_OF_WEEK]]['TRUCK_BRAND_NAME'].unique()
      TRUCK_BRAND_NAME = st.selectbox('Select a truck brand name', bn_mapping)
      return TRUCK_BRAND_NAME
        
    def get_truckCity(CITY):
      CITY = df[df['TRUCK_BRAND_NAME'] == bn_mapping[TRUCK_BRAND_NAME]]['CITY'].unique()
      CITY = st.selectbox('Select a city', ct_mapping)
      return CITY

    # Define the user input fields
    dow_input = get_dayOfWeek()
    bn_input = get_truckBrandName(dow_input)
    ct_input = get_truckCity(bn_mapping)

    # Map user inputs to integer encoding
    dow_int = dow_mapping[dow_input]
    bn_input = bn_mapping[bn_input]
    ct_input = ct_mapping[ct_input]



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
    dow_labels = [dow_reverse_mapping[i] for i in sorted(dow_reverse_mapping.keys())]

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
   

    def get_dayOfWeek():
      dayOfWeek = st.selectbox('Select a day of week', dow_labels)
      return dayOfWeek

    def get_menuType(DAY_OF_WEEK):
      # show only the menu items for the selected day of week
      MENU_TYPES = df[df['DAY_OF_WEEK'] == dow_mapping[DAY_OF_WEEK]]['MENU_TYPE'].unique()
      MENU_TYPE = st.selectbox('Select a menu type', mt_mapping)
      return MENU_TYPE

    def get_itemCat(MENU_TYPE):
      # show only the menu items for the selected day of week
      #ITEM_CATEGORYS = df[df['MENU_TYPE'] == dow_mapping[MENU_TYPE]]['ITEM_CATEGORY'].unique()
      ITEM_CATEGORY = st.selectbox('Select a item category', ic_mapping)
      return ITEM_CATEGORY  

    def get_itemSubCat(ITEM_CATEGORY):
      # show only the menu items for the selected day of week
      #ITEM_SUBCATEGORYs = df[df['ITEM_CATEGORY'] == dow_mapping[ITEM_CATEGORY]]['ITEM_SUBCATEGORY'].unique()
      ITEM_SUBCATEGORY = st.selectbox('Select a item sub-category', isc_mapping)
      return ITEM_SUBCATEGORY   

    # Define the user input fields
    dow_input = get_dayOfWeek()
    mt_input = get_menuType(dow_input)
    ic_input = get_itemCat(mt_input)
    isc_input = get_itemSubCat(ic_input)  
  

    # Map user inputs to integer encoding
    dow_int = dow_mapping[dow_input]
    mt_int = mt_mapping[mt_input]
    ic_int = ic_mapping[ic_input]
    isc_int = isc_mapping[isc_input]  





    

with tab3:
  st.write()

with tab4:
  st.write()

with tab5:
  st.write()
