import streamlit as st
import pandas as pd
import numpy as np
import PIL.Image
import snowflake.connector
import pydeck as pdk
import pickle
from PIL import Image
from sklearn.ensemble import RandomForestRegressor

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
    
    with open('rf_xinle.pkl', 'rb') as file:
        rf = pickle.load(file)

    # Load the cleaned and transformed dataset
    df = pd.read_csv('df_xinle.csv')
    price = df[['TOTAL_QUANTITY']] # extract price column from df_xinle

    dowMapping={'Monday':0,'Tuesday':1,'Wednesday':2,'Thursday':3,'Friday':4,'Saturday':5,'Sunday':6}
    dowReverseMapping = {v: k for k, v in dowMapping.items()}
    dowLabels = [dowReverseMapping[i] for i in sorted(dowReverseMapping.keys())]

    mt_mapping = {'Vegetarian': 0,
                  'Crepes': 1,
                  'Chinese': 2,
                  'Ice Cream': 3,
                  'Mac & Cheese': 4,
                  'Hot Dogs': 5,
                  'Ethiopian': 6,
                  'Grilled Cheese': 7,
                  'BBQ': 8,
                  'Gyros': 9,
                  'Indian': 10,
                  'Ramen': 11,
                  'Poutine': 12,
                  'Tacos': 13,
                  'Sandwiches': 14}
  
    ic_mapping = {'Main': 0, 'Beverage': 1, 'Dessert': 2, 'Snack': 3}

    isc_mapping= {'Hot Option': 0, 'Cold Option': 1, 'Warm Option': 2}
   

    def get_dayOfWeek2():
      dayOfWeek = st.selectbox('Select a day of week', dowLabels,key='tab2_dayOfWeekSelect')
      return dayOfWeek

    def get_menuType(DAY_OF_WEEK):
      # show only the menu items for the selected day of week
      MENU_TYPES = df[df['DAY_OF_WEEK'] == dowMapping[DAY_OF_WEEK]]['MENU_TYPE'].unique()
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
    dow_input = get_dayOfWeek2()
    mt_input = get_menuType(dow_input)
    ic_input = get_itemCat(mt_input)
    isc_input = get_itemSubCat(ic_input)  
  

    # Map user inputs to integer encoding
    dow_int = dowMapping[dow_input]
    mt_int = mt_mapping[mt_input]
    ic_int = ic_mapping[ic_input]
    isc_int = isc_mapping[isc_input]  

    # Display the prediction
    if st.button('Predict Price'):
        
        # Make the prediction   
        input_data = [[dow_int,mt_int,ic_int,isc_int]]
        input_df = pd.DataFrame(input_data, columns=['DAY_OF_WEEK', 'MENU_ITEM_NAME', 'MENU_TYPE','ITEM_CATEGORY','ITEM_SUBCATEGORY'])
        prediction = rf.predict(input_df)   
        # convert output data and columns, including price, to a dataframe avoiding TypeError: type numpy.ndarray doesn't define round method
        output_data = [DAY_OF_WEEK, MENU_ITEM_NAME, MENU_TYPE,ITEM_CATEGORY,ITEM_SUBCATEGORY, UNIT_PRICE, prediction[0]]

    
        output_df = pd.DataFrame([output_data], columns=['DAY_OF_WEEK', 'MENU_ITEM_NAME', 'MENU_TYPE','ITEM_CATEGORY','ITEM_SUBCATEGORY', 'predicted_quantity'])

        # Make the prediction   
        # show prediction on price in dollars and cents using the price column 
        input_data = [[dow_int, MENU_ITEM_NAME, mt_int,ic_int,isc_int]]

        predicted_price = xgb.predict(input_df)[0]
        st.write('The predicted average price is ${:.2f}.'.format(predicted_quantity))
        st.dataframe(output_df)





    

with tab3:
  st.write()

with tab4:
  st.write()

with tab5:
  st.write()
