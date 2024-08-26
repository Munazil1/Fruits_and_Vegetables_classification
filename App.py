import os
import shutil
import streamlit as st
from PIL import Image
import requests
from bs4 import BeautifulSoup
import numpy as np
from tensorflow.keras.utils import load_img, img_to_array
from tensorflow.keras.models import load_model

model = load_model('FV.h5')

labels = {0: 'apple', 1: 'banana', 2: 'beetroot', 3: 'bell pepper', 4: 'cabbage', 5: 'capsicum', 6: 'carrot',
          7: 'cauliflower', 8: 'chilli pepper', 9: 'corn', 10: 'cucumber', 11: 'eggplant', 12: 'garlic', 13: 'ginger',
          14: 'grapes', 15: 'jalepeno', 16: 'kiwi', 17: 'lemon', 18: 'lettuce',
          19: 'mango', 20: 'onion', 21: 'orange', 22: 'paprika', 23: 'pear', 24: 'peas', 25: 'pineapple',
          26: 'pomegranate', 27: 'potato', 28: 'raddish', 29: 'soy beans', 30: 'spinach', 31: 'sweetcorn',
          32: 'sweetpotato', 33: 'tomato', 34: 'turnip', 35: 'watermelon'}

fruits = ['Apple', 'Banana', 'Bello Pepper', 'Chilli Pepper', 'Grapes', 'Jalepeno', 'Kiwi', 'Lemon', 'Mango', 'Orange',
          'Paprika', 'Pear', 'Pineapple', 'Pomegranate', 'Watermelon']
vegetables = ['Beetroot', 'Cabbage', 'Capsicum', 'Carrot', 'Cauliflower', 'Corn', 'Cucumber', 'Eggplant', 'Ginger',
              'Lettuce', 'Onion', 'Peas', 'Potato', 'Raddish', 'Soy Beans', 'Spinach', 'Sweetcorn', 'Sweetpotato',
              'Tomato', 'Turnip']

def fetch_calories(prediction):
    try:
        url = 'https://www.google.com/search?&q=calories in ' + prediction
        req = requests.get(url).text
        scrap = BeautifulSoup(req, 'html.parser')
        calories = scrap.find("div", class_="BNeawe iBp4i AP7Wnd").text
        return calories
    except Exception as e:
        st.error("Can't able to fetch the Calories")
        print(e)

def prepare_image(img_path):
    img = load_img(img_path, target_size=(224, 224))  # Corrected target size (should be 224, 224)
    img = img_to_array(img)
    img = img / 255.0  # Normalized image
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    answer = model.predict(img)
    y_class = answer.argmax(axis=-1)
    res = labels[int(y_class)]
    return res.capitalize()

def move_image(img_path, category):
    fruit_folder = './upload_images/fruits/'
    vegetable_folder = './upload_images/vegetables/'

    if not os.path.exists(fruit_folder):
        os.makedirs(fruit_folder)
    if not os.path.exists(vegetable_folder):
        os.makedirs(vegetable_folder)

    if category in vegetables:
        shutil.move(img_path, os.path.join(vegetable_folder, os.path.basename(img_path)))
    else:
        shutil.move(img_path, os.path.join(fruit_folder, os.path.basename(img_path)))

def run():
    st.title("Fruits-Vegetable Classification")
    img_file = st.file_uploader("Choose an Image", type=["jpg", "png"])
    
    if img_file is not None:
        # Save the uploaded image
        if not os.path.exists('./upload_images'):
            os.makedirs('./upload_images')
        
        img = Image.open(img_file).resize((250, 250))
        st.image(img, use_column_width=False)
        save_image_path = './upload_images/' + img_file.name
        
        with open(save_image_path, "wb") as f:
            f.write(img_file.getbuffer())
        
        # Prepare and classify the image
        result = prepare_image(save_image_path)
        
        if result in vegetables:
            st.info('**Category : Vegetables**')
        else:
            st.info('**Category : Fruit**')
        
        st.success("**Predicted : " + result + '**')
        
        # Fetch and display calories information
        cal = fetch_calories(result)
        if cal:
            st.warning('**' + cal + '(100 grams)**')
        
        # Move the image to the appropriate folder
        move_image(save_image_path, result)

run()
