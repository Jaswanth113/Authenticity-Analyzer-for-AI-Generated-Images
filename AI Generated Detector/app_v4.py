import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageChops, ImageEnhance
import sqlite3 
import os

# Initialize session state
def init_session():
    if 'login_status' not in st.session_state:
        st.session_state.login_status = False
    if 'username' not in st.session_state:
        st.session_state.username = ""
    if 'search_history' not in st.session_state:
        st.session_state.search_history = []

# Security functions (no bcrypt)
def make_hashes(password):
    return password

def check_hashes(password, hashed_text):
    return password == hashed_text

# DB Management
conn = sqlite3.connect('data_v4.db')
c = conn.cursor()

# DB Functions
def create_usertable():
    c.execute('CREATE TABLE IF NOT EXISTS userstable504(username TEXT,email TEXT,password TEXT)')
    conn.commit()

def add_userdata(username, email, password):
    c.execute('INSERT INTO userstable504(username,email,password) VALUES (?,?,?)', (username, email, password))
    conn.commit()

def login_user(username, password):
    c.execute('SELECT * FROM userstable504 WHERE username =? AND password = ?', (username, password))
    data = c.fetchall()
    return data

def check_username(username):
    c.execute('SELECT * FROM userstable504 WHERE username = ?', (username,))
    data = c.fetchall()
    return len(data) > 0

def create_search_history_table():
    c.execute('''CREATE TABLE IF NOT EXISTS search_history
                 (username TEXT, image_path TEXT, result TEXT, probability REAL)''')
    conn.commit()

def add_to_search_history(username, image_path, result, probability):
    c.execute('INSERT INTO search_history (username, image_path, result, probability) VALUES (?, ?, ?, ?)', (username, image_path, result, probability))
    conn.commit()

def get_search_history(username):
    c.execute('SELECT image_path, result, probability FROM search_history WHERE username = ?', (username,))
    return c.fetchall()

# Load the trained model
model = tf.keras.models.load_model('model_casia_run.h5')

# Define the labels for prediction
labels = ['AI Generated / Tampered', 'Authentic']

def preprocess_image(image):
    temp_filename = 'temp_file_name.jpg'
    ela_filename = 'temp_ela.jpg'

    if image.mode != 'RGB':
        image = image.convert('RGB')

    image.save(temp_filename, 'JPEG', quality=90)
    temp_image = Image.open(temp_filename)

    ela_image = ImageChops.difference(image, temp_image)

    extrema = ela_image.getextrema()
    max_diff = max([ex[1] for ex in extrema])
    if max_diff == 0:
        max_diff = 1
    scale = 255.0 / max_diff

    ela_image = ImageEnhance.Brightness(ela_image).enhance(scale)

    ela_image = ela_image.convert('RGB')

    ela_image = ela_image.resize((128, 128))

    image_array = np.array(ela_image) / 255.0

    preprocessed_image = np.expand_dims(image_array, axis=0)

    return preprocessed_image

# Streamlit app
def main():
    try:
        init_session()  # Initialize session state
        
        st.set_page_config(layout="wide")  # Wide layout for better responsiveness
        
        st.markdown("# Authen-Lens")
        menu = ["Authen-Lens"]
        
        if st.session_state.login_status:
            menu.append("Logout")
            menu.append("Search History")
        else:
            menu.extend(["Login", "Signup"])
        
        choice = st.sidebar.selectbox("Menu", menu)

        if choice == "Authen-Lens":
            if not st.session_state.login_status:
                authen_lens_section()
            else:
                st.write(f"Welcome {st.session_state.username}")
                authen_lens_section()
        elif choice == "Login":
            login_section()
        elif choice == "Signup":
            signup_section()
        elif choice == "Logout":
            st.session_state.login_status = False
            st.session_state.username = ""
            st.success("Logged Out successfully")
        elif choice == "Search History":
            display_search_history()
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

def authen_lens_section():
    try:
        create_search_history_table()
        st.subheader("Image Forgery Detection")
        uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png", "webp"])
        if uploaded_file is not None:
            process_image(uploaded_file)
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

def login_section():
    try:
        st.subheader("Login Section")
        username = st.text_input("User Name", key="login-username")
        password = st.text_input("Password", type='password', value='', key="login-password")
        login_button = st.button("Login")

        create_usertable()  # Moved outside of the button click check to ensure table exists

        if login_button:
            data = login_user(username, password)

            if data:
                st.session_state.login_status = True
                st.session_state.username = username
                st.success("Logged In as {}".format(username))
            else:
                st.error("Incorrect Username or Password.")
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

def signup_section():
    try:
        st.subheader("Create New Account")
        new_user = st.text_input("Username", key="signup-username")
        mailid = st.text_input("Email", key="signup-email")
        new_password = st.text_input("Password", type='password', value='', key="signup-password")

        signup_button = st.button("Signup")

        if signup_button:
            create_usertable()
            add_userdata(new_user, mailid, make_hashes(new_password))
            st.success("You have successfully created a valid Account")
            st.info("Go to Login Menu to login")
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

def process_image(uploaded_file):
    try:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        st.info("Processing image...")
        progress_bar = st.progress(0)

        preprocessed_image = preprocess_image(image)

        progress_bar.progress(50)
        prediction = model.predict(preprocessed_image)
        prediction_label = labels[np.argmax(prediction)]
        prediction_confidence = np.max(prediction) * 100

        progress_bar.progress(100)

        st.subheader("Prediction:")
        st.write(prediction_label)
        st.subheader("Confidence:")
        st.write(f"{prediction_confidence:.2f}%")

        if st.session_state.login_status:
            username_folder = os.path.join('search_history', st.session_state.username)
            os.makedirs(username_folder, exist_ok=True)
            uploaded_file_path = os.path.join(username_folder, uploaded_file.name)
            image.save(uploaded_file_path)  # Save the uploaded file to the user's folder
            add_to_search_history(st.session_state.username, uploaded_file_path, prediction_label, prediction_confidence)
    except Exception as e:
        st.error(f"An error occurred while processing the image: {str(e)}")

def display_search_history():
    try:
        st.subheader("Search History")
        search_history = get_search_history(st.session_state.username)
        for image_path, result, probability in search_history:
            st.image(Image.open(image_path), caption=f"Result: {result}", use_column_width=True)
            st.write(f"Probability: {probability:.2f}%", style="font-size: 16px; font-weight: bold; color: #555; margin-top: -30px; margin-left: 15px")
            st.write("---")
    except Exception as e:
        st.error(f"An error occurred while fetching search history: {str(e)}")

if __name__ == "__main__":
    main()

