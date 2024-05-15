import random
import time
import pandas as pd
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Load the dataset
data = pd.read_csv("train.csv")

# Data preprocessing
data.dropna(subset=['Embarked'], inplace=True)  # Drop rows with missing Embarked values
data['Age'].fillna(data['Age'].median(), inplace=True)  # Fill missing Age values with median

# Feature engineering
data['FamilySize'] = data['SibSp'] + data['Parch']

# Select features and target variable
X = data[['Pclass', 'Sex', 'Age', 'FamilySize', 'Embarked']]
y = data['Survived']

# Define preprocessing steps for numerical and categorical features
numeric_features = ['Age', 'FamilySize']
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

categorical_features = ['Pclass', 'Sex', 'Embarked']
categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Define the model
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# Train the model
model.fit(X, y)

# Define passenger classes
passenger_classes = {
    1: "First Class",
    2: "Second Class",
    3: "Third Class"
}

# Define game start time (11:40 PM on April 14, 1912)
start_datetime = datetime(1912, 4, 14, 23, 40)

# Define time when the ship sank (2:20 AM on April 15, 1912)
end_datetime = start_datetime + timedelta(hours=2, minutes=40)

# Define Titanic status updates
status_updates = [
    (start_datetime, "April 14, 1912, 11:40 PM: Lookout Frederick Fleet spotted an iceberg dead ahead. The iceberg struck the Titanic on the starboard (right) side of her bow.\r"),
    (start_datetime + timedelta(minutes=10), "\nApril 14, 1912, 11:50 PM: Water had poured in and risen 14 feet in the front part of the ship.\r"),
    (start_datetime + timedelta(hours=0, minutes=20), "\nApril 15, 1912, 12:00 AM: The captain was told the ship can only stay afloat for a couple of hours. He gave the order to call for help over the radio."),
    (start_datetime + timedelta(hours=0, minutes=25), "\nApril 15, 1912, 12:05 AM: The order was given to uncover the lifeboats and to get passengers and crew ready on deck. There was only room in the lifeboats for half of the estimated 2,227 on board.\r"),
    (start_datetime + timedelta(hours=0, minutes=45), "\nApril 15, 1912, 12:25 AM: The lifeboats began being loaded with women and children first. The Carpathia, southeast of the Titanic by about 58 miles, picked up the distress call and began sailing towards the Titanic.\r"),
    (start_datetime + timedelta(hours=1, minutes=30), "\nApril 16, 1912, 2:10 AM: The Titanic's lights go out. The bow of the ship is completely underwater and the back of the boat is starting to life up. \r"),
    (start_datetime + timedelta(hours=1, minutes=37), "\nApril 16, 1912, 2:17 AM: The Titanic breaks in two \r"),
    (start_datetime + timedelta(hours=1, minutes=39), "\nApril 16, 1912, 2:17 AM: The Titanic's bow begins to sink \r"),
    (end_datetime, "\nApril 15, 1912, 2:20 AM: The Titanic sinks to the bottom of the ocean.\r")
]

def choose_passenger_class():
    while True:
        print("Choose your passenger class:")
        for class_num, class_name in passenger_classes.items():
            print(f"{class_num}. {class_name}")
        choice = input("Enter the number corresponding to your desired class: ")
        if choice in [str(i) for i in passenger_classes.keys()]:
            return int(choice)
        else:
            print("Invalid choice. Please try again.")

def assign_passenger(passenger_class):
    passengers = data[(data["Pclass"] == passenger_class) & (~data["Name"].isna())]
    if not passengers.empty:
        passenger = passengers.sample(1).iloc[0]
        name = passenger["Name"]
        sex = "Male" if passenger["Sex"] == "male" else "Female"
        age = passenger["Age"] if not pd.isna(passenger["Age"]) else "Unknown"
        print(f"\nWelcome to the first voyage of the Titanic! Your name is {name}, a {sex} passenger aged {age}.\r")
        print("  ")
        return passenger
    else:
        print("\nNo passengers found for the chosen class.")
        return None

def predict_survival_probability(passenger_data):
    return model.predict_proba(passenger_data)[0, 1]

def check_survival(survival_probability):
    return random.random() < survival_probability

def play_game():
    print("Welcome to the Titanic Survival Game!")
    passenger_class = choose_passenger_class()
    print(f"\nYou have chosen to travel in {passenger_classes[passenger_class]}.")
    passenger = assign_passenger(passenger_class)

    if passenger is not None:
        passenger_data = pd.DataFrame([[
            passenger["Pclass"],
            passenger["Sex"],
            passenger["Age"],
            passenger["SibSp"] + passenger["Parch"],
            passenger["Embarked"]
        ]], columns=['Pclass', 'Sex', 'Age', 'FamilySize', 'Embarked'])
        
        survival_probability = predict_survival_probability(passenger_data)
        
        print(f"Your estimated probability of survival is {survival_probability:.2f}")
        print("   ")

        current_datetime = start_datetime
        for update_datetime, update_text in status_updates:
            while current_datetime < update_datetime:
                time.sleep(1)  # Wait for 1 second 
                current_datetime += timedelta(minutes=1)
                print(f"\nCurrent time: {current_datetime.strftime('%B %d, %Y, %I:%M %p')}")

            print(update_text)

            if update_datetime == end_datetime:  # After the last update (ship sinking)
                if check_survival(survival_probability):
                    print("\nCongratulations! You have survived the Titanic disaster.\r")
                else:
                    print("\nUnfortunately, you did not survive the Titanic disaster.\r")
                break

    print("\nGame over. Thank you for playing!")

play_game()

