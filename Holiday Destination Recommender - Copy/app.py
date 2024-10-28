from flask import Flask, render_template, request
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

app = Flask(__name__)

# Sample dataset of destinations and user preferences
destinations = [
    {'name': 'Paris', 'type': 'city', 'temperature': 'mild', 'activities': 'sightseeing', 'flight time': 'above 17'},
    {'name': 'Barcelona', 'type': 'culture', 'temperature': 'hot', 'activities': 'relaxation', 'flight time': 'above 17'},
    {'name': 'New York', 'type': 'city', 'temperature': 'mild', 'activities': 'sightseeing', 'flight time': 'above 17'},
    {'name': 'Rome', 'type': 'culture', 'temperature': 'hot', 'activities': 'sightseeing', 'flight time': 'between 10 - 17'},
    {'name': 'Istanbul', 'type': 'city', 'temperature': 'hot', 'activities': 'shopping', 'flight time': 'between 10 - 17'},
    {'name': 'Cancun', 'type': 'beach', 'temperature': ' hot', 'activities': 'relaxation', 'flight time': 'above 17'},
    {'name': 'London', 'type': 'city', 'temperature': 'cold', 'activities': 'shopping', 'flight time': 'above 17'},
    {'name': 'Shanghai', 'type': 'city', 'temperature': 'mild', 'activities': 'shopping', 'flight time': 'less than 10'},
    {'name': 'Berlin', 'type': 'city', 'temperature': 'mild', 'activities': 'sightseeing', 'flight time': 'between 10 - 17'},
    {'name': 'Santorini', 'type': 'beach', 'temperature': 'hot', 'activities': 'relaxation', 'flight time': 'above 17'},
    {'name': 'Vienna', 'type': 'culture', 'temperature': 'mild', 'activities': 'sightseeing', 'flight time': 'between 10 - 17'},
    {'name': 'Phuket', 'type': 'beach', 'temperature': 'hot', 'activities': 'relaxation', 'flight time': 'less than 10'},
    {'name': 'Dubai', 'type': 'city', 'temperature': 'hot', 'activities': 'shopping', 'flight time': 'between 10 - 17'},
    {'name': 'Toronto', 'type': 'city', 'temperature': 'cold', 'activities': 'adventure/sport', 'flight time': 'above 17'},
    {'name': 'Lisbon', 'type': 'culture', 'temperature': 'mild', 'activities': 'sightseeing', 'flight time': 'above 17'},
    {'name': 'Tokyo', 'type': 'city', 'temperature': 'cold', 'activities': 'shopping', 'flight time': 'between 10 - 17'},
    {'name': 'Amsterdam', 'type': 'culture', 'temperature': 'mild', 'activities': 'sightseeing', 'flight time': 'Above 17'},
    {'name': 'Kuala Lumpur', 'type': 'city', 'temperature': 'hot', 'activities': 'sightseeing', 'flight time': 'less than 10'},
    {'name': 'Warsaw', 'type': 'culture', 'temperature': 'cold', 'activities': 'sightseeing', 'flight time': 'between 10 - 17'},
    {'name': 'Hong Kong', 'type': 'city', 'temperature': 'hot', 'activities': 'shopping', 'flight time': 'less than 10'},
]

# Prepare data for the model
X = []
for destination in destinations:
    X.append([destination['type'], destination['temperature'], destination['activities'], destination['flight time']])

# Encoding categorical features
label_encoders = {}
for i in range(len(X[0])):
    le = LabelEncoder()
    column = [row[i] for row in X]
    le.fit(column)
    label_encoders[i] = le
    for row in X:
        row[i] = le.transform([row[i]])[0]

X = np.array(X)

# Initialize the Decision Tree model
model = DecisionTreeClassifier()

# Fit the model
model.fit(X, [destination['name'] for destination in destinations])

# For prediction, use model.predict()
def get_recommendation(user_preferences):
    encoded_preferences = []  # You may need to encode these based on the features
    for i in range(len(user_preferences)):
        encoded_preferences.append(label_encoders[i].transform([user_preferences[i]])[0])

    encoded_preferences = np.array(encoded_preferences).reshape(1, -1)

    # Predict with the trained tree
    recommended_destination = model.predict(encoded_preferences)[0]

    return recommended_destination

    # Flask routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    destination_type = request.form['type']
    temperature = request.form['temperature']
    activities = request.form['activities']
    flight_time = request.form['flight_time']

    # Pass user preferences to the recommender system
    user_preferences = [destination_type, temperature, activities, flight_time]
    recommendation = get_recommendation(user_preferences)

    return render_template('result.html', destination=recommendation)

if __name__ == '__main__':
    app.run(debug=True)

