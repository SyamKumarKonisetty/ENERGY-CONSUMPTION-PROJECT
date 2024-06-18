from flask import Flask,render_template,redirect,request,url_for, send_file
import mysql.connector
from flask import Flask, request, render_template
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from keras.models import Sequential
from keras.layers import LSTM, SimpleRNN, GRU, Dense
import numpy as np
import pandas as pd
from keras.models import load_model

app = Flask(__name__)

mydb = mysql.connector.connect(
    host="localhost",
    user="root",
    password="admin",
    port="3306",
    database='electric'
)


mycursor = mydb.cursor()

def executionquery(query,values):
    mycursor.execute(query,values)
    mydb.commit()
    return

def retrivequery1(query,values):
    mycursor.execute(query,values)
    data = mycursor.fetchall()
    return data

def retrivequery2(query):
    mycursor.execute(query)
    data = mycursor.fetchall()
    return data


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/register', methods=["GET", "POST"])
def register():
    if request.method == "POST":
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']
        c_password = request.form['c_password']
        if password == c_password:
            query = "SELECT UPPER(email) FROM users"
            email_data = retrivequery2(query)
            email_data_list = []
            for i in email_data:
                email_data_list.append(i[0])
            if email.upper() not in email_data_list:
                query = "INSERT INTO users (name, email, password) VALUES (%s, %s, %s)"
                values = (name, email, password)
                executionquery(query, values)
                return render_template('login.html', message="Successfully Registered! Please go to login section")
            return render_template('register.html', message="This email ID is already exists!")
        return render_template('register.html', message="Conform password is not match!")
    return render_template('register.html')


@app.route('/login', methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form['email']
        password = request.form['password']
        
        query = "SELECT UPPER(email) FROM users"
        email_data = retrivequery2(query)
        email_data_list = []
        for i in email_data:
            email_data_list.append(i[0])

        if email.upper() in email_data_list:
            query = "SELECT UPPER(password) FROM users WHERE email = %s"
            values = (email,)
            password__data = retrivequery1(query, values)
            if password.upper() == password__data[0][0]:
                global user_email
                user_email = email

                return redirect("/home")
            return render_template('home.html', message= "Invalid Password!!")
        return render_template('login.html', message= "This email ID does not exist!")
    return render_template('login.html')


@app.route('/home')
def home():
    return render_template('home.html')



@app.route('/algorithm', methods=['GET', 'POST'])
def algorithm():
    if request.method == "POST":
        algorithm_type = request.form['algorithm']
        
        if algorithm_type == 'Linear Regression':
            accuracy = linear_regression()
        elif algorithm_type == 'RNN':
            accuracy = rnn()
        elif algorithm_type == 'LSTM':
            accuracy = lstm()
        elif algorithm_type == 'Stacked LSTM':
            accuracy = stacked()
        elif algorithm_type == 'GRU':
            accuracy = gru()

        return render_template('algorithm.html', accuracy=accuracy)

    return render_template('algorithm.html')


def linear_regression():
    new_df = pd.read_csv('new_df.csv')
    X_linear = new_df[['temperature', 'var1', 'pressure', 'windspeed','year','month','day','day_of_week','hour']]
    y_linear = new_df['electricity_consumption']
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_linear, y_linear, test_size=0.2, random_state=42)
    
    # Create and train the linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Make predictions on the test set
    y_pred = model.predict(X_test)
    # Evaluate the model
    mea = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    metrics = {
        'Mean Absolute Error': mea,
        'Mean Square Error': mse,
        'Root Mean Square Error': rmse,
        'R2 Score': r2
    }
    # Evaluate the model
    accuracy = r2_score(y_test, y_pred)
    
    return accuracy, metrics



def rnn():
    new_df = pd.read_csv('new_df.csv')
    Xrnn = new_df[['temperature', 'var1', 'pressure', 'windspeed','year','month','day','day_of_week','hour']].values
    yrnn = new_df['electricity_consumption'].values

    sequence_length = 1  # Each data point is treated independently

    # Reshape the input data into sequences
    X_seq = []
    y_seq = []

    for i in range(len(Xrnn) - sequence_length):
        X_seq.append(Xrnn[i:i + sequence_length])
        y_seq.append(yrnn[i + sequence_length])

    X_seq = np.array(X_seq)
    y_seq = np.array(y_seq)
    
    X_trainrnn, X_testrnn, y_trainrnn, y_testrnn = train_test_split(X_seq, y_seq, test_size=0.2, random_state=42)

    # Define the RNN model
    RNNmodel = Sequential()
    RNNmodel.add(SimpleRNN(50, activation='relu', input_shape=(X_trainrnn.shape[1], X_trainrnn.shape[2])))
    RNNmodel.add(Dense(1))  # Output layer with 1 neuron for regression
    RNNmodel.compile(optimizer='adam', loss='mean_squared_error')
    
    RNNmodel.fit(X_trainrnn, y_trainrnn, epochs=100, batch_size=32, verbose=1)

    y_predRNN = RNNmodel.predict(X_testrnn)
    
    # Evaluate the model
    mea = mean_absolute_error(y_testrnn, y_predRNN)
    mse = mean_squared_error(y_testrnn, y_predRNN)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_testrnn, y_predRNN)
    metrics = {
        'Mean Absolute Error': mea,
        'Mean Square Error': mse,
        'Root Mean Square Error': rmse,
        'R2 Score': r2
    }
    # Evaluate the model
    accuracy = r2_score(y_testrnn, y_predRNN)
    
    return accuracy, metrics


def lstm():
    new_df = pd.read_csv('new_df.csv')
    Xlstm = new_df[['temperature', 'var1', 'pressure', 'windspeed', 'year', 'month', 'day', 'day_of_week', 'hour']].values
    ylstm = new_df['electricity_consumption'].values
    
    sequence_length = 1  # Each data point is treated independently

    # Reshape the input data into sequences
    X_seq = []
    y_seq = []

    for i in range(len(Xlstm) - sequence_length):
        X_seq.append(Xlstm[i:i + sequence_length])
        y_seq.append(ylstm[i + sequence_length])

    X_seq = np.array(X_seq)
    y_seq = np.array(y_seq)

    # Split the data into training and testing sets
    X_trainlstm, X_testlstm, y_trainlstm, y_testlstm = train_test_split(X_seq, y_seq, test_size=0.2, random_state=42)
    
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(X_trainlstm.shape[1], X_trainlstm.shape[2])))
    model.add(Dense(1))  # Output layer with 1 neuron for regression
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the LSTM model
    model.fit(X_trainlstm, y_trainlstm, epochs=100, batch_size=32, verbose=1)

    # Evaluate the model on the test set
    y_predLSTM = model.predict(X_testlstm)
    # Evaluate the model
    mea = mean_absolute_error(y_testlstm, y_predLSTM)
    mse = mean_squared_error(y_testlstm, y_predLSTM)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_testlstm, y_predLSTM)
    r2 = abs(r2)
    error_rate = 1-r2
    metrics = {
        'Mean Absolute Error': mea,
        'Mean Square Error': mse,
        'Root Mean Square Error': rmse,
        'R2 Score': r2,
        'Error Rate': abs(error_rate)
    }
    # Evaluate the model
    accuracy = r2_score(y_testlstm, y_predLSTM)
    
    return accuracy, metrics

def stacked():
    new_df = pd.read_csv('new_df.csv')
    XSlstm = new_df[['temperature', 'var1', 'pressure', 'windspeed', 'year', 'month', 'day', 'day_of_week', 'hour']].values
    ySlstm = new_df['electricity_consumption'].values

    # Define sequence length (number of previous time steps to consider)
    sequence_length = 1  # Each data point is treated independently

    # Reshape the input data into sequences
    X_seq = []
    y_seq = []

    for i in range(len(XSlstm) - sequence_length):
        X_seq.append(XSlstm[i:i + sequence_length])
        y_seq.append(ySlstm[i + sequence_length])

    X_seq = np.array(X_seq)
    y_seq = np.array(y_seq)

    # Split the data into training and testing sets
    X_trainSlstm, X_testSlstm, y_trainSlstm, y_testSlstm = train_test_split(X_seq, y_seq, test_size=0.2, random_state=42)
    

    # Define the stacked LSTM model
    SLSTMmodel = Sequential()

    # First LSTM layer
    SLSTMmodel.add(LSTM(50, activation='relu', return_sequences=True, input_shape=(X_trainSlstm.shape[1], X_trainSlstm.shape[2])))

    # Second LSTM layer (stacked)
    SLSTMmodel.add(LSTM(50, activation='relu', return_sequences=True))

    # Third LSTM layer (stacked)
    SLSTMmodel.add(LSTM(50, activation='relu'))

    # Output layer with 1 neuron for regression
    SLSTMmodel.add(Dense(1))

    SLSTMmodel.compile(optimizer='adam', loss='mean_squared_error')

    # Train the LSTM model
    SLSTMmodel.fit(X_trainSlstm, y_trainSlstm, epochs=100, batch_size=32, verbose=1)

    # Evaluate the model on the test set
    y_predSLSTM = SLSTMmodel.predict(X_testSlstm)

    mea = mean_absolute_error(y_testSlstm, y_predSLSTM)
    mse = mean_squared_error(y_testSlstm, y_predSLSTM)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_testSlstm, y_predSLSTM)
    metrics = {
        'Mean Absolute Error': mea,
        'Mean Square Error': mse,
        'Root Mean Square Error': rmse,
        'R2 Score': r2
    }
    # Evaluate the model
    accuracy = r2_score(y_testSlstm, y_predSLSTM)
    
    return accuracy, metrics


def gru():
    new_df = pd.read_csv('new_df.csv')
    Xgru = new_df[['temperature', 'var1', 'pressure', 'windspeed','year','month','day','day_of_week','hour']].values
    ygru = new_df['electricity_consumption'].values

    # Define sequence length (number of previous time steps to consider)
    sequence_length = 1  # Each data point is treated independently

    # Reshape the input data into sequences
    X_seq = []
    y_seq = []

    for i in range(len(Xgru) - sequence_length):
        X_seq.append(Xgru[i:i + sequence_length])
        y_seq.append(ygru[i + sequence_length])

    X_seq = np.array(X_seq)
    y_seq = np.array(y_seq)

    X_traingru, X_testgru, y_traingru, y_testgru = train_test_split(X_seq, y_seq, test_size=0.2, random_state=42)

    # Define the single-layer GRU model
    GRUmodel = Sequential()

    # Single GRU layer
    GRUmodel.add(GRU(50, activation='relu', input_shape=(X_traingru.shape[1], X_traingru.shape[2])))

    # Output layer with 1 neuron for regression
    GRUmodel.add(Dense(1))

    # Compile the model
    GRUmodel.compile(optimizer='adam', loss='mean_squared_error')

   # Train the LSTM model
    GRUmodel.fit(X_traingru, y_traingru, epochs=100, batch_size=32, verbose=1)

    # Evaluate the model on the test set
    y_predGRU = GRUmodel.predict(X_testgru)
    
    # Make predictions on the test set
    y_pred = GRUmodel.predict(X_testgru)
    # Evaluate the model
    mea = mean_absolute_error(y_testgru, y_pred)
    mse = mean_squared_error(y_testgru, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_testgru, y_pred)
    metrics = {
        'Mean Absolute Error': mea,
        'Mean Square Error': mse,
        'Root Mean Square Error': rmse,
        'R2 Score': r2
    }
    # Evaluate the model
    accuracy = r2_score(y_testgru, y_pred)
    
    return accuracy, metrics




@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    if request.method == 'POST':
        a = float(request.form['a'])
        b = float(request.form['b'])
        c = float(request.form['c'])
        d = float(request.form['d'])
        e = float(request.form['e'])
        f = float(request.form['f'])
        g = float(request.form['g'])
        h = float(request.form['h'])
        i = float(request.form['i'])

        model = load_model('lstm_model.h5')
        input_data = np.array([[a, b, c, d, e, f, g, h, i]])
        input_data_reshaped = np.reshape(input_data, (1, input_data.shape[0], input_data.shape[1]))

        # Use the trained LSTM model to make prediction
        prediction = model.predict(input_data_reshaped)

        # Render prediction.html template with prediction value
        return render_template('prediction.html', prediction=prediction[0][0])

    return render_template('prediction.html')


@app.route('/graph')
def graph():
    return render_template('graph.html')


if __name__ == '__main__':
    app.run(debug = True)

