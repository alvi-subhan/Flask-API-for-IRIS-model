import numpy as np
import io
import tensorflow as tf
import keras
from keras import backend as K
from keras.models import Sequential
from keras.models import load_model
from flask import Flask
from flask import jsonify, render_template
from flask import request
from flask import abort



app=Flask(__name__)

def get_model():
    global model 
    model=load_model("new_model.h5")
    # this is key : save the graph after loading the model
    global graph
    graph = tf.get_default_graph()

    print("* Iris Model loaded!")

def preprocess_input(array_input): #yahan pe array dalni h inputs ki
    array_input=np.array(array_input)
    if array_input.shape != (1,4):
        array_input=array_input.reshape(1,4)
    return array_input

print("loading the keras model")

get_model()

tasks=[1,2,3]
print(tasks)

@app.route("/",methods=["GET","POST"])
def send():
    #post me server request ko post krega while in get we are getting the data from the user/client
    if request.method=="POST":
        l1=request.form["l1"]
        l2=request.form["l2"]
        l3=request.form["l3"]
        l4=request.form["l4"]
        reshape_inputs=np.array([l1,l2,l3,l4]).reshape(1,4) #reshaping the array
        with graph.as_default():
        #preds = model.predict(image)
            prediction=model.predict(reshape_inputs) #predicting the flower
        #K.clear_session()
        correct_arg_prediction=np.argmax(prediction[0]) #taking the argument of the max probabiltiy
        prediction=prediction.tolist() #converting to the list
        prediction=prediction[0][correct_arg_prediction] #probability of the expected output
        
        flowers=['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
        
        name_of_flower=flowers[correct_arg_prediction] #name of predicted flower


        return render_template("prediction.html", features=name_of_flower)
    
    return render_template("index.html") #receiving the data from the user (GET request)


@app.route('/iris/api/v1.0/<int:sl>/<int:sw>/<int:pl>/<int:pw>',methods=['GET'])
def get_task(sl,sw,pl,pw):
    reshape_inputs2=np.array([sl,sw,pl,pw]).reshape(1,4) #reshaping the array
    if len(reshape_inputs2[0]) != 4:
        abort(404)
    with graph.as_default():
        #preds = model.predict(image)
        prediction=model.predict(reshape_inputs2) #predicting the flower
        #K.clear_session()
    correct_arg_prediction=np.argmax(prediction[0]) #taking the argument of the max probabiltiy
    prediction=prediction.tolist() #converting to the list
    prediction=prediction[0][correct_arg_prediction] #probability of the expected output
        
    flowers=['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
        
    name_of_flower=flowers[correct_arg_prediction] #name of predicted flower

        
    return jsonify({'name of flower': name_of_flower})    


if __name__ == '__main__':
      app.run(host='0.0.0.0', port=5000,debug=True)
