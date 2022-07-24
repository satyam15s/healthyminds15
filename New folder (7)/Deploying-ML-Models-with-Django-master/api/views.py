#Import necessary libraries
import sounddevice as sd
from scipy.io.wavfile import write

fs = 44100  # Sample rate
seconds = 3  # Duration of recording

myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=2)
sd.wait()  # Wait until recording is finished
write('output.wav', fs, myrecording)  # Save as WAV file 
import pickle
from rest_framework.decorators import api_view
from rest_framework.response import Response
from django.shortcuts import render
import numpy as np

# Create your views here.
@api_view(['GET'])
def index_page(request):
    return_data = {
        "error" : "0",
        "message" : "Successful",
    }
    return Response(return_data)

@api_view(["POST"])
def predict_diabetictype(request):
    try:
        myrecording= write('output.wav', fs, myrecording)  # Save as WAV file 
            #Passing data to model & loading the model from disks
        model_path = 'ml_model/Data_noiseNshift.h5'
        classifier = pickle.load(open(model_path, 'rb'))
        prediction = classifier.predict([result])[0]
        conf_score =  np.max(classifier.predict_proba([result]))*100
        predictions = {
                'error' : '0',
                'message' : 'Successfull',
                'prediction' : prediction,
                'confidence_score' : conf_score
            }
    
    except Exception as e:
        predictions = {
            'error' : '2',
            "message": str(e)
        }
    
    return Response(predictions)
