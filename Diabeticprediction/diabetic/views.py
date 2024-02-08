from django.shortcuts import render,redirect
from django.http import HttpRequest,HttpResponse
from django.views.decorators.csrf import csrf_protect

import pandas as pd
import numpy as np
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
# Create your views here.
def home(request):
    return render(request,"home.html")
def predict(request):
    return render(request, 'predict.html')

@csrf_protect
def result(request):
    result1 = None  # Initialize with a default value

    if request.method == 'POST':
        try:
            # Assuming you have a form with appropriate input names (n1, n2, ..., n8)
            val1 = float(request.POST.get("n1",0.0))
            val2 = float(request.POST.get("n2",0.0))
            val3 = float(request.POST.get("n3",0.0))
            val4 = float(request.POST.get("n4",0.0))
            val5 = float(request.POST.get("n5",0.0))
            val6 = float(request.POST.get("n6",0.0))
            val7 = float(request.POST.get("n7",0.0))
            val8 = float(request.POST.get("n8",0.0))

            
            data = pd.read_csv(r"C:\Users\harsh\Desktop\MY PROJECT3\diabetes.csv")

            x = data.drop("Outcome", axis=1)
            y = data["Outcome"]

            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

            model = LogisticRegression()
            y_train_flatten = y_train.values.flatten()
            model.fit(x_train, y_train_flatten)

            # Make predictions
            prediction = model.predict([[val1, val2, val3, val4, val5, val6, val7, val8]])

            if prediction == [1]:
                result1 = "Diabetic"
            elif prediction == [0]:
                result1 = "Non-Diabetic"

        except ValueError:
            result1 = "Invalid input: Please enter valid numeric values for all fields."

    return render(request, "predict.html",{'result1': result1})
def about(request):
    return render(request,"about.html")
