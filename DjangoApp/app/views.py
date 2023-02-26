

from django.shortcuts import render
import joblib
import pandas as pd

loadModel = joblib.load('./models/RFModelforEducation.pkl')

def index(request):

    return render(request, 'index.html')


def graduateGuessing(request):
    if request.method == 'POST':
        mezunSayisi = request.POST.get('mezun')
        temp = {}
        temp['Mezun_ogr_say'] = mezunSayisi
        testData=pd.DataFrame({'x':temp}).transpose()
        result = loadModel.predict(testData)[0]
        context = {'result': result}
    return render(request, 'index.html', context)
