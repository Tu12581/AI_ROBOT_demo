from django.shortcuts import render, HttpResponse
from langmain import chunks
from vector_in import AI
import requests

# Create your views here.


def demo1(req):
    if req.method == 'POST':
        query = req.POST.get("box")  # 用name传参
        print(query)
        responce = AI(query)
        # responce = 'niumo'
        return render(req, 'temp_d.html', {"n1": responce})
    return render(req, 'temp_d.html')
