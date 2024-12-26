from django.shortcuts import render, HttpResponse
# from langmain import chunks
from vector_in import AI
import requests
import os
from history_vector import his_AI
from langchain_community.document_loaders import TextLoader


# Create your views here.


def demo1(req):
    if req.method == 'POST':
        query = req.POST.get("box")  # 用name传参
        search_choice = req.POST.get("search_choice")
        print(query)
        related_file = req.FILES.get("related_filename",None)
        if related_file:
        # 从获取的对象中获得文件名，用于写到本地
            file_name = related_file.name
            # 将提取的文件写到本地，使用w写法而非wb二进制写法，采用utf-8编码
            with open(file_name, "w", encoding='utf-8') as f:
                f.write(related_file.read().decode("utf-8")) #将二进制比特流解码为utf-8
            # 使用二进制数据流的写法
            # with open(file_name, "wb") as f:
                # f.write(related_file.read())
            # related_file = req.FILES["related_filename"].chunks()
        else:
            file_name = None
        responce = his_AI(query,file_name,search_choice) + '喵~'
        print(responce)
        if search_choice == 'mode1':
            print(related_file, 'hjkhj')
        # responce = 'niumo'
        return render(req, 'temp_d.html', {"n1": responce})
    return render(req, 'temp_d.html')
