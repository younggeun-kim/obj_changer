from django.shortcuts import render, redirect
from base_api.settings import STATIC_ROOT

from rest_framework.response import Response
from rest_framework.decorators import api_view
import os
import json
from generate import inference

from .functions.candidate_gen import *
# Create your views here.

default_data_dir=os.path.join(STATIC_ROOT, 'question.csv')
@api_view(['POST'])
def query_view(request):
    query=request.data['query']
    print(query)
    indices, distances, corres_texts=run(data_dir=default_data_dir, query=query, length=20)
    #format indices, data into JSON
    print(indices, distances, corres_texts)
    response={
        'indices': indices[0],
        'distances': distances[0],
        'corresponding_texts': corres_texts,
    }
    return Response(response)

@api_view(['GET'])
def api_overview(request):
    api_urls={
        '/': 'overview',
        'query/': 'Make query w.r.t database',
    }
    return Response(api_urls)
