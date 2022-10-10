from django.urls import path, include
from .views import *

urlpatterns=[
    path('', api_overview, name="overview"),
    path('query/', query_view, name='query'),
]