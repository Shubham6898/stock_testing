from django.contrib import admin
from django.urls import path
from Hello_app import views


urlpatterns = [
    path('', views.index,name='home'),
    path('about',views.about,name='about'),
    path('contact',views.contact,name='contact'),
    path('prediciton',views.prediciton,name='prediciton'),
    path('apple',views.apple,name='apple'),
    path('microsoft',views.microsoft,name='microsoft')
]