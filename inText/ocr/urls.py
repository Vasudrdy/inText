from django.urls import path
from . import views

urlpatterns = [
    path('extract_text/', views.extract_text_from_image, name='extract_text_from_image'),
]