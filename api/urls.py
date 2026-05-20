from django.urls import path
from .views import predict_review

urlpatterns = [
    path("predict/", predict_review, name="predict_review"),
]