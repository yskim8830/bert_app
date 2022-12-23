from django.urls import path
from bert_app import views
urlpatterns = [
    path('learn', views.train.as_view()),
    path('learn/', views.train.as_view()),
    path('question', views.question.as_view()),
    path('question/', views.question.as_view()), 
    path('question2', views.question2.as_view()),
    path('question2/', views.question2.as_view()), 
    path('convert', views.convert.as_view()), 
    path('convert/', views.convert.as_view()), 
    path('test/request/', views.test_request.as_view()), 
] 