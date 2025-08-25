from django.urls import path
from . import views

urlpatterns = [
    path('', views.dashboard_view, name='dashboard'),
    path('upload/', views.upload_file_view, name='upload_file'),
    path('tables/', views.tables_view, name='tables'),
    path('analysis/', views.analysis_view, name='analysis'),
    path('messaging/', views.messaging_view, name='messaging'),
    path('data_downloads/', views.data_downloads_view, name='data_downloads'),
]