from django.urls import path
from . import views

urlpatterns = [
    path('login/', views.custom_login_view, name='login'),
    path('logout/', views.custom_logout_view, name='logout'),
    path('dashboard/', views.dashboard_view, name='dashboard'),
    path('upload/', views.upload_file_view, name='upload_file'),
    path('tables/', views.tables_view, name='tables'),
    path('analysis/', views.analysis_view, name='analysis'),
    path('messaging/', views.messaging_view, name='messaging'),
    path('api/exchange_rates/', views.exchange_rates_api, name='exchange_rates_api'),
    path('', views.dashboard_view, name='home'), # Redirect root to dashboard
]