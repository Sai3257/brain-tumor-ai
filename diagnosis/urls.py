from django.urls import path
from . import views

urlpatterns = [
    path("", views.home_view, name="home"),
    path("upload/", views.upload_mri, name="upload"),
    path("dashboard-data/", views.dashboard_data),
    path("register/", views.register_view, name="register"),
    path("login/", views.login_view, name="login"),
    path("logout/", views.logout_view, name="logout"),
    path("dashboard/", views.dashboard, name="dashboard"),
    path("export-dashboard/", views.export_dashboard, name="export-dashboard"),
    
    # AI Detection Module (NEW)
    path("ai-detect/", views.ai_detect, name="ai_detect"),
    path("ai-detect-quick/", views.ai_detect_quick, name="ai_detect_quick"),
]
