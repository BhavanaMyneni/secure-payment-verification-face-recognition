from django.urls import path

from . import views

urlpatterns = [
    path("verify/face/", views.verify_face, name="verify_face"),
    path("verify/otp/", views.verify_otp_code, name="verify_otp"),
]
