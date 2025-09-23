from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('admin/', admin.site.urls),
    path('summarize/', include('summarize.urls')),
    path('query/', include('query.urls')),
    path('upload/', include('upload_handler.urls')),
    path('ocr/', include('ocr.urls')),
]
