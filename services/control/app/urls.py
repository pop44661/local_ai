from django.urls import path
from . import views

urlpatterns = [
    path("", views.index),               # /
    path("api/gpu/", views.gpu_info),
    path("api/docker/up/", views.docker_up),
    path("api/docker/down/", views.docker_down),
    path("api/container_gpu_stats/", views.container_gpu_stats),
    path("api/compose_state/", views.compose_state),
    path("api/container_status/", views.container_status),
    path("api/restart/service/", views.restart_service_api),
    path("api/models/get/", views.get_models_api),
    path("api/models/download/", views.download_model_api),
    path("api/models/delete/", views.delete_model_api),
    path("api/models/select/", views.select_model_api),
    path('v1/chat/completions', views.ChatCompletions.as_view()),
    path('v1/embeddings', views.Embeddings.as_view()),
    path("embed", views.Embed.as_view()),
    path('v1/audio/speech', views.SpeechSynthesis.as_view()),
    path('v1/speakers', views.CreateSpeaker.as_view()),
    path('v1/audio/transcriptions', views.Transcriptions.as_view()),
    path("license/generate/", views.generate_license_api),
    path("license/upload/", views.upload_license_api),
]