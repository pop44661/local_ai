from django.apps import AppConfig
import os

class AppConfig(AppConfig):
    name = "app"

    def ready(self):
        if os.environ.get("RUN_MAIN") != "true":
            return

        from .model_init import init_models
        init_models()