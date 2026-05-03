"""
WSGI config for brain_tumor_ai project.

It exposes the WSGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/5.2/howto/deployment/wsgi/
"""

import os

# IMPORTANT: Set TensorFlow environment variables BEFORE Django imports
# This prevents recursion errors with Python 3.13
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from django.core.wsgi import get_wsgi_application

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'brain_tumor_ai.settings')

application = get_wsgi_application()
