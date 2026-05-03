from django.contrib import admin
from .models import MRIUpload, UserProfile

@admin.register(MRIUpload)
class MRIUploadAdmin(admin.ModelAdmin):
    list_display = ('tumor_type', 'confidence', 'created_at')
    list_filter = ('tumor_type',)
    ordering = ('-created_at',)

@admin.register(UserProfile)
class UserProfileAdmin(admin.ModelAdmin):
    list_display = ('user', 'role')
    list_filter = ('role',)
