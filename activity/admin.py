from django.contrib import admin

# Register your models here.
from .models import Activity

class ActivityAdmin(admin.ModelAdmin):
    # fields = ['pub_date', 'question_text']
    list_display = ('id', 'user', 'url', 'event_date', 'ip', 
    	'server_name', 'is_staff', 'is_superuser', 'can_export', 'groups')
    list_filter = ['user', 'url', 'event_date', 'server_name']
    search_fields = ['user']


admin.site.register(Activity, ActivityAdmin)