from django.contrib import admin

# Register your models here.
from .models import Gantt, DataSource

class GanttAdmin(admin.ModelAdmin):
    # fields = ['pub_date', 'question_text']
    list_display = ('id', 'scorecard', 'task_name', 'start_date', 'end_date', 'status', 'author', 'comment', 'scorecard_id')
    list_filter = ['scorecard', 'task_name', 'start_date', 'end_date', 'status', 'author']
    search_fields = ['task_name']

class DataSourceAdmin(admin.ModelAdmin):
    # fields = ['pub_date', 'question_text']
    list_display = ('name', 'scorecard', 'comment')
    list_filter = ['name', 'scorecard']
    search_fields = ['name']

admin.site.register(Gantt, GanttAdmin)
admin.site.register(DataSource, DataSourceAdmin)