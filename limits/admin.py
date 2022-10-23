from django.contrib import admin

# Register your models here.
from .models import SegmentInfo

class SegmentInfoAdmin(admin.ModelAdmin):
    # fields = ['pub_date', 'question_text']
    list_display = ('id', 'client_type', 'business_type', 'spr', 'execution_key', 'share_a', 'share_b', 
    	'share_c', 'share_d', 'limit_a', 'limit_b', 'limit_c', 'limit_d', 'active', 'date', 'scorecards', 'add_info')
   
    list_filter = ['client_type', 'business_type', 'spr', 'execution_key']
    search_fields = ['business_type']

admin.site.register(SegmentInfo, SegmentInfoAdmin)