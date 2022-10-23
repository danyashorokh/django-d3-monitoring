from django.contrib import admin

# Register your models here.
from .models import ClientType, BusinessType, Scorecard

class ScorecardAdmin(admin.ModelAdmin):
    # fields = ['pub_date', 'question_text']
    list_display = ('id', 'client_type', 'business_type', 'name', 'date', 'working', 'fullname')
    list_filter = ['client_type', 'business_type', 'name', 'working', 'date']
    search_fields = ['name']

admin.site.register(Scorecard, ScorecardAdmin)
admin.site.register(ClientType)
admin.site.register(BusinessType)
# admin.site.register(Scorecard)