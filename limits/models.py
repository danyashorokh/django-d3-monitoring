from django.db import models
from datetime import datetime

# Create your models here.

class SegmentInfo(models.Model):

    CLIENT_TYPES = (
        ('Новый', 'Новый'),
        ('Повторный', 'Повторный'),
    )

    BUSINESS_TYPES = (
        ('Core', 'Core'),
        ('Digital', 'Digital'),
        ('POS', 'POS'),
        ('MigOne', 'MigOne'),
        ('contact_center', 'contact_center'),
        ('Upsale','Upsale'),
        ('Upsale_Online','Upsale_Online'),
        ('AutoLoan','AutoLoan'),
        ('SMS_loan','SMS_loan'),
        ('Telegram_loan','Telegram_loan'),
        ('PTS_loan','PTS_loan')
    )

    client_type = models.CharField(max_length=100, choices=CLIENT_TYPES, default=CLIENT_TYPES[0][0])
    business_type = models.CharField(max_length=100, choices=BUSINESS_TYPES, default=BUSINESS_TYPES[0][0])

    spr = models.CharField(max_length=100, default='', blank=True)
    execution_key = models.CharField(max_length=500, default='', blank=True)

    share_a = models.FloatField(default=0)
    share_b = models.FloatField(default=0)
    share_c = models.FloatField(default=0)
    share_d = models.FloatField(default=0)

    limit_a = models.FloatField(default=0)
    limit_b = models.FloatField(default=0)
    limit_c = models.FloatField(default=0)
    limit_d = models.FloatField(default=0)

    active = models.BooleanField(default=True)
    date = models.DateField(auto_now=True)

    def __str__(self):
        return '%s_%s_%s' % (self.business_type, self.client_type, self.execution_key)

    add_info = models.CharField(max_length=500, default='', blank=True)

    def __unicode__(self):
        return '%s_%s_%s' % (self.business_type, self.client_type, self.execution_key)

    def get_label(self):
        return '%s_%s_%s' % (self.business_type, self.client_type, self.execution_key.replace(',','_'))
    
    label = property(get_label)
    scorecards = models.CharField(max_length=1000, default='', blank=True)