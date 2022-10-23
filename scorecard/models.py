# -*- coding: utf-8 -*-
from django.db import models
from datetime import datetime


# Create your models here.

class ClientType(models.Model):
    name = models.CharField(max_length=100, default='')
    def __str__(self):
        return self.name


class BusinessType(models.Model):
    # client_type = models.ForeignKey(ClientType, on_delete=models.SET_NULL, null=True)
    name = models.CharField(max_length=100, default='')

    def __str__(self):
        return self.name

class Scorecard(models.Model):

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
    )

    # client_type = models.ForeignKey(ClientType, on_delete=models.SET_NULL, null=True)
    client_type = models.CharField(max_length=100, choices=CLIENT_TYPES, default=CLIENT_TYPES[0][0])
    # business_type = models.ForeignKey(BusinessType, on_delete=models.SET_NULL, null=True)
    business_type = models.CharField(max_length=100, choices=BUSINESS_TYPES, default=BUSINESS_TYPES[0][0])
    name = models.CharField(max_length=100)
    date = models.DateField(auto_now=True)
    working = models.BooleanField(default=True)
    # date = models.CharField(max_length=100, default=datetime.strftime(datetime.now(), "%Y_%m.%d"))
    def __str__(self):
        return '%s (%s %s)' % (self.name, self.business_type, self.client_type)

    def get_full_name(self):
        # Returns the person's full name."
        return '%s (%s %s)' % (self.name, self.business_type, self.client_type)
    
    fullname = property(get_full_name)

    add_info = models.CharField(max_length=500, default='', blank=True)

    def __unicode__(self):
        return '%s (%s %s)' % (self.name, self.business_type, self.client_type)