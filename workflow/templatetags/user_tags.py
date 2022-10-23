# -*- coding:utf-8 -*-
from __future__ import unicode_literals

# Stdlib imports

# Core Django imports
from django import template
from django.contrib.auth.models import Group
# Third-party app imports

# Realative imports of the 'app-name' package


register = template.Library()


@register.filter('has_group')
def has_group(user, group_name):
    """
    Verifica se este usuário pertence a um grupo
    """
    groups = user.groups.all().values_list('name', flat=True)
    return True if group_name in groups else False


@register.filter
def get_item(dictionary, key):
    return dictionary.get(key)