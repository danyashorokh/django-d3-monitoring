from django.db import models
from datetime import datetime
# Create your models here.
from django.contrib.auth import get_user_model
# from django.contrib.auth.models import User

# from django.conf import settings
# USER_MODEL = getattr(settings, 'AUTH_USER_MODEL', User)

User = get_user_model()

class Activity(models.Model):

	# user = models.ForeignKey(User, on_delete=models.CASCADE)
	
	user = models.ForeignKey(get_user_model(), on_delete=models.CASCADE)
	# user = models.OneToOneField(User, on_delete=models.CASCADE)
	# user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)

	# user = models.CharField(max_length=500, default='') # работает
	
	event_date = models.DateTimeField(default=datetime.now)

	url = models.CharField(max_length=500, default='', blank=True)
	ip = models.CharField(max_length=500, default='', blank=True)
	server_name = models.CharField(max_length=500, default='', blank=True)

	is_staff = models.BooleanField(default=False)
	is_superuser = models.BooleanField(default=False)

	can_export = models.BooleanField(default=False)

	groups = models.CharField(max_length=1000, default='', blank=True)