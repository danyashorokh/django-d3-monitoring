from django.db import models
from datetime import datetime
from scorecard.models import Scorecard, BusinessType, ClientType

# Create your models here.

class Gantt(models.Model):

    scorecard = models.ForeignKey(Scorecard, on_delete=models.SET_NULL, null=True)
    TASKS = (
        ('Анализ сегмента', 'Анализ сегмента'),
        ('Анализ индикатора', 'Анализ индикатора'),
        ('Сбор данных', 'Сбор данных'),
        ('Создание выборки', 'Создание выборки'),
        ('Создание переменных', 'Создание переменных'),
        ('Построение модели', 'Построение модели'),
        ('Дополнительный анализ', 'Дополнительный анализ'),
        ('Расчет финансового эффекта', 'Расчет финансового эффекта'),
        ('Описание модели', 'Описание модели'),
        ('Презентация модели', 'Презентация модели'),
        ('Вынос на тест', 'Вынос на тест'),
        ('Тестирование', 'Тестирование'),
        ('Вынос на prod', 'Вынос на prod'),
        ('Развертывание на 50%', 'Развертывание на 50%'),
        ('Развертывание на 100%', 'Развертывание на 100%'),
        ('Мониторинг модели', 'Мониторинг модели'),
    )
    task_name = models.CharField(max_length=100, choices=TASKS)
    # task_name = models.CharField(max_length=200, default='')
    start_date = models.DateTimeField(default=datetime.now())
    end_date = models.DateTimeField(default=datetime.now())

    STATUTES = (
        ('Ready', 'Ready'),
        ('In work', 'In work'),
        ('Is planned', 'Is planned'),
        ('Canceled', 'Canceled'),
    )
    status = models.CharField(max_length=200, choices=STATUTES)
    author = models.CharField(max_length=500, default='', blank=True)
    comment = models.CharField(max_length=500, default='', blank=True)


class DataSource(models.Model):

    SOURCES = (
        ('APPLICATION', 'APPLICATION'),
        ('ONLINE', 'ONLINE'),
        ('BEHAVIOUR', 'BEHAVIOUR'),
        ('OKB', 'OKB'),
        ('EQUIFAX', 'EQUIFAX'),
        ('NBCH', 'NBCH'),
        ('MEGAFON', 'MEGAFON'),
        ('COLLECTION', 'COLLECTION'),
        ('JUICY', 'JUICY'),

    )
    name = models.CharField(max_length=100, choices=SOURCES)
    # name = models.CharField(max_length=200, default='')
    scorecard = models.ForeignKey(Scorecard, on_delete=models.SET_NULL, null=True)
    comment = models.CharField(max_length=500, default='', blank=True)

    folder_url = models.CharField(max_length=200, default='', blank=True)
    git_url = models.CharField(max_length=200, default='', blank=True)
