"""workflow URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/1.10/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  url(r'^$', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  url(r'^$', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.conf.urls import url, include
    2. Add a URL to urlpatterns:  url(r'^blog/', include('blog.urls'))
"""
from django.conf.urls import include, url
from django.contrib import admin

from django.conf.urls.static import static
from django.conf import settings
from django.views.generic import TemplateView
from . import views
from django.views.defaults import server_error, page_not_found, permission_denied
from django.conf.urls import handler404

urlpatterns = [
	
	url(r'^$', views.index, name='index'),

    url(r'^risks/$', views.index_risks),
    url(r'^risks/stage_sankey/$', views.index_stage_sankey),
    url(r'^marketing/$', views.index_marketing),

    url(r'^admin/', admin.site.urls),
    url(r'month/', views.scorecards_per_month),  # amount per month

    url(r'one_scorecard_counts/', views.one_scorecard_counts),

    url(r'gini_badrate/', views.gini_badrate),
    url(r'gini_fpd/', views.gini_fpd),
    url(r'gini_example/', views.gini_example),

    url(r'graph_example/', views.draw_graph_example), 
    url(r'graph_contacts/', views.draw_graph_contacts),

    url(r'financed_sankey/', views.financed_sankey),

    # url(r'^timeline/', views.scorecards_timeline),

    url(r'open_now/', views.limits),
    url(r'open_now_new_repeat/', views.limits_new_repeat),
    url(r'open_now_sql/', views.limits_sql),
    
    url(r'scorecard_monitoring/', views.scorecard_monitoring),
    url(r'scorecard_monitoring_new/', views.scorecard_monitoring_new),

    url(r'vintages/', views.vintages),

    url(r'scorecard_gantt/', views.scorecard_gantt),

    url(r'^login', views.login_view),
    url(r'^logout', views.logout_view),

    url(r'^mosaic/', views.mosaic),
    
    url(r'entrance_sankey/', views.entrance_sankey),

    url(r'stage_sankey/(\w+)$', views.stage_sankey),

    url(r'monthly_report/', views.monthly_report),

    url(r'monitoring/', views.monitoring),

    url(r'map/', views.map),

    url(r'map_russia/', views.map_russia),
    url(r'map_fpd/', views.map_fpd),
    

]

# handler404 = curry(page_not_found, exception=Exception('Page not Found'), template_name='404.html')
urlpatterns.append(url(r'^.*$', views.page404))
# handler404 = views.page404

# urlpatterns.append(url(r'^accounts/', include('django.contrib.auth.urls')))

urlpatterns += [
    url(r'^accounts/', include('django.contrib.auth.urls'))
]
