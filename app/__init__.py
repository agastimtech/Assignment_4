import os
import aiohttp_jinja2
from app.manage import create_app
from aiohttp import web
import jinja2

app= create_app()

from app.resource.predict import HomeView,PredictView

app.router.add_view('/', HomeView)
app.router.add_view('/predict',PredictView)
