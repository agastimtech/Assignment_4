import asyncio
import aiohttp.web
import pandas as pd
import numpy as np
import pickle
# from app.model.model import load_model
from aiohttp import web
import aiohttp_jinja2
import json
from decimal import Decimal

async def predict_score(model, study_hours):
    study_hours_numeric = np.array([float(hour) for hour in study_hours])
    scores = np.round(model.predict(study_hours_numeric.reshape(-1, 1)))
    return scores.tolist()  # Convert ndarray to list

class HomeView(web.View):
    @aiohttp_jinja2.template('index.html')
    async def get(self):
        return {}

class PredictView(web.View):
    async def post(self):
        data = await self.request.json()
        study_hours = data.get('study_hours', [])

        if not isinstance(study_hours, list):
            study_hours = [study_hours]

        with open(r'/home/agasti/Desktop/Assignment 4/app/model/model_pkl', 'rb') as file:
            pickled_model = pickle.load(file)

        scores = await predict_score(pickled_model, study_hours)

        return web.Response(text=json.dumps({'scores': scores}), content_type='application/json')








