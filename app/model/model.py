from app.common.utils import save_model
import pickle
import asyncio

def load_model():
    pickled_model = pickle.load(open(save_model(), 'rb'))
    return pickled_model