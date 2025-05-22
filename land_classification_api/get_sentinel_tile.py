import os
import ee
import requests
from PIL import Image
from io import BytesIO

# Authenticate Earth Engine with service account JSON
service_account = 'earthengine-access@farmwise-classifier.iam.gserviceaccount.com'
json_path = os.path.join(os.path.dirname(__file__), 'farmwise-classifier-cb9d3d442e98.json')
credentials = ee.ServiceAccountCredentials(service_account, json_path)
ee.Initialize(credentials)

def get_sentinel_tile(lat, lon, size=64):
    point = ee.Geometry.Point([lon, lat])
    collection = (
        ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
        .filterBounds(point)
        .filterDate('2022-01-01', '2022-12-31')
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 10))
        .median()
        .select(['B4', 'B3', 'B2'])
    )
    vis = {
        'bands': ['B4', 'B3', 'B2'],
        'min': 0,
        'max': 3000,
        'dimensions': size,
        'format': 'png',
        'region': point.buffer(300).bounds()
    }
    url = collection.getThumbURL(vis)
    resp = requests.get(url)
    return Image.open(BytesIO(resp.content))