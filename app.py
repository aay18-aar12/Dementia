from flask import Flask
import requests
import math

app = Flask(__name__)

@app.route('/all')
def hello_world():

    request2 = requests.get('https://maps2.dcgis.dc.gov/dcgis/rest/services/DCGIS_DATA/Health_WebMercator/MapServer/16/query?where=1%3D1&outFields=*&outSR=4326&f=json')

    return request2.json()


@app.route('/health')
def retHealth():
    return 'Web Service is Working'
