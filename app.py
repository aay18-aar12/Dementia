from flask import Flask
import requests
import math

app = Flask(__name__)

@app.route('/all')
def hello_world():

    request2 = requests.get('https://maps2.dcgis.dc.gov/dcgis/rest/services/DCGIS_DATA/Health_WebMercator/MapServer/32/query?where=1%3D1&outFields=ID2,NAME,ID,TOTALPOP,P_POC,P_CHILD,P_ELDERLY,P_POVERTY,P_DISABILITY,P_LIMENG,ASTHMA,CHD,OBESITY,HSI,P_TREECOVER,P_NOTREE,P_IMPSURF,AIRTEMP_MEAN,HEI,HSEI,CREATED,EDITED&returnGeometry=false&outSR=4326&f=json')

    return request2.json()


@app.route('/health')
def retHealth():
    return 'Web Service is Working'
