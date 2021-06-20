import os, io
from draw_vertice import drawVertices
from google.cloud import vision_v1
import pandas as pd

#os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r'ServiceAccountToken.json'
client = vision_v1.ImageAnnotatorClient()

file_name = 'google.png'
image_folder = '~/Desktop/'
image_path = os.path.join(image_folder, file_name)

with io.open(image_path, 'rb') as image_file:
	content = image_file.read()

image = vision_v1.types.Image(content=content)
response = client.logo_detection(image=image)
logos = response.logo_annotations

for logo in logos:
	print('Logo Description:', logo.description)
	print('Confidence Score:', logo.score)
	print('-' * 50)
	vertices = logo.bounding_poly.vertices
	print('Vertices value {}'.format(vertices))
	
	drawVertices(content, vertices, logo.description)

