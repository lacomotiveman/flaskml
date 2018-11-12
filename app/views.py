from flask import render_template
from app import app
import pandas
import pickle
import os

@app.route('/')
@app.route('/index')
def index():
    return render_template("index.html")

'''
порядок исходных параметров:
	"fixed acidity";
	"volatile acidity";
	"citric acid";
	"residual sugar";
	"chlorides";
	"free sulfur dioxide";
	"total sulfur dioxide";
	"density";
	"pH";
	"sulphates";
	"alcohol";
надо будет по загруженной модели предсказать "quality"
'''
# будет работать по строке в браузере /wine/0.1/0.2/0.3/0.4/0.5/0.6/0.7/0.8/0.9/0.10/0.11
@app.route('/wine/<float:param1>/<float:param2>/<float:param3>/<float:param4>/<float:param5>/<float:param6>/<float:param7>/<float:param8>/<float:param9>/<float:param10>/<float:param11>')
def wine(param1,param2,param3,param4,param5,param6,param7,param8,param9,param10,param11):
	filename = os.path.join(app.root_path, 'RF_clf.sav')
	#filename = 'RF_clf.sav'
	loaded_model = pickle.load(open(filename, 'rb'))  # ожидается RF_clf
	d = [param1,param2,param3,param4,param5,param6,param7,param8,param9,param10,param11]
	df = pandas.DataFrame({'fixed acidity':[d[0]], 'volatile acidity':[d[1]], 'citric acid':[d[2]], 'residual sugar':[d[3]], 'chlorides' :[d[4]], 'free sulfur dioxide':[d[5]], 'total sulfur dioxide':[d[6]], 'density':[d[7]], 'pH':[d[8]], 'sulphates':[d[9]], 'alcohol':[d[10]] })
	result = loaded_model.predict(df)
	return  '{ \"RESULT\":\"'+ str(result[0])+'\"}'
