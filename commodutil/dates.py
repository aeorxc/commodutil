import pandas as pd

curmon = pd.datetime.now().month
curyear = pd.datetime.now().year
curmonyear = pd.to_datetime('{}-{}-1'.format(curyear, curmon))
curmonyear_str = '{}-{}'.format(curyear, curmon) # get pandas time filtering

