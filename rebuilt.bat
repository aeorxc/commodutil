pip uninstall -y commodutil
python setup.py bdist_wheel
pip install dist\commodutil-1.0.18-py2.py3-none-any.whl
REM pip install git+https://github.com/aeorxc/commodutil#egg=commodutil