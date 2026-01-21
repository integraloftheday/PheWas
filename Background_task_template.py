#!/usr/bin/env python
# coding: utf-8

# In[13]:


import glob
import gzip
import nbformat
from nbconvert.preprocessors import CellExecutionError
from nbconvert.preprocessors import ExecutePreprocessor
import os
import shutil
import tensorflow as tf
import time


# In[14]:


#---[ Change this to be the name of the notebook in the current working directory that you wish to run. ]-----------
NOTEBOOK_TO_RUN = '01_Data_Accumulation_py.ipynb'

#---[ Change the following to 'Python' if you have a Python notebook, or to 'R' if you have an R notebook.
KERNEL = 'Python'


# In[15]:


TIMESTAMP_FILE_SUFFIX = time.strftime('_%Y%m%d_%H%M%S.ipynb')
OUTPUT_NOTEBOOK = NOTEBOOK_TO_RUN.replace('.ipynb', TIMESTAMP_FILE_SUFFIX)

print(f'Executed notebook will be written to filename "{OUTPUT_NOTEBOOK}" on the local disk and the workspace bucket.')


# In[16]:


DATESTAMP = time.strftime('%Y%m%d')
HAIL_LOG_DIR_FOR_PROVENANCE = os.path.join(os.getenv('WORKSPACE_BUCKET'), 'hail-logs', DATESTAMP)

print(f'Hail logs, if any, will be copied to {HAIL_LOG_DIR_FOR_PROVENANCE}')


# In[17]:


def get_kernel(kernel):
    return 'ir' if kernel.lower() == 'r' else 'python3'

KERNEL_NAME = get_kernel(KERNEL)


# In[18]:


# See also https://nbconvert.readthedocs.io/en/latest/execute_api.html
with open(NOTEBOOK_TO_RUN) as f_in:
    nb = nbformat.read(f_in, as_version=4)
    ep = ExecutePreprocessor(timeout=-1, kernel_name=KERNEL_NAME)
    try:
        out = ep.preprocess(nb, {'metadata': {'path': ''}})
    except CellExecutionError:
        out = None
        print(f'''Error executing the notebook "{NOTEBOOK_TO_RUN}".
        See notebook "{OUTPUT_NOTEBOOK}" for the traceback.''')
    finally:
        with open(OUTPUT_NOTEBOOK, mode='w', encoding='utf-8') as f_out:
            nbformat.write(nb, f_out)
        # Save the executed notebook to the workspace bucket.
        output_notebook_path = os.path.join(os.getenv('WORKSPACE_BUCKET'), 'notebooks', OUTPUT_NOTEBOOK)
        tf.io.gfile.copy(src=OUTPUT_NOTEBOOK, dst=output_notebook_path)
        print(f'Wrote executed notebook to {output_notebook_path}')

# Save the hail logs, if any, to the workspace bucket.
for hail_log in glob.glob('hail*.log'):
    with open(hail_log, 'rb') as f_in:
        compressed_hail_log = f'{hail_log}.gz'
        with gzip.open(compressed_hail_log, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    hail_log_path = os.path.join(HAIL_LOG_DIR_FOR_PROVENANCE, compressed_hail_log)
    if not tf.io.gfile.exists(hail_log_path):
        tf.io.gfile.copy(src=compressed_hail_log, dst=hail_log_path)
        print(f'Wrote hail log to {hail_log_path}')


# In[ ]:




