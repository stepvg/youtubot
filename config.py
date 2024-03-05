# -*- coding: utf-8 -*-

import os, logging

logging_level						= logging.INFO		# logging.WARNING	# logging.DEBUG	# logging.INFO
logging.basicConfig( level=logging_level )

try:
	import security_keys
except ImportError:
	os.environ.setdefault('YC_FOLDER_ID', '')
	os.environ.setdefault('YC_API_KEY', '')
	os.environ.setdefault('TELEGRAM_BOT_TOKEN', '')
