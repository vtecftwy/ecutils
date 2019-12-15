# ec-utils
Collection of my utility functions for several applications.

General Use:
- ipython_utils

ML:
- fastai_utils

Algotrading and other financial:
- historical_price_configs
- historical_price_handling
- mintos_utils


Currently these utility functions are linked to other project through symlinks. 
Namely, a symlink is created in the src folder of each project using the utils.

E.g.:
- fx-bt\src\fx-utilities is a symlink



Create a symlink in Windows:
- open cmd in admin
- mklink Link Target (for a file)
- mklink /D Link Target (for a directory)

In our case, 
- mklink /D path_to_current_directory\symlink_name D:\PyProjects\ec-utils