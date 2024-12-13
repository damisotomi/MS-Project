import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/sotomi/ms_project/src/vfm_pkg/install/vfm_pkg'
