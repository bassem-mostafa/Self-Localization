#
# Objective: Make use of surrounding objects for self-localization in GPS not available areas
#

try:
    import os                   as OS
    import time                 as TIME
    import cv2                  as CV

    _VERSION = ['{:04}{:02}{:02}-{:02}{:02}'.format( *( _VERSION.tm_year, _VERSION.tm_mon, _VERSION.tm_mday, _VERSION.tm_hour, _VERSION.tm_min ) ) for _VERSION in [ TIME.localtime( OS.path.getmtime( __file__ ) ) ] ][0]

    print( "{:<20} {}".format( "Script version", _VERSION ) )
    print( "{:<20} {}".format( "OpenCV version", CV.__version__ ) )

# Note: Nothing Should Be Done After This Point
    print( "\n{}".format( "Success" ) )
except Exception as e:
    print( "\n{}".format( "Failed" ) )
    print( "\n{}".format( e ) )
