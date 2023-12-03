import melee
import enet
import stat
import shutil
import threading
import time

# Modified version of the code from the libmelee library, since that code as provided doesn't work.
# Taken from slippstream.py/shutdown() and console.py/stop()
def stop(console: melee.Console):
    """Do NOT try using your Console after using this function. Instantiate a new one."""
    console.connected = False
    # Close down the socket and connection to the console
    if console._slippstream._peer:
        console._slippstream._peer.send(0, enet.Packet())
        console._slippstream._host.service(100)
        console._slippstream._peer.disconnect()
        console._slippstream._peer = None

    if console._slippstream._host:
        console._slippstream._host = None

    if console._process is not None:
        console._process.terminate()
        console._process = None
    
    if console.temp_dir:
        # Temp files are still being used according to the OS. Don't want to just sleep this, so we'll do it on
        # another thread.
        thread = threading.Timer(3, shutil.rmtree, [console.temp_dir])
        thread.start()
        console.temp_dir = None