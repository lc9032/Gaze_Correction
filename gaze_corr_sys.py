import time
from multiprocessing import Process

from Model.gcs_server import GazeCorrSys_server
from Model.gcs_client import GazeCorrSys_client

def run_server():
    gcsServer = GazeCorrSys_server()
    gcsServer.run_server()

def run_client():
    gcsClient = GazeCorrSys_client()
    gcsClient.client()

def process_video():
    gcsServer = GazeCorrSys_server()
    gcsServer.run_mp4()


if __name__ == '__main__':
    # server_thread = Process(target=run_server)
    # client_thread = Process(target=run_client)

    # server_thread.start()
    # time.sleep(1) 
    # client_thread.start()
    
    # server_thread.join()
    # client_thread.join()

    video_thread = Process(target=process_video)
    video_thread.start()
    video_thread.join()