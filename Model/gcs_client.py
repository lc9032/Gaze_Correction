import socket
import cv2 # type: ignore
import pickle
import struct


class GazeCorrSys_client():
    def __init__(self):
        self.width = 640
        self.height = 480
        pass

    def client(self, host='127.0.0.1', port=9999):
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.connect((host, port))
        video_capture = cv2.VideoCapture(0)  # Change to video file path if needed

        video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        while True:
            ret, frame = video_capture.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            cv2.imshow('Original Frame', frame)

            data = pickle.dumps(frame)
            message_size = struct.pack("Q", len(data))
            client_socket.sendall(message_size + data)

            data = b""
            payload_size = struct.calcsize("Q")
            while len(data) < payload_size:
                packet = client_socket.recv(4 * 1024)  # 4K
                if not packet: break
                data += packet
            packed_msg_size = data[:payload_size]
            data = data[payload_size:]
            msg_size = struct.unpack("Q", packed_msg_size)[0]

            while len(data) < msg_size:
                data += client_socket.recv(4 * 1024)
            frame_data = data[:msg_size]
            frame = pickle.loads(frame_data)

            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.imshow('Processed Frame', frame_bgr)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


        video_capture.release()
        cv2.destroyAllWindows()
        client_socket.close()


# if __name__ == '__main__':
#     gc = GazeCorrSys_client()
#     gc.client()