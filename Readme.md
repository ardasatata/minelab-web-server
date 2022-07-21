# Project structure

 * [classifier](./classifier) Deep Learning Module
 * [client](./client) Front-end Module
 * [deleted](./deleted) Erhu deleted video temp
   * video1.mov
   * video2.mov
   * *.mov
 * [predict](./predict) Erhu predicted video
   * video1.mov
   * video2.mov
   * *.mov
 * [processed](./processed) Erhu temp processed files
 * [server](./server) Flask server directory
   * [FaceMosaicMediaPipe.py](./server/FaceMosaicMediaPipe.py) Face mosaic script
   * [server.py](./server/server.py) Main server file *
   * [streaming.py](./server/streaming.py) Streaming server file *
 * [sign_language](./sign_language) Sign Language server related files *
 * [sliced](./sliced) Erhu video temp folder
 * [README.md](./README.md)
 * [upload](./upload) Erhu uploaded video directory

## Serve frontend
1.Install NODE & NPM by following
[this](https://nodejs.org/en/download/)

2.After installing Node & NPM please follow this command
```bash
cd client
npm install
npm start
```

## Serve main backend & streaming service
1. Main backend
```bash
cd server
python server.py
```

2. Streaming service
```bash
cd server
python streaming.py
```

## Things that need to pay attention ⚠️
1. Python >= 3.7
2. MM-Pose == 0.25.0
3. MM-Detection == 2.23.0
4. mediapipe == 0.8.9.1
5. Flask == 2.1.2
6. CUDA == 11.6
7. HTTPS certificate files on `./server/*.key` & `./server/*.crt`