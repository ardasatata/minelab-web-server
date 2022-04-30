import React, { useState } from 'react';
// import { Api } from './api';

import LoadingOverlay from 'react-loading-overlay';
import BounceLoader from 'react-spinners/BounceLoader';

import VideoRecorder from 'react-video-recorder';
import { Helmet } from 'react-helmet-async';
import { NavBar } from '../../components/NavBar';
import { PageWrapperMain } from '../RecordVideo';
import axios from 'axios';

const RECORDER_TIME_CONFIG = {
  countdownTime: 3000,
  timeLimit: 8000,
};

function Collection() {
  const [prediction, setPrediction] = useState(['Press Record First!']);

  const [isLoading, setIsloading] = useState(false);

  const [filename, setFilename] = useState(null);
  const [duration, setDuration] = useState(1);

  const [videoBlob, setVideoBlob] = useState(null);

  const renderVideoRecorder = () => {
    if (filename === null || filename === '') {
      return (
        <div className="font-bold text-xl text-white">
          {'Please input the file name first ==>'}
        </div>
      );
    }
    return (
      <div
        style={{ aspectRatio: '4/3', minHeight: '40em' }}
        className="p-2 bg-black rounded-md"
      >
        <VideoRecorder
          isReplayingVideo={false}
          showReplayControls={true}
          isOnInitially={true}
          countdownTime={RECORDER_TIME_CONFIG.countdownTime}
          // timeLimit={RECORDER_TIME_CONFIG.timeLimit}
          timeLimit={duration * 1000}
          onRecordingComplete={async videoBlob => {
            // Do something with the video...
            setPrediction(['Uploading your video']);
            // await uploadVideo(
            //   videoBlob,
            //   selectedOption.value,
            //   selectedSubject.value,
            // );
            console.log('videoBlob', videoBlob);
            setVideoBlob(videoBlob);
          }}
          onStartRecording={() => {
            setPrediction(['Recording video...']);
          }}
          onTurnOnCamera={() => {
            setPrediction(['Press Record First!']);
          }}
        />
      </div>
    );
  };

  const reset = async () => {
    window.location.reload();
  };

  const startUpload = async () => {
    await uploadFile(videoBlob, filename);
  };

  const uploadFile = async (blob, filename) => {
    const headers = {
      'Content-Type': 'multipart/form-data',
      'Access-Control-Allow-Origin': '*',
    };

    const formData = new FormData();
    // @ts-ignore
    formData.append('video', blob, `${filename}.mp4`);

    const result = await axios.post(
      'https://140.115.51.243/api/send-video',
      formData,
      { headers },
    );

    console.log(result);

    if (result.data.ok) {
      // setIsUploaded(true);
      // setFile(null);
    } else {
      // setIsError(true);
      // setIsUploaded(false);
    }
  };

  return (
    <>
      <Helmet>
        <title>Home Page</title>
        <meta
          name="description"
          content="A React Boilerplate application homepage"
        />
      </Helmet>
      <NavBar />
      <PageWrapperMain>
        <div className="flex flex-row h-screen w-full">
          {/*<Sidebar />*/}
          <div className="flex flex-col text-gray-800 bg-black w-full">
            <LoadingOverlay
              active={isLoading}
              spinner={<BounceLoader color={'lightblue'} />}
            >
              <div className="flex flex-row justify-center items-center h-screen">
                <div className="flex w-2/3 items-center justify-center ">
                  {renderVideoRecorder()}
                </div>
                <div className="flex flex-col w-1/3">
                  <div className="max-w-md py-4 px-8 bg-white shadow-lg rounded-lg my-20 pb-8">
                    <div>
                      <h3 className="text-gray-800 text-3xl font-semibold mb-2">
                        Settings
                      </h3>
                      {/*<p className="mt-2 text-gray-600 mb-8 text-red-500">*/}
                      {/*  Make sure you already select the sentence number and*/}
                      {/*  subject number before record the video!*/}
                      {/*</p>*/}
                      <h3 className="text-gray-800 text-xl font-semibold mb-2">
                        File Name
                      </h3>
                      <div className="mt-1 relative rounded-md shadow-sm mb-4">
                        <input
                          type="text"
                          className="focus:ring-indigo-500 focus:border-indigo-500 block w-full pl-2 pr-12 sm:text-lg border-gray-300 rounded-md"
                          placeholder="please enter the file name..."
                          value={filename}
                          onChange={event => {
                            setFilename(event.target.value);
                          }}
                        />
                        <div className="absolute inset-y-0 right-0 pr-3 flex items-center pointer-events-none">
                          <span
                            className="text-gray-500 sm:text-lg"
                            id="price-currency"
                          >
                            .mp4
                          </span>
                        </div>
                      </div>

                      <h3 className="text-gray-800 text-xl font-semibold mb-2">
                        Duration
                      </h3>
                      <div className="mt-1 relative rounded-md shadow-sm">
                        <input
                          type="text"
                          className="focus:ring-indigo-500 focus:border-indigo-500 block w-full pl-2 pr-12 sm:text-lg border-gray-300 rounded-md"
                          placeholder="please enter duration "
                          value={duration}
                          onChange={event => {
                            // @ts-ignore
                            setDuration(event.target.value);
                          }}
                        />
                        <div className="absolute inset-y-0 right-0 pr-3 flex items-center pointer-events-none">
                          <span
                            className="text-gray-500 sm:text-lg"
                            id="price-currency"
                          >
                            seconds
                          </span>
                        </div>
                      </div>
                    </div>
                    <div className="flex items-center justify-between mt-8">
                      {videoBlob ? (
                        <button
                          onClick={startUpload}
                          className="bg-green-500 hover:bg-green-700 text-white font-bold py-2 px-4 rounded focus:outline-none focus:shadow-outline"
                          type="button"
                        >
                          Upload Video
                        </button>
                      ) : null}
                      <button
                        onClick={reset}
                        className="bg-red-500 hover:bg-red-700 text-white font-bold py-2 px-4 rounded focus:outline-none focus:shadow-outline"
                        type="button"
                      >
                        Record New
                      </button>
                    </div>
                  </div>
                </div>
                {/*<div style={{position: 'absolute', bottom: 0, color: 'white', backgroundColor: 'black', padding: '1rem', marginBottom: '1rem'}} className="text-center text-3xl font-bold font-sans">*/}
                {/*    {prediction}*/}
                {/*</div>*/}
              </div>
            </LoadingOverlay>
          </div>
        </div>
      </PageWrapperMain>
    </>
  );
}

export default Collection;
