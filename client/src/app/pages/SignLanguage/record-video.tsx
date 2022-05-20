/**
 *
 * RecordVideo
 *
 */
import * as React from 'react';
import styled from 'styled-components/macro';
import { StyleConstants } from '../../../styles/StyleConstants';
import { Helmet } from 'react-helmet-async';

import { useState } from 'react';

import axios from 'axios';
import { NavBar } from './NavBar';

import BounceLoader from 'react-spinners/BounceLoader';
import { sleep } from '../../../utils/sleep';

import LoadingOverlay from 'react-loading-overlay';
import { useAlert } from 'react-alert';
import VideoRecorder from 'react-video-recorder';
import Webcam from 'react-webcam';

import { Form, Switch, Input, Button } from 'antd';

import 'antd/dist/antd.css';

interface Props {}

const RECORDER_TIME_CONFIG = {
  countdownTime: 3000,
  timeLimit: 8000,
};

export function RecordVideo(props: Props) {
  const webcamRef = React.useRef(null);

  const [prediction, setPrediction] = useState(['Press Record First!']);

  const [isLoading, setIsloading] = useState(false);

  const [filename, setFilename] = useState(null);
  const [duration, setDuration] = useState(5);

  const [videoBlob, setVideoBlob] = useState(null);

  const alert = useAlert();

  const renderVideoRecorder = () => {
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
    setIsloading(true);
    const headers = {
      'Content-Type': 'multipart/form-data',
      'Access-Control-Allow-Origin': '*',
    };

    const formData = new FormData();
    // @ts-ignore
    formData.append('video', blob, `${filename}.mp4`);

    const result = await axios.post(
      'https://140.115.51.243/api/sign-language/send-video',
      formData,
      { headers },
    );

    console.log(result);

    if (result.data.ok) {
      // setIsUploaded(true);
      // setFile(null);
      setFilename(null);
      setVideoBlob(null);
      alert.success('File Uploaded Successfully!');
      alert.success(`Saved as ${result.data.filename}`);
      sleep(3000);
      setIsloading(false);
      window.location.reload();
    } else {
      setFilename(null);
      setVideoBlob(null);
      alert.error('Error Occurred');
      sleep(3000);
      setIsloading(false);
    }
  };

  const mediaRecorderRef = React.useRef(null);
  const [capturing, setCapturing] = React.useState(false);
  const [recordedChunks, setRecordedChunks] = React.useState([]);

  const [enableDuration, setEnableDuration] = React.useState(false);

  const handleStartCaptureClick = React.useCallback(() => {
    setCapturing(true);
    // @ts-ignore
    mediaRecorderRef.current = new MediaRecorder(webcamRef.current.stream, {
      mimeType: 'video/webm',
    });
    // @ts-ignore
    mediaRecorderRef.current.addEventListener(
      'dataavailable',
      handleDataAvailable,
    );
    // @ts-ignore
    mediaRecorderRef.current.start();

    alert.info('Recording Start ▶️');
  }, [webcamRef, setCapturing, mediaRecorderRef]);

  const handleDataAvailable = React.useCallback(
    ({ data }) => {
      if (data.size > 0) {
        setRecordedChunks(prev => prev.concat(data));
      }
    },
    [setRecordedChunks],
  );

  const handleStopCaptureClick = React.useCallback(async () => {
    // @ts-ignore
    mediaRecorderRef.current.stop();
    setCapturing(false);
    alert.info('Recording Stop 🛑');
    if (recordedChunks.length) {
      const blob = await new Blob(recordedChunks, {
        type: 'video/webm',
      });
      setVideoBlob(blob);
    }
  }, [mediaRecorderRef, webcamRef, setCapturing, recordedChunks]);

  const handleDownload = React.useCallback(() => {
    if (recordedChunks.length) {
      const blob = new Blob(recordedChunks, {
        type: 'video/webm',
      });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      document.body.appendChild(a);
      // @ts-ignore
      a.style = 'display: none';
      a.href = url;
      a.download = 'react-webcam-stream-capture.webm';
      a.click();
      window.URL.revokeObjectURL(url);
      setRecordedChunks([]);
    }
  }, [recordedChunks]);

  const handleUpload = React.useCallback(async () => {
    if (recordedChunks.length) {
      const blob = new Blob(recordedChunks, {
        type: 'video/webm',
      });
      // setVideoBlob(blob);
      await uploadFile(blob, filename);
    }
  }, [recordedChunks]);

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
                  <div className={'flex flex-col items-center'}>
                    {/*@ts-ignore*/}
                    <Webcam
                      // @ts-ignore
                      audio={false}
                      // @ts-ignore
                      ref={webcamRef}
                      mirrored={true}
                      hidden={false}
                      height={1920}
                      width={1080}
                    />
                  </div>
                </div>
                <div className="flex flex-col w-1/3">
                  <div className="max-w-md py-4 px-8 bg-white shadow-lg rounded-lg my-20 pb-8">
                    <div>
                      <h3 className="text-gray-800 text-3xl font-semibold mb-2">
                        Settings
                      </h3>

                      <Form
                        labelCol={{
                          span: 8,
                        }}
                        layout="horizontal"
                        size={'large'}
                      >
                        <Form.Item label="Enable Duration">
                          <Switch
                            defaultChecked={false}
                            onChange={checked => {
                              setEnableDuration(checked);
                            }}
                          />
                        </Form.Item>
                        {enableDuration ? (
                          <Form.Item label="Duration">
                            <Input
                              value={duration}
                              onChange={event => {
                                // @ts-ignore
                                setDuration(event.target.value);
                              }}
                            />
                          </Form.Item>
                        ) : null}

                        {capturing ? (
                          <Form.Item label="Record">
                            <Button
                              onClick={handleStopCaptureClick}
                              type={'primary'}
                              danger={true}
                            >
                              Stop
                            </Button>
                          </Form.Item>
                        ) : (
                          <Form.Item label="Record">
                            <Button
                              onClick={handleStartCaptureClick}
                              type={'primary'}
                            >
                              Start
                            </Button>
                          </Form.Item>
                        )}
                        {recordedChunks.length > 0 && (
                          <Form.Item label="Upload">
                            <Button onClick={handleUpload}>Upload</Button>
                          </Form.Item>
                        )}
                      </Form>
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

export const PageWrapperMain = styled.div`
  display: flex;
  margin: 0 auto;
  box-sizing: content-box;
  height: calc(100vh - ${StyleConstants.NAV_BAR_HEIGHT});
`;
