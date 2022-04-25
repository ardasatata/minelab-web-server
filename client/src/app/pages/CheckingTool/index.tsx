/**
 *
 * RecordVideo
 *
 */
import * as React from 'react';
import styled from 'styled-components/macro';
import { StyleConstants } from '../../../styles/StyleConstants';
import { PageWrapper } from '../../components/PageWrapper';
import { Helmet } from 'react-helmet-async';

import { FileUploader } from 'react-drag-drop-files';
import { useEffect, useState } from 'react';
import {
  ArrowUpOutlined,
  CloseOutlined,
  LoadingOutlined,
} from '@ant-design/icons';
import axios from 'axios';
import { NavBar } from '../../components/NavBar';
import guide from '../../../assets/guide.png';
import Webcam from 'react-webcam';
import { io } from 'socket.io-client';

interface Props {}

const FPS = 1;

export function CheckingTool(props: Props) {
  const webcamRef = React.useRef(null);
  const mediaRecorderRef = React.useRef(null);
  const [capturing, setCapturing] = React.useState(false);
  const [recordedChunks, setRecordedChunks] = React.useState([]);

  const [socket, setSocket] = useState<any>(null);
  const [webcamFrame, setWebcamFrame] = useState<any>(null);

  const [img, setImg] = useState<string>('');
  const [data, setData] = useState<any>(null);

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
  }, [webcamRef, setCapturing, mediaRecorderRef]);

  const handleDataAvailable = React.useCallback(
    ({ data }) => {
      if (data.size > 0) {
        setRecordedChunks(prev => prev.concat(data));
      }
    },
    [setRecordedChunks],
  );

  const handleStopCaptureClick = React.useCallback(() => {
    // @ts-ignore
    mediaRecorderRef.current.stop();
    setCapturing(false);
  }, [mediaRecorderRef, webcamRef, setCapturing]);

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

  const capture = React.useCallback(() => {
    // @ts-ignore
    const imageSrc = webcamRef.current.getScreenshot();
    if (imageSrc) {
      // console.log(imageSrc);
    }
    setWebcamFrame(imageSrc);
    // socket.emit('webcam-stream', imageSrc);
  }, [webcamRef, socket]);

  useEffect(() => {
    if (socket) {
      setInterval(() => {
        capture();
      }, 1000 / FPS);
    }
  }, [capture, setSocket]);

  // @ts-ignore
  useEffect(() => {
    const newSocket = io(`http://140.115.51.243:5000/stream-checking`, {
      transports: ['websocket'],
      upgrade: false,
    });
    // @ts-ignore
    newSocket.connect();
    setSocket(newSocket);

    newSocket.on('image', data => {
      const obj = JSON.parse(data);
      // setFrame(obj.frame);
      setData(obj.data);
      setImg('data:image/jpeg;base64,' + obj.image);

      // let labelsNew = labels;

      // labelsNew.push(obj.frame);
      // setLabels(Object.assign({}, labels, labelsNew));

      console.log(obj);
    });

    // setInterval(() => {
    //   capture();
    // }, 1000 / FPS);

    return () => newSocket.close();
  }, [setSocket]);

  useEffect(() => {
    if (webcamFrame) {
      socket.emit('webcam-stream', webcamFrame);
      // console.log(webcamFrame);
      // console.log(socket);
    }
  }, [socket, setSocket, webcamFrame]);

  const play = () => {
    // socket.emit('play', query.get('title'));
  };

  const stop = () => {
    socket.emit('stop');
    // eslint-disable-next-line no-restricted-globals
    location.reload();
  };

  // @ts-ignore
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
        <div className={'flex h-full w-full bg-black'}>
          <div
            className={
              'flex flex-col items-center justify-center text-white w-full'
            }
          >
            <div className={'text-sm text-black'}>
              chrome://flags/#unsafely-treat-insecure-origin-as-secure
              <div>http://140.115.51.243:3000/checking</div>
            </div>
            <div className={'flex flex-row'}>
              <div className={'flex flex-col items-center flex-1'}>
                <h1>SERVER</h1>
                {data ? (
                  <img
                    src={img}
                    alt={'main-stream'}
                    className={'object-contain'}
                    // style={{
                    //   aspectRatio: '16/9',
                    // }}
                  />
                ) : (
                  <div className={'text-white text-9xl m-auto'} onClick={play}>
                    <LoadingOutlined />
                  </div>
                )}
              </div>

              <div className={'flex flex-col items-center flex-1'}>
                <h1>ORIGIN</h1>
                {/*@ts-ignore*/}
                <Webcam
                  // @ts-ignore
                  audio={false}
                  // @ts-ignore
                  ref={webcamRef}
                  mirrored={true}
                />
              </div>
            </div>

            {/*{capturing ? (*/}
            {/*  <button onClick={handleStopCaptureClick}>Stop Capture</button>*/}
            {/*) : (*/}
            {/*  <button onClick={handleStartCaptureClick}>Start Capture</button>*/}
            {/*)}*/}
            {/*{recordedChunks.length > 0 && (*/}
            {/*  <button onClick={handleDownload}>Download</button>*/}
            {/*)}*/}
          </div>
        </div>
        {/*<div className={'h-full flex bg-white w-1/4'}>cok</div>*/}
      </PageWrapperMain>
    </>
  );
}

const Wrapper = styled.header`
  box-shadow: 0 1px 0 0 ${p => p.theme.borderLight};
  height: ${StyleConstants.NAV_BAR_HEIGHT};
  display: flex;
  position: fixed;
  top: 0;
  width: 100%;
  background-color: ${p => p.theme.background};
  z-index: 2;

  @supports (backdrop-filter: blur(10px)) {
    backdrop-filter: blur(10px);
    background-color: ${p =>
      p.theme.background.replace(
        /rgba?(\(\s*\d+\s*,\s*\d+\s*,\s*\d+)(?:\s*,.+?)?\)/,
        'rgba$1,0.75)',
      )};
  }

  ${PageWrapper} {
    display: flex;
    align-items: center;
    justify-content: space-between;
  }
`;

export const PageWrapperMain = styled.div`
  display: flex;
  margin: 0 auto;
  box-sizing: content-box;
  height: calc(100vh - ${StyleConstants.NAV_BAR_HEIGHT});
`;
