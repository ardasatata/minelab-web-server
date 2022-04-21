/**
 *
 * RecordVideo
 *
 */
import * as React from 'react';
import styled from 'styled-components/macro';
import { StyleConstants } from '../../../styles/StyleConstants';
import { PageWrapper } from '../../components/PageWrapper';
import { Logo } from '../../components/NavBar/Logo';
import { Nav } from '../../components/NavBar/Nav';
import { Helmet } from 'react-helmet-async';
import { Masthead } from '../HomePage/Masthead';
import { Features } from '../HomePage/Features';
import VideoRecorder from 'react-video-recorder';

import { FileUploader } from 'react-drag-drop-files';
import { useState } from 'react';
import {
  ArrowUpOutlined,
  ClearOutlined,
  CloseOutlined,
  PlayCircleOutlined,
} from '@ant-design/icons';
import axios from 'axios';
import { NavBar } from '../../components/NavBar';
import guide from '../../../assets/guide.png';

interface Props {}

const fileTypes = ['MP4', 'MOV'];

export function RecordVideo(props: Props) {
  const [file, setFile] = useState<any>(null);

  const [isUploaded, setIsUploaded] = useState<boolean>(false);
  const [isError, setIsError] = useState<boolean>(false);

  const handleChange = async file => {
    setFile(file);
    setIsUploaded(false);
    setIsError(false);
  };

  const clearFile = () => {
    setFile(null);
  };

  const uploadFile = async () => {
    let videoFile = file[0];

    var videoUrl = URL.createObjectURL(videoFile);

    console.log(videoUrl);

    let blob = await fetch(videoUrl).then(r => r.blob());

    console.log(blob);

    // var reader = new FileReader();

    // @ts-ignore
    // console.log(reader.read(blobVideo))

    const headers = {
      'Content-Type': 'multipart/form-data',
      'Access-Control-Allow-Origin': '*',
    };

    const formData = new FormData();
    // @ts-ignore
    formData.append('video', blob, file[0].name);

    const result = await axios.post(
      'http://140.115.51.243:5000/send-video',
      formData,
      { headers },
    );

    console.log(result);

    if (result.data.ok) {
      setIsUploaded(true);
      setFile(null);
    } else {
      setIsError(true);
      setIsUploaded(false);
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
        <div className={'flex h-full w-full bg-black'}>
          {/*<VideoRecorder*/}
          {/*  isReplayingVideo={false}*/}
          {/*  showReplayControls={true}*/}
          {/*  isOnInitially={true}*/}
          {/*  // countdownTime={RECORDER_TIME_CONFIG.countdownTime}*/}
          {/*  // timeLimit={RECORDER_TIME_CONFIG.timeLimit}*/}
          {/*  mimeType="video/webm;codecs=vp8,opus"*/}
          {/*  // onRecordingComplete={async (videoBlob) => {*/}
          {/*  //   // Do something with the video...*/}
          {/*  //   // setPrediction(['Uploading your video'])*/}
          {/*  //   await submitVideo(videoBlob)*/}
          {/*  //   sum(5, 6)*/}
          {/*  //   console.log('videoBlob', videoBlob)*/}
          {/*  // }}*/}
          {/*  // onStartRecording={() => {*/}
          {/*  //   setPrediction(['Recording video...'])*/}
          {/*  // }}*/}
          {/*  // onTurnOnCamera={*/}
          {/*  //   () => {*/}
          {/*  //     setPrediction(['Press Record First!'])*/}
          {/*  //   }*/}
          {/*  // }*/}
          {/*/>*/}
          <div
            className={
              'flex flex-col items-center justify-center text-white w-full'
            }
          >
            {isUploaded ? (
              <>
                <h1 className={'text-4xl mb-2'}>Your file is uploaded!</h1>
                <h1 className={'text-2xl mb-12 text-center text-teal-300'}>
                  We appreciate your contribution to allow us for using your
                  video file for research purposes.
                </h1>
              </>
            ) : (
              <div className={'flex flex-col items-center'}>
                <h1 className={'text-4xl mb-4 text-amber-50'}>
                  Make sure to follow this guideline below before you record a
                  video!
                </h1>
                <img src={guide} className={'max-w-4xl'} />
              </div>
            )}

            {file !== null ? (
              <h1 className={'text-5xl mb-12'}>
                Your file is ready to upload!
              </h1>
            ) : (
              <h1 className={'text-2xl mb-4 text-amber-50'}>
                Please Drag a video file below!
              </h1>
            )}
            {file !== null ? (
              <div className={'flex flex-row'}>
                <div
                  className={
                    'px-4 text-white text-4xl py-4 transition ease-in-out bg-teal-500 hover:bg-indigo-700 duration-300 items-center justify-center flex cursor-pointer'
                  }
                  onClick={uploadFile}
                >
                  <ArrowUpOutlined className={'mr-2'} />
                  {'UPLOAD'}
                </div>
                <div
                  className={
                    'ml-4 px-4 text-white text-4xl py-4 transition ease-in-out bg-red-400 hover:bg-indigo-700 duration-300 items-center justify-center flex cursor-pointer'
                  }
                  onClick={clearFile}
                >
                  <CloseOutlined className={'mr-2'} />
                  {'CLEAR'}
                </div>
              </div>
            ) : (
              <FileUploader
                multiple={true}
                handleChange={handleChange}
                name="file"
                types={fileTypes}
              />
            )}
            <p className={'mt-6 text-amber-200'}>
              {file ? `File name: ${file[0].name}` : 'no files uploaded yet'}
            </p>
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
