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
import { useState } from 'react';
import {
  ArrowUpOutlined,
  CloseOutlined,
  LoadingOutlined,
} from '@ant-design/icons';
import axios from 'axios';
import { NavBar } from './NavBar';

import guide from './assets/csl-wisnu.gif';

interface Props {}

const fileTypes = ['MP4', 'MOV', 'webm'];

export function UploadVideo(props: Props) {
  const [file, setFile] = useState<any>(null);

  const [isUploaded, setIsUploaded] = useState<boolean>(false);
  const [isError, setIsError] = useState<boolean>(false);

  const [isLoading, setIsloading] = useState<boolean>(false);

  const handleChange = async file => {
    setFile(file);
    setIsUploaded(false);
    setIsError(false);
  };

  const clearFile = () => {
    setFile(null);
  };

  const uploadFile = async () => {
    setIsloading(true);
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
      'https://140.115.51.243/api/sign-language/send-video',
      formData,
      { headers },
    );

    console.log(result);

    if (result.data.ok) {
      setIsUploaded(true);
      setFile(null);
      setIsloading(false);
    } else {
      setIsError(true);
      setIsUploaded(false);
      setIsloading(false);
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
        {isLoading ? (
          <div
            className={
              'flex flex-col text-white text-7xl m-auto items-center justify-center'
            }
          >
            <LoadingOutlined />
            <div className={'text-6xl max-w-3xl text-center'}>
              <h1 className={'text-5xl mb-12 text-center text-red-500 mt-12'}>
                Please wait...
              </h1>
            </div>
          </div>
        ) : (
          <div className={'flex h-full w-full bg-black'}>
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
                <div className={'flex flex-col items-center mb-12'}>
                  <img src={guide} className={'max-w-4xl mb-4'} />
                </div>
              )}

              {file !== null ? (
                <h1 className={'text-5xl mb-12 mt-12'}>
                  Your file is ready to upload!
                </h1>
              ) : (
                <h1 className={'text-2xl mb-4 text-amber-50 mt-12'}>
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
        )}

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
