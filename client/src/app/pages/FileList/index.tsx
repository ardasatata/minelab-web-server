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
import { useEffect, useState } from 'react';
import { ArrowUpOutlined, CloseOutlined } from '@ant-design/icons';
import axios from 'axios';
import { Link } from '../../components/Link';
import { NavBar } from '../../components/NavBar';

interface Props {}

const fileTypes = ['MP4', 'MOV'];

export function FileList(props: Props) {
  const [file, setFile] = useState<any>(null);
  const handleChange = async file => {
    setFile(file);
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
      'https://140.115.51.243/api/send-video',
      formData,
      { headers },
    );

    console.log(result);
  };

  const [isLoading, setIsloading] = useState(false);

  const [datalist, setDatalist] = useState([]);

  useEffect(() => {
    const getFileList = async () => {
      setIsloading(true);

      const data = await axios.get('https://140.115.51.243/api/predict-list');

      console.log(data.data.filepath);
      setDatalist(data.data.filepath);

      console.log(data);
      setIsloading(false);
    };
    getFileList();
  }, []);

  return (
    <>
      <Helmet>
        <title>List</title>
        <meta
          name="description"
          content="A React Boilerplate application homepage"
        />
      </Helmet>
      <NavBar />
      <PageWrapperMain>
        <div className={'flex h-full w-full bg-black justify-center'}>
          <div className="flex flex-col max-h-screen mt-8">
            <div className="text-center text-5xl font-bold font-sans text-white">
              File List
            </div>
            <div className="text-center text-md font-light font-sans text-red-500 mb-4">
              *Some files might take some time to predict, Thanks for your
              patience & understanding
            </div>
            <div className="flex flex-col bg-white w-full overflow-y-auto">
              {datalist.map((item, index) => {
                return (
                  <Link to={process.env.PUBLIC_URL + `/playback?title=${item}`}>
                    <div className="px-4 py-2 cursor-pointer bg-white odd:bg-blue-100 text-sm font-light hover:bg-blue-500 hover:text-white">
                      {item}
                    </div>
                  </Link>
                );
              })}
            </div>
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
